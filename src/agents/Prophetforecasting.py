import logging
import json
import os
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)

class ProphetForecastAgent(BaseAgent):
    """
    Agent specialized in pharmaceutical product demand forecasting using FB Prophet,
    with real DB integration and conditional retraining.
    """
    MODELS_DIR = "models"
    META_PATH = os.path.join(MODELS_DIR, "metadata.json")
    RETRAIN_THRESHOLD_DAYS = 30

    def __init__(self):
        super().__init__()
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        
        # Define system messages for different tasks
        self.parsing_system_message = """
        You are an expert in pharmaceutical sales forecasting.
        Extract forecasting parameters from user queries: product (product_id/product_name/category), horizon days,
        optional Prophet parameters (seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality),
        and confidence interval (default 0.95).
        
        Return your response in JSON format with these keys:
        - parameters: containing product_id, product_name, category, horizon, seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality, confidence_interval
        - thought_process: your reasoning
        - is_valid_query: true/false
        - error_message: error if any
        """

        self.insights_system_message = """
        You are an expert in pharmaceutical inventory management. Given forecast data and stats,
        provide concise, actionable insights on trend, anomalies, inventory implications and risks.
        Focus especially on:
        1. Overall trend direction and magnitude
        2. Seasonal patterns and their business implications
        3. Potential stock-out risks and when they might occur
        4. Suggested inventory adjustments based on the forecast
        Keep your insights practical and focused on business outcomes.
        """

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse user query to extract forecast parameters.
        """
        try:
            example = {
                "parameters": {
                    "product_name": "Paracetamol",
                    "product_id": None,
                    "category": None,
                    "horizon": 30,
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "confidence_interval": 0.95
                },
                "thought_process": "Standard forecast request for Paracetamol using default parameters",
                "is_valid_query": True,
                "error_message": None
            }
            
            prompt = f"""
            Extract forecasting parameters from this user query: "{query}"
            
            Respond with a valid JSON object using this structure:
            {json.dumps(example, indent=2)}
            
            Make sure to set is_valid_query to false if essential parameters like product identification are missing.
            """
            
            response = self._get_llm_response(prompt, system_message=self.parsing_system_message)
            
            # Extract JSON content
            json_str = response.strip()
            if '{' not in json_str:
                raise ValueError("Response does not contain JSON")
                
            start = json_str.find('{')
            end = json_str.rindex('}') + 1
            clean_json = json_str[start:end]
            
            parsed_json = json.loads(clean_json)
            return parsed_json
                
        except Exception as e:
            logger.error(f"Query parsing error: {str(e)}\nRaw response: {response}")
            return {
                "parameters": {
                    "product_name": None,
                    "product_id": None,
                    "category": None,
                    "horizon": 30,
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "confidence_interval": 0.95
                },
                "thought_process": "Failed to parse response",
                "is_valid_query": False,
                "error_message": f"Failed to parse LLM response: {str(e)}"
            }

    def _generate_insights(self, product_name: str, forecast_data: List[Dict[str, Any]], 
                         statistics: Dict[str, Any], metrics: Dict[str, float]) -> str:
        """
        Generate natural language insights about the forecast.
        """
        try:
            prompt = f"""
            Product: {product_name}
            Forecast Data: {json.dumps(forecast_data[:5])}... (showing first 5 days of {len(forecast_data)} total)
            Statistics: {json.dumps(statistics)}
            Evaluation Metrics: {json.dumps(metrics)}
            
            Based on this forecast data, provide concise, actionable insights on:
            1. Overall trend direction and magnitude
            2. Seasonal patterns (if any) and their business implications
            3. Potential stock-out risks and when they might occur
            4. Suggested inventory adjustments based on the forecast
            
            Keep your insights practical and focused on business outcomes.
            """
            return self._get_llm_response(prompt, system_message=self.insights_system_message)
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return "Insights unavailable"

    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """
        Process a user query to generate a forecast.
        Inherits from BaseAgent's abstract method.
        """
        try:
            parsed = self._parse_query(query)
            if not parsed["is_valid_query"]:
                return {'error': parsed["error_message"] or 'Invalid forecast query', 
                        'response': parsed["error_message"]}

            params = parsed["parameters"]
            df = self._fetch_historical_data(session, params)
            if df.empty:
                return {'error': 'No data', 'response': 'No historical data found.'}

            result = self._generate_forecast(df, params)
            insights = self._generate_insights(
                result['product_info']['name'], 
                result['forecast_values'], 
                result['statistics'],
                result['evaluation_metrics']
            )
            result['insights'] = insights
            return {'response': 'Forecast generated.', 'data': result}

        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return {'error': str(e), 'response': 'Forecast failed.'}

    def _load_metadata(self) -> Dict[str, Any]:
        """Load model metadata from disk."""
        if os.path.exists(self.META_PATH):
            with open(self.META_PATH, 'r') as f:
                return json.load(f) 
        return {}

    def _save_metadata(self, meta: Dict[str, Any]):
        """Save model metadata to disk."""
        with open(self.META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)

    def _train_model(self, df: pd.DataFrame, key: str, params: Dict[str, Any]):
        """Train a new Prophet model with the given parameters."""
        try:
            # Prepare data for Prophet
            prophet_df = df.copy()
            
            # Reset index to get the date as a column and rename it
            prophet_df = prophet_df.reset_index()
            prophet_df.columns = prophet_df.columns.str.lower()
            
            # Debug logging
            logger.debug(f"Columns before rename: {prophet_df.columns.tolist()}")
            logger.debug(f"DataFrame head before rename:\n{prophet_df.head()}")
            
            # Check and rename columns for Prophet
            if 'index' in prophet_df.columns:
                prophet_df = prophet_df.rename(columns={'index': 'ds'})
            elif 'date' in prophet_df.columns:
                prophet_df = prophet_df.rename(columns={'date': 'ds'})
            else:
                raise ValueError(f"No date column found. Available columns: {prophet_df.columns.tolist()}")
            
            prophet_df = prophet_df.rename(columns={'total_sales': 'y'})
            
            # Verify required columns
            if 'ds' not in prophet_df.columns or 'y' not in prophet_df.columns:
                raise ValueError(f"Missing required columns. Available columns: {prophet_df.columns.tolist()}")
            
            # Debug logging
            logger.debug(f"Columns after rename: {prophet_df.columns.tolist()}")
            logger.debug(f"Prophet DataFrame head:\n{prophet_df[['ds', 'y']].head()}")
            
            # Configure and train model
            model = Prophet(
                seasonality_mode=params["seasonality_mode"],
                yearly_seasonality=params["yearly_seasonality"],
                weekly_seasonality=params["weekly_seasonality"],
                daily_seasonality=params["daily_seasonality"],
                interval_width=params["confidence_interval"]
            )
            
            model.fit(prophet_df)
            
            # Save model
            now = prophet_df['ds'].max().date().isoformat()
            model_path = os.path.join(self.MODELS_DIR, f"{key}_prophet.pkl")
            with open(model_path, 'wb') as f:
                joblib.dump(model, f)
            
            return model, {'last_trained': now, 'model_path': model_path}
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise

    def _load_or_update_model(self, df: pd.DataFrame, key: str, params: Dict[str, Any]):
        """Load an existing model or train a new one if needed."""
        meta = self._load_metadata()
        entry = meta.get(key)
        last_date = df.index.max().date()
        
        if entry:
            trained = datetime.fromisoformat(entry['last_trained']).date()
            if (last_date - trained).days <= self.RETRAIN_THRESHOLD_DAYS and os.path.exists(entry['model_path']):
                logger.info(f"Loading existing model for {key}, trained on {entry['last_trained']}")
                with open(entry['model_path'], 'rb') as f:
                    return joblib.load(f)
                
        # Need to retrain
        logger.info(f"Training new Prophet model for {key}")
        model, m = self._train_model(df, key, params)
        meta[key] = m
        self._save_metadata(meta)
        return model

    def _fetch_historical_data(self, session: Session, params: Dict[str, Any]) -> pd.DataFrame:
        """Fetch and preprocess historical sales data from the database."""
        try:
            self.validate_data_integrity(session)
            
            # Enhanced SQL query with more features
            sql = """
                SELECT 
                    s.sale_date AS date,  -- Keep as 'date' for consistency
                    SUM(s.quantity) AS total_sales,
                    MIN(s.product_name) as product_name,
                    MIN(s.category) as category,
                    MIN(p.min_stock_level) as min_stock_level,
                    MIN(p.reorder_lead_time_days) as lead_time,
                    MIN(i.current_quantity) as current_stock,
                    COUNT(DISTINCT s.facility_id) as num_facilities,
                    AVG(s.unit_price) as avg_unit_price,
                    SUM(s.total_value) as total_value,
                    bool_or(s.is_weekend) as is_weekend,
                    bool_or(s.is_holiday) as is_holiday
                FROM sales s
                LEFT JOIN products p ON s.product_id = p.product_id
                LEFT JOIN inventory i ON s.product_id = i.product_id
                WHERE 1=1
            """
            
            # Add filters
            if params.get("product_id"):
                sql += " AND s.product_id = :pid"
            elif params.get("product_name"):
                sql += " AND s.product_name ILIKE :pname"
            elif params.get("category"):
                sql += " AND s.category ILIKE :cat"
                
            # Add grouping
            sql += """ 
                GROUP BY s.sale_date::date 
                ORDER BY s.sale_date::date
            """
            
            # Execute query
            logger.debug(f"Executing query: {sql}")
            result = session.execute(text(sql), {
                'pid': params.get("product_id"),
                'pname': f"%{params.get('product_name')}%" if params.get("product_name") else None,
                'cat': f"%{params.get('category')}%" if params.get("category") else None
            })
            
            # Convert to DataFrame
            df = pd.DataFrame(result.fetchall(), columns=['date', 'total_sales', 'product_name', 'category', 
                                                        'min_stock_level', 'lead_time', 'current_stock', 
                                                        'num_facilities', 'avg_unit_price', 'total_value', 
                                                        'is_weekend', 'is_holiday'])
            if df.empty:
                logger.info("No historical data found for the given parameters.")
                return df
            
            # Preprocess data properly
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Ensure continuous daily data
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
            df = df.reindex(date_range)
            df['total_sales'] = df['total_sales'].fillna(0)  # Replace NaN with 0
            
            # Forward fill product info
            df['product_name'] = df['product_name'].ffill()
            df['category'] = df['category'].ffill()
            
            logger.info(f"DataFrame columns after fetch: {df.columns.tolist()}")
            logger.info(f"DataFrame head:\n{df.head()}")
            
            logger.info(f"DataFrame columns after preprocessing: {df.columns.tolist()}")
            logger.info(f"DataFrame head after preprocessing:\n{df.head()}")
            
            return df

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            session.rollback()
            raise

    def validate_data_integrity(self, session: Session):
        """Enhanced data validation checks."""
        try:
            checks = [
                ("Missing product links", 
                 "SELECT COUNT(*) FROM sales s WHERE NOT EXISTS (SELECT 1 FROM products p WHERE p.product_id = s.product_id)"),
                ("Products below min stock", 
                 "SELECT COUNT(*) FROM inventory i JOIN products p ON i.product_id = p.product_id WHERE i.current_quantity < p.min_stock_level"),
                ("Expired products", 
                 "SELECT COUNT(*) FROM inventory WHERE expiry_date < CURRENT_DATE"),
                ("Non-compliant storage", 
                 "SELECT COUNT(*) FROM inventory WHERE temperature_compliant = false")
            ]
            
            for check_name, query in checks:
                count = session.execute(text(query)).scalar()
                if count > 0:
                    logger.warning(f"{check_name}: {count} records affected")
                    
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            session.rollback()
            raise

    def _calculate_evaluation_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calculate forecast evaluation metrics."""
        # Handle NaN values that might be present
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        # If no valid pairs, return zeros
        if len(actual_clean) == 0:
            return {'mae': 0.0, 'mse': 0.0, 'rmse': 0.0, 'mape': 0.0}
        
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        
        # Calculate MAPE avoiding division by zero
        mask_nonzero = actual_clean != 0
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((actual_clean[mask_nonzero] - predicted_clean[mask_nonzero]) / actual_clean[mask_nonzero])) * 100
        else:
            mape = np.nan
            
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape) if not np.isnan(mape) else None
        }

    def _generate_forecast(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced forecast generation with inventory insights."""
        try:
            # Get product info from the data
            product_info = {
                'name': data['product_name'].iloc[0],
                'category': data['category'].iloc[0]
            }
            
            # Create a unique key for the model
            model_key = f"{product_info['name']}".lower().replace(' ', '_')
            
            # Load or train Prophet model
            model = self._load_or_update_model(data, model_key, params)
            
            # Get horizon (default to 30 if not specified)
            horizon = params.get("horizon", 30)
            
            # Prepare Prophet dataframe for forecast
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future)
            
            # Calculate evaluation metrics - only for historical period
            historical_dates = data.index
            historical_forecast = forecast[forecast['ds'].dt.date.isin([d.date() for d in historical_dates])]
            historical_forecast = historical_forecast.set_index('ds')
            
            # Align indices for evaluation
            aligned_actual = data.loc[historical_forecast.index, 'total_sales']
            aligned_pred = historical_forecast['yhat']
            
            metrics = self._calculate_evaluation_metrics(aligned_actual, aligned_pred)
            
            # Get forecast for future dates only
            future_forecast = forecast[~forecast['ds'].isin(historical_dates)]
            
            # Prepare forecast values
            forecast_values = []
            for _, row in future_forecast.iterrows():
                forecast_values.append({
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'forecast': float(row['yhat']),
                    'lower': float(row['yhat_lower']),
                    'upper': float(row['yhat_upper']),
                    'trend': float(row['trend']),
                })
            
            # Extract seasonality components if available
            seasonality_components = {}
            for col in forecast.columns:
                if any(col.startswith(prefix) for prefix in ['yearly', 'weekly', 'daily']):
                    seasonality_components[col] = float(future_forecast[col].mean())
            
            # Enhanced statistics
            statistics = {
                'trend': float(future_forecast['trend'].mean()),
                'trend_direction': 'increasing' if future_forecast['trend'].iloc[-1] > future_forecast['trend'].iloc[0] else 'decreasing',
                'seasonality_mode': params.get("seasonality_mode", "additive"),
                'mean_forecast': float(future_forecast['yhat'].mean()),
                'max_forecast': float(future_forecast['yhat'].max()),
                'min_forecast': float(future_forecast['yhat'].min()),
                'seasonality_components': seasonality_components,
                'inventory_metrics': {
                    'current_stock': float(data['current_stock'].iloc[-1]),
                    'min_stock_level': float(data['min_stock_level'].iloc[-1]),
                    'lead_time': int(data['lead_time'].iloc[-1]),
                    'stock_coverage_days': float(data['current_stock'].iloc[-1] / data['total_sales'].mean()),
                    'reorder_point': float(data['min_stock_level'].iloc[-1] * 1.5)
                }
            }
            
            # Create visualization
            viz = self._create_visualization(
                data['total_sales'],
                forecast,
                product_info['name']
            )
            
            return {
                'product_info': product_info,
                'forecast_values': forecast_values,
                'statistics': statistics,
                'evaluation_metrics': metrics,
                'visualization_base64': viz
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def _create_visualization(self, hist_series, forecast_df, name):
        """Create visualization for the forecast."""
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot historical data
        hist_df = hist_series.reset_index()
        hist_df.columns = ['ds', 'y']
        ax1.plot(hist_df['ds'], hist_df['y'], 'k.', label='Historical')
        
        # Plot forecast
        ax1.plot(forecast_df['ds'], forecast_df['yhat'], 'b-', label='Forecast')
        ax1.fill_between(forecast_df['ds'], 
                        forecast_df['yhat_lower'], 
                        forecast_df['yhat_upper'], 
                        color='b', alpha=0.2, label='Confidence Interval')
        
        # Mark the forecast start
        forecast_start = hist_df['ds'].max()
        ax1.axvline(x=forecast_start, color='r', linestyle='--', label='Forecast Start')
        
        # Plot components
        ax2.plot(forecast_df['ds'], forecast_df['trend'], 'g-', label='Trend')
        
        # Add weekly seasonality if present
        if 'weekly' in forecast_df.columns:
            ax2.plot(forecast_df['ds'], forecast_df['weekly'], 'c-', label='Weekly Seasonality')
        
        # Add yearly seasonality if present
        if 'yearly' in forecast_df.columns:
            ax2.plot(forecast_df['ds'], forecast_df['yearly'], 'm-', label='Yearly Seasonality')
        
        # Titles and legends
        ax1.set_title(f"FB Prophet Forecast for {name}")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_title("Components")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close()
        
        return img