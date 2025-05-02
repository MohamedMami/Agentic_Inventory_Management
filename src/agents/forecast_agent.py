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
from sqlalchemy import text
from sqlalchemy.orm import Session
from statsmodels.tsa.arima.model import ARIMA
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .base_agent import BaseAgent

# Configure logging
logger = logging.getLogger(__name__)

class ForecastParameters(BaseModel):
    """Parameters for the ARIMA forecast."""
    product_id: Optional[str] = Field(None, description="ID of the product to forecast")
    product_name: Optional[str] = Field(None, description="Name of the product to forecast")
    category: Optional[str] = Field(None, description="Product category to forecast")
    horizon: int = Field(30, description="Number of days to forecast")
    p: int = Field(1, description="ARIMA parameter: autoregressive order")
    d: int = Field(1, description="ARIMA parameter: differencing order")
    q: int = Field(1, description="ARIMA parameter: moving average order")
    confidence_interval: float = Field(0.95, description="Confidence interval for forecast")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "product_name": "Paracetamol",
                    "horizon": 30,
                    "p": 1,
                    "d": 1,
                    "q": 1
                }
            ]
        }

class ForecastQueryParser(BaseModel):
    """Parser for forecast queries."""
    parameters: ForecastParameters = Field(..., description="Extracted forecast parameters")
    thought_process: str = Field(..., description="Reasoning behind parameter extraction")
    is_valid_query: bool = Field(..., description="Whether this is a valid forecast query")
    error_message: Optional[str] = Field(None, description="Error message if query is invalid")

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "parameters": {
                        "product_name": "Paracetamol",
                        "horizon": 30,
                        "p": 1,
                        "d": 1,
                        "q": 1
                    },
                    "thought_process": "Query asks for Paracetamol forecast for default horizon",
                    "is_valid_query": True,
                    "error_message": None
                }
            ]
        }

class ForecastResponse(BaseModel):
    """Structured forecast response."""
    product_info: Dict[str, Any] = Field(..., description="Information about the forecasted product")
    forecast_values: List[Dict[str, Any]] = Field(..., description="Forecasted values with dates")
    statistics: Dict[str, Any] = Field(..., description="Statistical information about the forecast")
    visualization_base64: Optional[str] = Field(None, description="Base64 encoded visualization")
    insights: str = Field(..., description="NL insights about the forecast")

class ForecastAgent(BaseAgent):
    """
    Agent specialized in pharmaceutical product demand forecasting using ARIMA,
    with real DB integration and conditional retraining.
    """
    MODELS_DIR = "models"
    META_PATH = os.path.join(MODELS_DIR, "metadata.json")
    RETRAIN_THRESHOLD_DAYS = 30

    def __init__(self):
        super().__init__()
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        self.query_parser = PydanticOutputParser(pydantic_object=ForecastQueryParser)
        self._init_chains()
        
        # Updated template to enforce direct JSON response
        self.parsing_template = """You are an expert in pharmaceutical sales forecasting. Your task is to extract forecasting parameters from user queries.

                    Query: {query}

                    IMPORTANT: Respond ONLY with a valid JSON object. No additional text, no XML-like tags, no explanations.
                    Use this exact structure:
                    {{
                        "parameters": {{
                            "product_id": null,
                            "product_name": "<product_name>",
                            "category": "<category>",
                            "horizon": <days>,
                            "p": <p_value>,
                            "d": <d_value>,
                            "q": <q_value>,
                            "confidence_interval": 0.95
                        }},
                        "thought_process": "Brief explanation of parameter choices",
                        "is_valid_query": true,
                        "error_message": null
                    }}

                    Here's a valid example:
                    {example_response}"""

    def _init_chains(self):
        # System messages for different tasks
        self.parsing_system_message = """
        You are an expert in pharmaceutical sales forecasting.
        Extract forecasting parameters from user queries: product (ID/name/category), horizon days,
        optional ARIMA(p,d,q), and confidence interval (default 0.95).
        """

        self.insights_system_message = """
        You are an expert in pharmaceutical inventory management. Given forecast data and stats,
        provide concise, actionable insights on trend, anomalies, inventory implications and risks.
        """

    def _parse_query(self, query: str) -> ForecastQueryParser:
        try:
            example = {
                "parameters": {
                    "product_name": "Paracetamol",
                    "product_id": None,
                    "category": None,
                    "horizon": 30,
                    "p": 1,
                    "d": 1,
                    "q": 1,
                    "confidence_interval": 0.95
                },
                "thought_process": "Standard forecast request for Paracetamol using default parameters",
                "is_valid_query": True,
                "error_message": None
            }
            
            prompt = self.parsing_template.format(
                query=query,
                example_response=json.dumps(example, indent=2)
            )
            
            response = self._get_llm_response(prompt, system_message="You are a JSON-only response generator. No additional formatting or tags allowed.")
            
            # Clean the response
            json_str = response.strip()
            if '{' not in json_str:
                raise ValueError("Response does not contain JSON")
                
            # Extract JSON content
            start = json_str.find('{')
            end = json_str.rindex('}') + 1
            clean_json = json_str[start:end]
            
            parsed_json = json.loads(clean_json)
            return ForecastQueryParser(**parsed_json)
                
        except Exception as e:
            logger.error(f"Query parsing error: {str(e)}\nRaw response: {response}")
            return ForecastQueryParser(
                parameters=ForecastParameters(),
                thought_process="Failed to parse response",
                is_valid_query=False,
                error_message=f"Failed to parse LLM response: {str(e)}"
            )

    def _generate_insights(self, product_name: str, forecast_data: List[Dict[str, Any]], 
                         statistics: Dict[str, Any]) -> str:
        try:
            prompt = f"""
            Product: {product_name}
            Forecast Data: {json.dumps(forecast_data)}
            Statistics: {json.dumps(statistics)}
            """
            return self._get_llm_response(prompt, system_message=self.insights_system_message)
        except Exception as e:
            logger.error(f"Insights generation failed: {e}")
            return "Insights unavailable"

    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        try:
            parsed = self._parse_query(query)
            if not parsed.is_valid_query:
                return {'error': parsed.error_message or 'Invalid forecast query', 
                        'response': parsed.error_message}

            params = parsed.parameters
            df = self._fetch_historical_data(session, params)
            if df.empty:
                return {'error': 'No data', 'response': 'No historical data found.'}

            result = self._generate_forecast(df, params)
            insights = self._generate_insights(
                result['product_info']['name'], 
                result['forecast_values'], 
                result['statistics']
            )
            result['insights'] = insights
            return {'response': 'Forecast generated.', 'data': result}

        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return {'error': str(e), 'response': 'Forecast failed.'}

    def _load_metadata(self) -> Dict[str, Any]:
        if os.path.exists(self.META_PATH):
            return json.load(open(self.META_PATH))
        return {}

    def _save_metadata(self, meta: Dict[str, Any]):
        json.dump(meta, open(self.META_PATH, 'w'), indent=2)

    def _train_model(self, ts: pd.Series, key: str, params: ForecastParameters):
        model = ARIMA(ts, order=(params.p, params.d, params.q)).fit()
        now = ts.index.max().date().isoformat()
        model_path = os.path.join(self.MODELS_DIR, f"{key}_{params.p}{params.d}{params.q}.pkl")
        joblib.dump(model, model_path)
        return model, {'last_trained': now, 'model_path': model_path}

    def _load_or_update_model(self, ts: pd.Series, key: str, params: ForecastParameters):
        meta = self._load_metadata()
        entry = meta.get(key)
        last_date = ts.index.max().date()
        if entry:
            trained = datetime.fromisoformat(entry['last_trained']).date()
            if (last_date - trained).days <= self.RETRAIN_THRESHOLD_DAYS and os.path.exists(entry['model_path']):
                logger.info(f"Loading existing model for {key}, trained on {entry['last_trained']}")
                return joblib.load(entry['model_path'])
        # retrain
        logger.info(f"Retraining model for {key}")
        model, m = self._train_model(ts, key, params)
        meta[key] = m
        self._save_metadata(meta)
        return model

    def _fetch_historical_data(self, session: Session, params: ForecastParameters) -> pd.DataFrame:
        try:
            # Build the base query
            query = """
            SELECT 
                DATE(s.sale_date) as date,
                SUM(s.quantity) as total_sales
            FROM sales s
            JOIN products p ON s.product_id = p.id
            WHERE 1=1
            """
            
            query_params = {}
            
            # Add conditions based on parameters
            if params.product_name:
                query += " AND p.name ILIKE :product_name"
                query_params['product_name'] = f"%{params.product_name}%"
            
            if params.product_id:
                query += " AND p.id = :product_id"
                query_params['product_id'] = params.product_id
                
            if params.category:
                query += " AND p.category ILIKE :category"
                query_params['category'] = f"%{params.category}%"
                
            # Group by date to handle duplicates
            query += " GROUP BY DATE(s.sale_date) ORDER BY date"
            
            # Execute query
            result = session.execute(text(query), query_params)
            df = pd.DataFrame(result.fetchall(), columns=['date', 'total_sales'])
            
            # Ensure date is index and handle any remaining duplicates
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Resample to daily frequency and fill missing values
            df = df.resample('D').sum()
            df = df.fillna(0)
            
            if df.empty:
                raise ValueError("No data found for the given parameters")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise

    def _generate_forecast(self, data: pd.DataFrame, params: ForecastParameters) -> Dict[str, Any]:
        try:
            # Fit ARIMA model
            model = ARIMA(data['total_sales'], 
                         order=(params.p, params.d, params.q))
            results = model.fit()
            
            # Generate forecast
            forecast = results.get_forecast(steps=params.horizon)
            mean_forecast = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=1-params.confidence_interval)
            
            # Create date range for forecast
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                         periods=params.horizon,
                                         freq='D')
            
            # Prepare forecast values
            forecast_values = []
            for date, pred, lower, upper in zip(forecast_dates, 
                                              mean_forecast, 
                                              conf_int.iloc[:, 0], 
                                              conf_int.iloc[:, 1]):
                forecast_values.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'forecast': float(pred),
                    'lower': float(lower),
                    'upper': float(upper)
                })
            
            # Calculate statistics
            statistics = {
                'arima_order': (params.p, params.d, params.q),
                'mean': float(mean_forecast.mean()),
                'std': float(mean_forecast.std()),
                'aic': results.aic,
                'bic': results.bic
            }
            
            return {
                'forecast_values': forecast_values,
                'statistics': statistics
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise

    def _create_visualization(self, hist, mean, ci, dates, name):
        plt.figure(figsize=(10,6))
        plt.plot(hist.index, hist.values, label='History')
        plt.plot(dates, mean, label='Forecast')
        plt.fill_between(dates, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
        plt.title(f"Forecast for {name}")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close()
        return img
