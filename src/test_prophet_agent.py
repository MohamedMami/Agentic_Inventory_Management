import logging
from datetime import datetime
import os
from src.agents.Prophetforecasting import ProphetForecastAgent
from src.database import Session
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_additional_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculate additional performance metrics."""
    try:
        # Add debug logging
        logger.info(f"Calculating metrics for arrays of shape: true={y_true.shape}, pred={y_pred.shape}")
        logger.info(f"Sample values - true: {y_true[:5]}, pred: {y_pred[:5]}")
        
        # Validate inputs
        if len(y_true) == 0 or len(y_pred) == 0:
            logger.error("Empty arrays provided for metric calculation")
            return {}
            
        if np.isnan(y_true).any() or np.isnan(y_pred).any():
            logger.error("Arrays contain NaN values")
            return {}

        # R2 Score
        r2 = r2_score(y_true, y_pred)
        logger.info(f"Calculated R2 score: {r2}")
        
        # Convert continuous values to binary for classification metrics
        y_mean = np.mean(y_true)
        y_true_binary = (y_true > y_mean).astype(int)
        y_pred_binary = (y_pred > y_mean).astype(int)
        
        # Calculate classification metrics with error handling
        try:
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary)
            recall = recall_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary)
        except Exception as e:
            logger.error(f"Error in classification metrics: {str(e)}")
            return {'r2_score': r2}  # Return at least R2 if classification metrics fail
        
        metrics = {
            'r2_score': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        logger.info(f"Calculated metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating additional metrics: {str(e)}")
        return {}

def test_prophet_forecasting():
    """Test the Prophet forecasting agent with sample queries."""
    
    # Initialize agent and database session
    agent = ProphetForecastAgent()
    session = Session()
    
    # Print agent configuration
    print(f"Models directory: {agent.MODELS_DIR}")
    print(f"Metadata path: {agent.META_PATH}")
    
    try:
        # Check if models directory exists
        if not os.path.exists(agent.MODELS_DIR):
            print(f"Warning: Models directory does not exist at {agent.MODELS_DIR}")
        else:
            print(f"Models directory exists at {agent.MODELS_DIR}")
    
        # Test queries
        test_queries = [
            "can you forecast Metformin 500mg Tablet for the next 90 days",
        ]
        
        # Process each query
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Testing Query: {query}")
            print(f"{'='*80}")
            
            # Get forecast
            result = agent.process_query(query, session)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                print(f"Response: {result['response']}")
                continue
                
            # Print forecast results
            data = result['data']
            
            # Calculate additional metrics if actual values are available
            if 'actual_values' in data and 'forecast_values' in data:
                try:
                    logger.info("Extracting actual and predicted values")
                    
                    # Extract and validate values
                    actual_values = [v.get('actual') for v in data['actual_values'] if v.get('actual') is not None]
                    predicted_values = [v.get('forecast') for v in data['forecast_values'] if v.get('forecast') is not None]
                    
                    if not actual_values or not predicted_values:
                        logger.error("No valid actual or predicted values found")
                        continue
                        
                    actual = np.array(actual_values, dtype=float)
                    predicted = np.array(predicted_values, dtype=float)
                    
                    # Calculate additional metrics using the agent's method
                    additional_metrics = agent.calculate_additional_metrics(actual, predicted)
                    
                    if additional_metrics:
                        print("\nAdditional Performance Metrics:")
                        for metric, value in additional_metrics.items():
                            print(f"{metric}: {value:.4f}")
                    else:
                        print("\nNo additional metrics could be calculated")
                        
                except Exception as e:
                    logger.error(f"Error processing actual/predicted values: {str(e)}")
    
            print("\nTraining Data Information:")
            print(f"Number of training samples: {data.get('training_info', {}).get('n_samples', 'N/A')}")
            print(f"Training period: {data.get('training_info', {}).get('date_range', 'N/A')}")
            
            print("\nProduct Information:")
            print(f"Name: {data['product_info']['name']}")
            print(f"Category: {data['product_info']['category']}")
            
            print("\nForecast Statistics:")
            stats = data['statistics']
            print(f"Trend Direction: {stats['trend_direction']}")
            print(f"Mean Forecast: {stats['mean_forecast']:.2f}")
            print(f"Range: {stats['min_forecast']:.2f} - {stats['max_forecast']:.2f}")
            
            print("\nEvaluation Metrics:")
            metrics = data['evaluation_metrics']
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.2f}")
            
            print("\nFirst 5 Forecast Values:")
            for value in data['forecast_values'][:5]:
                print(f"Date: {value['date']}")
                print(f"Forecast: {value['forecast']:.2f} ({value['lower']:.2f} - {value['upper']:.2f})")
            
            print("\nInsights:")
            print(data['insights'])
            
            # Save visualization if present
            if data.get('visualization_base64'):
                import base64
                viz_dir = 'data/forecasts'
                os.makedirs(viz_dir, exist_ok=True)
                
                filename = f"{viz_dir}/forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                with open(filename, 'wb') as f:
                    f.write(base64.b64decode(data['visualization_base64']))
                print(f"\nVisualization saved to: {filename}")
            
            print(f"\n{'='*80}")

    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    finally:
        session.close()

if __name__ == "__main__":
    test_prophet_forecasting()