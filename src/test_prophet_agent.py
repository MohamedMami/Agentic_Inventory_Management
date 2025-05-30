import logging
from datetime import datetime
import os
from src.agents.Prophetforecasting import ProphetForecastAgent
from src.database import Session

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_prophet_forecasting():
    """Test the Prophet forecasting agent with sample queries."""
    
    # Initialize agent and database session
    agent = ProphetForecastAgent()
    session = Session()
    
    try:
        # Test queries
        test_queries = [
            "can you forecast Azithromycin 500mg Tablet for the next 90 days",
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
                viz_dir = 'forecast_visualizations'
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