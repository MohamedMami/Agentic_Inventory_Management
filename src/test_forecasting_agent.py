import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from datetime import datetime
import pandas as pd
from src.agents.ArimaForecast_agent import ForecastAgent
from src.database import Session

def test_forecast_agent():
    # Initialize the agent and database session
    agent = ForecastAgent()
    session = Session()
    
    try:
        # Test different types of forecast queries
        test_queries = [
            "Show me the sales forecast for Paracetamol for the next 30 days",
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Testing query: {query}")
            print(f"{'='*50}")
            
            # Process the query
            result = agent.process_query(query, session)
            
            # Print results
            if 'error' in result:
                print(f"Error: {result['error']}")
                print(f"Response: {result['response']}")
            else:
                print(f"Response: {result['response']}")
                if 'data' in result:
                    data = result['data']
                    print("\nProduct Info:")
                    print(f"- Name: {data['product_info']['name']}")
                    print(f"- Category: {data['product_info']['category']}")
                    
                    print("\nForecast Statistics:")
                    print(f"- ARIMA Order: {data['statistics']['arima_order']}")
                    print(f"- Mean Forecast: {data['statistics']['mean']:.2f}")
                    
                    print("\nFirst 5 Forecast Values:")
                    for i, value in enumerate(data['forecast_values'][:5]):
                        print(f"- {value['date']}: {value['forecast']:.2f} "+
                              f"({value['lower']:.2f} - {value['upper']:.2f})")
                    
                    print("\nInsights:")
                    print(data['insights'])
                    
                    if data.get('visualization_base64'):
                        print("\nVisualization generated successfully")
                        
                        # Optionally save the visualization
                        if not os.path.exists('test_outputs'):
                            os.makedirs('test_outputs')
                        
                        import base64
                        img_data = base64.b64decode(data['visualization_base64'])
                        filename = f"test_outputs/forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        with open(filename, 'wb') as f:
                            f.write(img_data)
                        print(f"Visualization saved to: {filename}")
            
            print(f"\n{'='*50}")
    
    finally:
        session.close()

if __name__ == "__main__":
    test_forecast_agent()