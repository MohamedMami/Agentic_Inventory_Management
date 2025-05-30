from database import Session
from sqlalchemy.orm import sessionmaker
import asyncio
import os
from typing import Dict, Any
from redis.exceptions import ConnectionError as RedisConnectionError
import json
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List, Optional

# Import the supervisor agent and specialized agents
from agents.supervisorReAct import SupervisorReActAgent
from agents.inventory_agent import InventoryAgent
from agents.visualization_agent import VisualizationAgent
from agents.Prophetforecasting import ProphetForecastAgent

class TestResults:
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.start_time = datetime.now()
        
    def add_result(self, query: str, result: Dict[str, Any], duration: float):
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_type": result.get("query_type"),
            "success": not bool(result.get("error")),
            "error": result.get("error"),
            "duration_seconds": duration,
            "has_visualization": bool(result.get("visualization_base64")),
            "response_length": len(str(result.get("response", "")))
        })
    
    def save_results(self):
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        with open(self.output_dir / f"detailed_results_{timestamp}.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary DataFrame
        df = pd.DataFrame(self.results)
        summary = {
            "total_queries": len(df),
            "successful_queries": df["success"].sum(),
            "failed_queries": (~df["success"]).sum(),
            "average_duration": df["duration_seconds"].mean(),
            "queries_with_viz": df["has_visualization"].sum(),
            "by_query_type": df.groupby("query_type").size().to_dict()
        }
        
        # Save summary
        with open(self.output_dir / f"summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save as Excel for easy viewing
        df.to_excel(self.output_dir / f"test_results_{timestamp}.xlsx", index=False)
        
        return summary

async def main():
    test_results = TestResults()
    
    try:
        # Set up database connection
        session = Session()
        
        # Create the supervisor agent with llama-3.3-70b-versatile model
        supervisor = SupervisorReActAgent(
            model_name="llama-3.3-70b-versatile", 
            temperature=0.1,  # Low temperature for more consistent output
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        
        # Create and register specialized agents
        inventory_agent = InventoryAgent()
        visualization_agent = VisualizationAgent()
        forecasting_agent = ProphetForecastAgent()
        
        supervisor.register_agent("inventory", inventory_agent)
        supervisor.register_agent("visualization", visualization_agent)
        supervisor.register_agent("forecasting", forecasting_agent)
        # Comprehensive test queries
        queries = [
            "what are the top 5 selling products?",
            "show me the inventory levels for antibiotics",
            "forecast demand for Azithromycin 200mg/5ml Suspension for next month",
            "visualize monthly sales trends for 2024",
            "which products are below reorder point?",
            "what's the current value of inventory by category?"
        ]
        
        conversation_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for query in queries:
            try:
                print(f"\n=== Processing Query: {query} ===")
                start_time = time.time()
                
                result = await supervisor.process_query(query, session, conversation_id)
                duration = time.time() - start_time
                
                test_results.add_result(query, result, duration)
                
                # Ensure result is a dictionary
                if isinstance(result, str):
                    print(f"Error: Received string response instead of dictionary: {result}")
                    continue
                
                print(f"Query Type: {result.get('query_type', 'UNKNOWN')}")
                
                if result.get('query_type') == 'composite':
                    print("Integrated Response:")
                    print(result.get('integrated_response', 'No integrated response'))
                    if result.get('sub_responses'):
                        print("\nSub-responses:")
                        for sub_response in result.get('sub_responses', []):
                            if isinstance(sub_response, dict):
                                response_text = sub_response.get('result', {}).get('response', 'No response')
                                print(f"- {sub_response.get('query_type')}: {response_text[:100]}...")
                            else:
                                print(f"Invalid sub-response format: {sub_response}")
                else:
                    print(f"Response: {result.get('response', 'No response available')}")
                
                if result.get('error'):
                    print(f"Error: {result.get('error')}")
                    
            except Exception as e:
                print(f"Error processing query: {str(e)}")
                test_results.add_result(query, {"error": str(e)}, time.time() - start_time)
            
            finally:
                print("=" * 50)
        
        # Save and print summary
        summary = test_results.save_results()
        print("\n=== Test Summary ===")
        print(json.dumps(summary, indent=2))
            
    except RedisConnectionError:
        print("Error: Could not connect to Redis. Please ensure Redis server is running.")
        print("Run 'sudo service redis-server start' in WSL to start Redis")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    
    finally:
        if 'session' in locals():
            session.close()

if __name__ == "__main__":
    asyncio.run(main())