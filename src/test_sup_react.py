from database import Session
from sqlalchemy.orm import sessionmaker
import asyncio
import os
from typing import Dict, Any
from redis.exceptions import ConnectionError as RedisConnectionError

# Import the supervisor agent and specialized agents
from agents.supervisorReAct import SupervisorReActAgent
from agents.inventory_agent import InventoryAgent
from agents.visualization_agent import VisualizationAgent

async def main():
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
        
        supervisor.register_agent("inventory", inventory_agent)
        supervisor.register_agent("visualization", visualization_agent)
        
        # Example queries remain the same
        queries = [
            "what are the top 5 selling products?",
            "Create a bar chart of our top 10 selling products this quarter",
        ]
        
        conversation_id = "test_conversation"
        
        for query in queries:
            try:
                print(f"\n=== Processing Query: {query} ===")
                result = await supervisor.process_query(query, session, conversation_id)
                
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
                
            finally:
                print("=" * 50)
            
    except RedisConnectionError:
        print("Error: Could not connect to Redis. Please ensure Redis server is running.")
        print("Run 'sudo service redis-server start' in WSL to start Redis")
    except Exception as e:
        print(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())