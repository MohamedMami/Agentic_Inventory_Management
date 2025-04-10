import sys
from pathlib import Path
from database import Session
from agents.inventory_agent import InventoryAgent

def test_inventory_queries():
    # Create agent instance
    agent = InventoryAgent()
    
    # Get database session
    session = Session()
    
    try:
        # Test queries
        test_queries = [
            "What is the current stock of all products?",
            "Show me products that expire in the next 30 days",
            "What are the top 5 selling products?"
        ]
        
        for query in test_queries:
            print("\n" + "="*50)
            print(f"Testing query: {query}")
            print("="*50)
            
            # Process query
            result = agent.process_query(query, session)
            
            # Print results
            print("\nSQL Query:")
            print(result['sql_query'])
            print("\nData:")
            print(result['data'])
            print("\nResponse:")
            print(result['response'])
            
            if result['error']:
                print("\nError:")
                print(result['error'])
                
    finally:
        session.close()

if __name__ == "__main__":
    test_inventory_queries()