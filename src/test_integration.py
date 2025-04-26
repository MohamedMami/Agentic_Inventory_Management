import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.integration import system
from sqlalchemy.orm import Session
from src.database import engine

def test_queries():
    session = Session(engine)
    
    test_questions = [
        "How many units of Amoxicillin do we have in stock?",
        "Show me a graph of inventory by product category",
        "Which products are below minimum stock level?",
        "Display sales trends for the last 3 months"
    ]

    try:
        for question in test_questions:
            print(f"\nTesting query: {question}")
            print("-" * 50)
            
            result = system.process_query(question, session)
            
            print(f"Query type: {result.get('query_type', 'Unknown')}")
            print(f"Response: {result.get('response', 'No response')}")
            if result.get('error'):
                print(f"Error: {result['error']}")
            print("-" * 50)
    
    finally:
        session.close()

if __name__ == "__main__":
    test_queries()