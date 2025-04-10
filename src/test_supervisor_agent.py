import sys
from pathlib import Path
from database import Session
from agents.supervisor_agent import SupervisorAgent, QueryType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_query_classification():
    """Test the query classification functionality"""
    agent = SupervisorAgent()
    
    # Test cases with expected results
    test_cases = [
        ("What is the current stock level of all products?", QueryType.INVENTORY),
        ("Show me a chart of sales trends over the last 6 months", QueryType.VISUALIZATION),
        ("Predict the demand for antibiotics next month", QueryType.FORECAST),
        ("Hello, how are you?", QueryType.UNKNOWN),
    ]
    
    for query, expected_type in test_cases:
        print("\n" + "="*50)
        print(f"Testing query: {query}")
        print(f"Expected type: {expected_type.value}")
        
        result = agent.classify_query(query)
        print(f"Actual type: {result.value}")
        
        assert result == expected_type, f"Expected {expected_type.value}, but got {result.value}"

def test_query_processing():
    """Test the complete query processing pipeline"""
    agent = SupervisorAgent()
    session = Session()
    
    try:
        test_queries = [
            "Show me the current inventory levels",
            "Create a pie chart of product categories",
            "Predict stock requirements for next quarter",
            "Invalid query !!!",
        ]
        
        for query in test_queries:
            print("\n" + "="*50)
            print(f"Processing query: {query}")
            
            result = agent.process_query(query, session)
            
            print("Response:")
            for key, value in result.items():
                print(f"{key}: {value}")
            
    finally:
        session.close()

if __name__ == "__main__":
    try:
        print("Testing query classification...")
        test_query_classification()
        print("\nQuery classification tests passed successfully!")
        
        print("\nTesting query processing...")
        test_query_processing()
        print("\nQuery processing tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)