from database import Session
from agents.visualization_agent import VisualizationAgent
import webbrowser
import os

def test_viz():
    agent = VisualizationAgent()
    session = Session()
    
    query = "Show me a bar chart of the  sales in a march 2025 by category."
    result = agent.process_query(query, session)
    code = agent.generate_visualization_code(query, result["data"])
    print ("the code :" + code)
    print("test"+result["response"])
    print("query "+result["query"])
    print(result["data"])
    if result["visualization_path"]:
        abs_path = os.path.abspath(result["visualization_path"])
        print(f"Opening visualization: {abs_path}")
        webbrowser.open(f"file://{abs_path}")
    
    session.close()

if __name__ == "__main__":
    test_viz()