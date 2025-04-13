from database import Session
from agents.visualization_agent import VisualizationAgent
import webbrowser
import os

def test_viz():
    agent = VisualizationAgent()
    session = Session()
    
    query = "Show me a bar chart of product quantities by warehouse"
    result = agent.process_query(query, session)
    
    if result["visualization_path"]:
        abs_path = os.path.abspath(result["visualization_path"])
        print(f"Opening visualization: {abs_path}")
        webbrowser.open(f"file://{abs_path}")
    
    session.close()

if __name__ == "__main__":
    test_viz()