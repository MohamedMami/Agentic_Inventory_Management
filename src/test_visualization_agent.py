from database import Session
from agents.visualization_agent import VisualizationAgent
import webbrowser
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_viz():
    agent = VisualizationAgent()
    session = Session()
    
    try:
        query = "Show me a bar chart of the sales in 2025 by category."
        # Await the async process_query call
        result = await agent.process_query(query, session)
        
        logger.info(f"Query: {result['query']}")
        logger.info(f"Response: {result['response']}")
        logger.info(f"Data: {result['data']}")
        
        if result.get("visualization_base64"):
            logger.info("Visualization generated successfully")
            # Handle base64 visualization data
            if result["visualization_base64"]:
                # Save to temporary file and open
                temp_path = "temp_viz.html"
                with open(temp_path, "w") as f:
                    f.write(result["visualization_base64"])
                abs_path = os.path.abspath(temp_path)
                logger.info(f"Opening visualization: {abs_path}")
                webbrowser.open(f"file://{abs_path}")
        else:
            logger.warning("No visualization generated")
            
    except Exception as e:
        logger.error(f"Error during visualization test: {str(e)}")
        raise
    finally:
        session.close()

if __name__ == "__main__":
    # Run the async function using asyncio
    asyncio.run(test_viz())