import logging
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from typing import Dict, Any 
from src.agents.supervisorReAct import SupervisorReActAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.visualization_agent import VisualizationAgent
import os
import datetime
import asyncio

class SystemIntegration:
    def __init__(self):
        # Initialize the Supervisor with proper configuration   
        self.supervisor = SupervisorReActAgent(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )
        
        # Initialize and register specialized agents
        self.inventory_agent = InventoryAgent()
        self.visualization_agent = VisualizationAgent()
        # self.forecast_agent = ForecastAgent()  # Uncomment if available
        
        self.supervisor.register_agent("inventory", self.inventory_agent)
        self.supervisor.register_agent("visualization", self.visualization_agent)
        # self.supervisor.register_agent("forecast", self.forecast_agent)

    async def process_query(
        self, 
        query: str, 
        session: Session = None, 
        conversation_id: str = None) -> Dict[str, Any]:
        close_session = False
        engine = create_engine(os.getenv("DATABASE_URL"))
        
        if session is None:
            session = Session(engine)
            close_session = True
        
        try:
            # Generate conversation ID
            if not conversation_id:
                conversation_id = f"conv_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Let the Supervisor handle everything
            result = await self.supervisor.process_query(
                query=query,
                session=session,
                conversation_id=conversation_id
            )
            
            return result  # Supervisor returns a structured response
            
        except Exception as e:
            logging.error(f"SystemIntegration Error: {str(e)}")
            return {
                "query": query,
                "response": "An error occurred. Please try again.",
                "data": None,
                "error": str(e)
            }
        finally:
            if close_session:
                session.close()