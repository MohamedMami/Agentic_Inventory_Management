"""
integration module for the system components
"""
import logging
from sqlalchemy.orm import Session
from database import engine
from agents.visualization_agent import VisualizationAgent
from agents.supervisor_agent import  SupervisorAgent
from agents.inventory_agent import InventoryAgent 
from agents.forecast_agent import  ForcastAgent 
from logger import get_logger 

logger = get_logger("integration")

class SystemIntegration:
    def __init__(self):
        logger.info("initialisation of the system ")
        
        self.supervisor = SupervisorAgent()
        self.inventory = InventoryAgent()
        self.forecast = ForcastAgent()
        self.visualization = VisualizationAgent()
        
        logger.info("initialisation of agents")
    def process_query(self,query, session: None):
        
        
        logger.info("processing query")
        close_session = False
        if session is None:
            session = Session(engine)
            close_session = True
        try :
            query_type = self.supervisor.classify_query(query)
            logger.info(f"query classified as: {query_type.value}")
            if query_type == SupervisorAgent.QueryType.INVENTORY:
                results = self.inventory.process_query(query, session)
                results["query_type"] = "inventory"
            elif query_type == SupervisorAgent.QueryType.VISUALIZATION:
                results = self.visualization.process_query(query, session)
                results["query_type"] = "visualization"
            # elif query_type == SupervisorAgent.QueryType.FORECAST:
            #     results = self.forecast.process_query(query, session)
            #     results["query_type"] = "forecast"
            else:
                logger.warning(f"request type unknown: {query_type.value}")
                results = {
                    "query": query,
                    "query_type": "Unknown",
                    "response": "sorry ,we can't comprehend your question.Please reformulate again.",
                    "data": None,
                    "error": "request type unknown"
                }  
            if "error" in results and results["error"]:
                logger.error(f"error while handling the request: {results['error']}")
            else:
                logger.info("resuest successfully handled")
            
            return results
            
        except Exception as e:
            logger.error(f"Unhandled error while processing the request: {e}")
            return {
                "query": query,
                "response": f"An error occurred while processing your request: {e}",
                "data": None,
                "error": str(e)
            }
        finally:
            # Fermeture de la session si créée localement
            if close_session:
                session.close()
                logger.debug("Session closed")
                
system = SystemIntegration()