"""
Agent superviseur pour le système de gestion d'inventaire pharmaceutique.
"""
import logging
from typing import Dict, Any, List, Optional
from enum import Enum

from sqlalchemy.orm import Session

from agents.base_agent import BaseAgent

# Configuration du logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types de requêtes possibles."""
    INVENTORY = "inventory"
    VISUALIZATION = "visualization"
    FORECAST = "forecast"
    UNKNOWN = "unknown"

class SupervisorAgent(BaseAgent):
    """
    Agent superviseur responsable de router les requêtes vers les agents spécialisés.
    """
    
    def __init__(self):
        """Initialise l'agent superviseur."""
        super().__init__()
        self.system_message = """
            You are an assistant specializing in pharmaceutical inventory management.
            Your task is to analyze user queries and determine which agent type should handle each query.
            Available agent types:
            1. Inventory Agent: For queries regarding inventory, products, warehouses, etc.
            2. Visualization Agent: For queries requesting charts, tables, or visualizations
            3. Forecasting Agent: For queries regarding demand forecasts, future trends, etc.

            Respond only with the appropriate agent type.
            """
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classifies a user request to determine which agent should handle it.
        Args:
        query: The natural language query
        Returns:
        The query type (enum QueryType)
        """
        prompt = f"""
        Analyzes the following query and determines which agent type should handle it.
        Responds only with one of the following types: INVENTORY, VISUALIZATION, FORECAST.

        Query: "{query}"
        """
        
        response = self._get_llm_response(prompt, self.system_message)
        
        # Normalization of the response
        response = response.strip().upper()
        
        if "INVENTORY" in response:
            return QueryType.INVENTORY
        elif "VISUALIZATION" in response:
            return QueryType.VISUALIZATION
        elif "FORECAST" in response:
            return QueryType.FORECAST
        else:
            return QueryType.UNKNOWN
    
    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """
            Processes a user request by routing it to the appropriate agent.

            Args:
            query: The natural language query
            session: Database session

            Returns:
            Dictionary containing the response and associated data
        """
        # classify the query
        query_type = self.classify_query(query)
        
        # logging
        logger.info(f"query classified as: {query_type.value}")
        
        response = {
            "query": query,
            "query_type": query_type.value,
            "response": None,
            "data": None,
            "visualization": None
        }
        # At this stage, we are only performing the classification.
        # Specific agents will be called in a later step.
        
        return response
