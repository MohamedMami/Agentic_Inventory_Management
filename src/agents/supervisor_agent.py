"""
Agent superviseur pour le système de gestion d'inventaire pharmaceutique.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from functools import lru_cache
from sqlalchemy.orm import Session
from datetime import datetime

from src.agents.base_agent import BaseAgent

# Configuration du logging
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Query types supported by the system."""
    INVENTORY = "inventory"
    VISUALIZATION = "visualization"
    FORECAST = "forecast"
    UNKNOWN = "unknown"

class SupervisorAgent(BaseAgent):
    """Optimized supervisor agent for routing queries to specialized agents."""
    
    # Add QueryType as a class attribute
    QueryType = QueryType
    
    def __init__(self):
        super().__init__()
        self._initialize_agent()
        self.query_patterns = {
            QueryType.INVENTORY: [
                "stock", "inventory", "quantity", "available", "warehouse",
                "expiry", "batch", "supply", "units", "products"
            ],
            QueryType.VISUALIZATION: [
                "show", "display", "graph", "chart", "plot", "visualize",
                "trend", "compare", "distribution", "breakdown"
            ],
            QueryType.FORECAST: [
                "predict", "forecast", "future", "projection", "estimate",
                "demand", "next", "upcoming", "expected", "trend"
            ]
        }

    def _initialize_agent(self):
        """Initialize agent with optimized system message."""
        self.system_message = """
        You are an efficient pharmaceutical inventory management assistant.
        Classify queries into: INVENTORY, VISUALIZATION, or FORECAST.
        - INVENTORY: Stock levels, product info, warehouse data
        - VISUALIZATION: Charts, graphs, visual analysis
        - FORECAST: Predictions, trends, future projections
        Respond only with the classification type.
        """

    def classify_query(self, query: str) -> QueryType:
        """Classify query using pattern matching."""
        query_lower = query.lower()
        
        # Try pattern matching first
        for query_type, patterns in self.query_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return query_type
        
        # Fall back to LLM for complex queries
        response = self._get_llm_response(
            f"Query: {query}\nClassification:",
            self.system_message
        )
        
        response = response.strip().upper()
        if "INVENTORY" in response:
            return QueryType.INVENTORY
        elif "VISUALIZATION" in response:
            return QueryType.VISUALIZATION
        elif "FORECAST" in response:
            return QueryType.FORECAST
            
        return QueryType.UNKNOWN

    async def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """Process queries with improved error handling and response structure."""
        try:
            query_type = self.classify_query(query)
            
            response = {
                "query": query,
                "query_type": query_type.value,
                "timestamp": datetime.now().isoformat(),
                "response": None,
                "data": None,
                "visualization": None,
                "error": None
            }

            logger.info(f"Query processed - Type: {query_type.value}")
            return response

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return {
                "query": query,
                "query_type": QueryType.UNKNOWN.value,
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
