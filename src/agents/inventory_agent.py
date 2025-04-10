import logging
import json
from typing import Dict, Any, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from agents.base_agent import BaseAgent
from models import sales, product, inventory

# Configuration du logging
logger = logging.getLogger(__name__)

class InventoryAgent(BaseAgent):
    """
    Agent spécialisé dans la gestion de l'inventaire pharmaceutique.
    """
    
    def __init__(self):
        """inventory agent initialization"""
        super().__init__()
        self.system_message = """
                You are an assistant specializing in pharmaceutical inventory management.
                Your task is to analyze user queries regarding inventory and convert them into SQL queries to extract relevant information from the database.

                Database schema:
                - inventory : inventory_id,product_id,product_name,batch_number,current_quantity,manufacturing_date,expiry_date,warehouse_location,temperature_compliant,last_checked
                - product :product_id,product_name,generic_name,strength_form,category,manufacturer,atc_code,storage_instructions,prescription_required,controlled_substance_class,requires_refrigeration,unit_price,min_stock_level,reorder_lead_time_days,approval_date,package_size
                - sales : sale_id,product_id,product_name,category,sale_date,quantity,unit_price,total_value,region,is_weekend,month,year,day_of_week
                Respond ONLY with the appropriate SQL query and don't generate other text than the sql response.
        """
    
    def generate_sql_query(self, query: str) -> str:
        """
        Generates an SQL query from a natural language query.

        Args:
        query: The natural language query

        Returns:
        The generated SQL query
        """
        prompt = f"""Convert the following query to SQL to retrieve the requested information from the database.
                Use only the tables and columns mentioned in the schema.

                Query: "{query}"

                Reply with only the SQL query, without comments or explanations.
        """
        
        response = self._get_llm_response(prompt, self.system_message)
        sql_query = response.strip()
        
        # basic verification 
        if "DROP" in sql_query.upper() or "DELETE" in sql_query.upper() or "UPDATE" in sql_query.upper() or "INSERT" in sql_query.upper():
            logger.warning(f"Potentially dangerous SQL query attempt: {sql_query}")
            return "SELECT 'Unauthorized query' as message"
        
        return sql_query
    
    def execute_sql_query(self, sql_query: str, session: Session) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Executes an SQL query and returns the results.

        Args:
        sql_query: The SQL query to execute
        session: Database session

        Returns:
        Tuple containing the list of results and any error messages
        """
        try:
            
            result = session.execute(text(sql_query))
            column_names = result.keys()
            rows = []
            
            for row in result:
                row_dict = {}
                for i, column in enumerate(column_names):
                    row_dict[column] = row[i]
                rows.append(row_dict)
            
            return rows, None
            
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            return [], f"Error executing query: {e}"
    
    def generate_natural_language_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generates a natural language response from the query results.

        Args:
        query: The original natural language query
        results: The results of the SQL query

        Returns:
        A natural language response
        """
        results_json = json.dumps(results, ensure_ascii=False, default=str)
        
        prompt = f"""
        Here's a user query and the results returned from the database.
        Generate a clear and concise response in French that addresses the user's query.

        Query: "{query}"

        Results: {results_json}

        Respond in a professional and informative manner, including relevant figures.
        """
        
        system_message = """
        You are an assistant specializing in pharmaceutical inventory management.
        Your task is to generate clear and informative responses based on inventory data.
        Use a professional and precise tone, appropriate for the pharmaceutical industry.
        """
        
        response = self._get_llm_response(prompt, system_message)
        
        return response
    
    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """
        Processes a user query regarding inventory.

        Args:
        query: The natural language query
        session: Database session

        Returns:
        Dictionary containing the response and associated data
        """
        # sql generation
        sql_query = self.generate_sql_query(query)
        logger.info(f"sql generated: {sql_query}")
        
        results, error = self.execute_sql_query(sql_query, session)
        response = {
            "query": query,
            "sql_query": sql_query,
            "data": results,
            "error": error
        }
        
        # Generate natural language response if no error
        if not error and results:
            natural_response = self.generate_natural_language_response(query, results)
            response["response"] = natural_response
        elif not error and not results:
            response["response"] = "No data found for this query."
        else:
            response["response"] = f"Sorry, an error has occurred: {error}"
        
        return response
