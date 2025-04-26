import logging
import json
from typing import Dict, Any, List, Optional, Tuple
import re
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.agents.base_agent import BaseAgent
from src.models import Inventory, Product, Sale

# Configuration du logging
logger = logging.getLogger(__name__)

class InventoryAgent(BaseAgent):
    """
    Agent specialized in the pharmaceutical inventory management.
    """
    
    def __init__(self):
        """inventory agent initialization"""
        super().__init__()
        self.system_message = """
                You are an assistant specializing in pharmaceutical inventory management.
                Your task is to analyze user queries regarding inventory and convert them into PostgreSQL queries.
                Database schema:
                Tables (all lowercase):
                - inventory: inventory_id,product_id,product_name,batch_number,current_quantity,manufacturing_date,expiry_date,warehouse,location,temperature_compliant,last_checked
                - products: product_id,product_name,generic_name,strength_form,category,manufacturer,atc_code,storage_instructions,prescription_required,controlled_substance_class,requires_refrigeration,unit_price,min_stock_level,reorder_lead_time_days,registration_date,package_size
                - sales: sale_id,product_id,product_name,category,sale_date,quantity,unit_price,total_value,facility_id,facility_name,facility_type,region,governorate,is_weekend,is_holiday,month,year,day_of_week,cost_per_unit,total_cost,profit
                
                Use PostgreSQL syntax and lowercase table names. For date operations, use CURRENT_DATE and interval '30 days'.
                Examples:
                    Query: "What products expire in the next 30 days?"
                    SQL: "SELECT p.product_name, i.expiry_date FROM products p 
                        JOIN inventory i ON p.product_id = i.product_id 
                        WHERE i.expiry_date BETWEEN CURRENT_DATE AND CURRENT_DATE + interval '30 days';"
                    
                    Query: "What are the top 5 selling products?"
                    SQL: "SELECT product_name, SUM(quantity) as total_sales 
                        FROM sales GROUP BY product_name 
                        ORDER BY total_sales DESC LIMIT 5;"
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
                Return ONLY the SQL query without any XML tags, thoughts, or explanations.

                Query: "{query}"
        """
        
        response = self._get_llm_response(prompt, self.system_message)
        
        # Remove any XML-like tags and their content
        response = re.sub(r'<[^>]+>.*?</[^>]+>', '', response, flags=re.DOTALL)
        # Remove any remaining tags
        response = re.sub(r'<[^>]+>', '', response)
        # Clean up extra whitespace
        sql_query = ' '.join(response.strip().split())
        
        # basic verification 
        if any(keyword in sql_query.upper() for keyword in ['DROP', 'DELETE', 'UPDATE', 'INSERT']):
            logger.warning(f"Potentially dangerous SQL query attempt: {sql_query}")
            return "SELECT 'Unauthorized query' as message"
        
        return sql_query
    
    def execute_sql_query(self, sql_query: str, session: Session) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Executes an SQL query and returns the results.
        """
        try:
            # Start a new transaction
            session.begin()
            
            # Execute query
            result = session.execute(text(sql_query))
            column_names = result.keys()
            rows = []
            
            for row in result:
                row_dict = {}
                for i, column in enumerate(column_names):
                    row_dict[column] = row[i]
                rows.append(row_dict)
            
            # Commit the transaction
            session.commit()
            return rows, None
            
        except Exception as e:
            # Rollback on error
            session.rollback()
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
        Generate a clear and concise response in english that addresses the user's query.

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

