"""
Agent de visualisation pour le système de gestion d'inventaire pharmaceutique.
"""
import logging
import json
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import base64
import asyncio
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.agents.base_agent import BaseAgent  
from src.agents.inventory_agent import InventoryAgent


logger = logging.getLogger(__name__)

class VisualizationAgent(BaseAgent):
    """
   Agent specializing in the visualization of pharmaceutical inventory data.
    """
    
    def __init__(self):
        """Initializes the visualization agent."""
        super().__init__()
        self.inventory_agent = InventoryAgent()
        self.system_message = """
            You are an assistant specializing in pharmaceutical inventory data visualization.
            Your task is to analyze user queries regarding visualization and generate
            the appropriate Python code to create meaningful graphs using Plotly.

            Use plotly.express (as px) for simple charts:
            - px.bar() for bar charts
            - px.line() for time series
            - px.pie() for proportions
            - px.scatter() for relationships
            - px.histogram() for distributions
            - px.heatmap() for correlations

            For more complex visualizations, use plotly.graph_objects (as go).
            
            The code should:
            1. Convert the data to a pandas DataFrame
            2. Create an appropriate plotly figure
            3. Add a descriptive title and axis labels
            4. Use appropriate color schemes
            5. Save the figure using fig.write_html() for interactive plots
            
            Respond only with the Python code, without explanations.
        """
        
        # Create directories for visualizations
        os.makedirs("data/visualizations", exist_ok=True)
    
    def generate_visualization_code(self, query: str, data: List[Dict[str, Any]]) -> str:
        """
        Generates Python code to create a visualization from the data.
        """
        self.current_data = data
        data_sample = json.dumps(data[:2], ensure_ascii=False, default=str)
        
        prompt = f"""
        Generate Python code using Plotly to create an interactive visualization.
        
        Query: "{query}"
        Sample data: {data}
        
        For time series data, use this format:
        df = pd.DataFrame(data)
        # Group by month and sum values
        df['month'] = pd.to_datetime(df['sale_date']).dt.strftime('%Y-%m')
        monthly_data = df.groupby('month').agg({{'quantity': 'sum'}}).reset_index()
        
        fig = px.line(monthly_data, 
            x='month',
            y='quantity',
            title='Monthly Sales Trends'
        )
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title='Total Quantity'
        )
        
        For aggregated data, use this format:
        df = pd.DataFrame(data)
        agg_df = df.groupby(['category']).agg({{'quantity': 'sum'}}).reset_index()
        fig = px.bar(agg_df,
            x='category',
            y='quantity',
            title='Total Quantity by Category'
        )
        
        The code must:
        1. Handle proper date grouping for time series
        2. Use appropriate aggregation functions
        3. Include clear labels and titles
        4. Use plotly.express for visualization
        
        
        Return only the Python code, no explanations.
        """
        
        raw_response = self._get_llm_response(prompt, self.system_message)
        
        # Clean up the code
        cleaned_code = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        cleaned_code = re.sub(r'```.*?\n', '', cleaned_code)  # Remove opening ```
        cleaned_code = re.sub(r'\n```$', '', cleaned_code)    # Remove closing ```
        cleaned_code = re.sub(r'\s*fig\.write_html\([^)]*\)\s*', '', cleaned_code)  # Remove fig.write_html() if present
        # Find the first import statement
        import_match = re.search(r'(?:import|from)\s+\w+', cleaned_code)
        if import_match:
            cleaned_code = cleaned_code[import_match.start():]
        
        # Clean up whitespace and ensure proper ending
        cleaned_code = cleaned_code.strip()
        if cleaned_code.endswith('```'):
            cleaned_code = cleaned_code[:-3].strip()
        
        # Ensure code ends with proper closing parenthesis if needed
        if '(' in cleaned_code and not cleaned_code.strip().endswith(')'):
            cleaned_code = cleaned_code.rstrip('`').strip()
        
        # Add required imports
        required_imports = [
            'import pandas as pd',
            'import plotly.express as px'
        ]
        
        for imp in required_imports:
            if imp not in cleaned_code:
                cleaned_code = f"{imp}\n{cleaned_code}"
        
        return cleaned_code
    
    async def execute_visualization_code(self, code: str, query: str) -> Tuple[str, Optional[str]]:
        """
        Executes the visualization code and saves the result.
        """
        try:
            # Create filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"data/visualizations/viz_{timestamp}.html"
            
            # Convert data to DataFrame if it's not already
            if isinstance(self.current_data, list):
                df = pd.DataFrame(self.current_data)
            else:
                df = pd.DataFrame([self.current_data])
            
            # Prepare the execution environment
            namespace = {
                'pd': pd,
                'px': plotly.express,
                'go': plotly.graph_objects,
                'data': df,  # Pass DataFrame instead of raw data
                'html_path': html_path,
                'np': np  # Add numpy for numerical operations
            }
            
            # Clean up the code
            code = code.strip()
            
            # Ensure DataFrame creation uses the provided data
            if 'pd.DataFrame(data)' not in code:
                code = f"df = pd.DataFrame(data)\n{code}"
            
            # Add save command if not present
            if "fig.write_html(" not in code:
                code += f"\nfig.write_html('{html_path}')"
            
            # Execute the code in the prepared namespace
            exec(code, namespace)
            
            if os.path.exists(html_path):
                # Convert HTML to base64
                with open(html_path, "r", encoding="utf-8") as file:
                    html_content = file.read()
                return html_content, None
            else:
                return "", "Visualization file was not created"
                
        except Exception as e:
            logger.error(f"Error running visualization code: {e}")
            error_msg = f"Error creating visualization: {str(e)}\nCode:\n{code}"
            return "", error_msg
    
    async def generate_visualization_description(self, query: str, data: List[Dict[str, Any]]) -> str:
        """
        Generates a description of the created visualization.

        Args:
        query: The original query in natural language
        data: The data used for the visualization

        Returns:
        A natural language description of the visualization
        """
       
        data_json = json.dumps(data[:5], ensure_ascii=False, default=str)  
        
        prompt = f"""
        Here is a user query and the data used to create a visualization.
        Generate a clear and concise description of the visualization created.

        Query: "{query}"

        Sample data: {data_json}

        Explain what the visualization shows, the key trends or insights,
        and how it answers the user query.
        """

        system_message = """
        You are an assistant specializing in pharmaceutical data analysis.
        Your task is to clearly explain inventory data visualizations.
        Use a professional and precise tone, appropriate for the pharmaceutical industry.
        """
        
        response = await self._get_llm_response(prompt, system_message)
        
        return response
    
    async def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """
        Processes a user query regarding the visualization asynchronously.

        Args:
            query: The natural language query
            session: Database session

        Returns:
            Dictionary containing the response and associated data
        """
        try:
            # Using the inventory agent to get the data asynchronously
            inventory_response = await self.inventory_agent.process_query(query, session)
            data = inventory_response.get("data", [])
            
            # Preparing the response
            response = {
                "query": query,
                "data": data,
                "visualization_base64": None,  # Changed from visualization_path to visualization_base64
                "error": None,
                "response": None,
                "query_type": "visualization"  # Added query_type
            }
            
            # If no data, return an error
            if not data:
                response["error"] = "No data available to create a visualization."
                response["response"] = "Sorry, I was unable to obtain data to create a visualization based on your query."
                return response
            
            # Generating the visualization code
            viz_code = await self.generate_visualization_code(query, data)
            logger.info(f"Visualization code generated: {viz_code[:100]}...")
            
            # Executing the visualization code and getting base64 image
            viz_base64, error = await self.execute_visualization_code(viz_code, query)
            
            if error:
                response["error"] = error
                response["response"] = f"Sorry, an error occurred while creating the visualization: {error}"
                return response
            
            # Adding the visualization html  data to the response
            response["visualization__html"] = viz_base64
            
            # Generating the visualization description
            description = await self.generate_visualization_description(query, data)
            response["response"] = description
            
            return response
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your visualization request: {str(e)}",
                "query_type": "visualization"
            }
