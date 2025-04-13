"""
Agent de visualisation pour le système de gestion d'inventaire pharmaceutique.
"""
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.orm import Session

from agents.base_agent import BaseAgent  
from agents.inventory_agent import InventoryAgent

# Configuration du logging
logger = logging.getLogger(__name__)
from agents.inventory_agent import InventoryAgent

# Configuration du logging
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
        # Store data for execution context
        self.current_data = data
        
        data_sample = json.dumps(data[:2], ensure_ascii=False, default=str)
        
        prompt = f"""
        Generate Python code using Plotly to create an interactive visualization.
        
        Query: "{query}"
        Sample data: {data_sample}
        
        Use this exact format:
        df = pd.DataFrame(data)
        fig = px.bar(df, 
            x='column_name',
            y='value_column',
            title='Descriptive Title'
        )
        fig.update_layout(
            xaxis_title='X Axis Label',
            yaxis_title='Y Axis Label'
        )
        
        The code must:
        1. Use the 'data' variable that's already available
        2. Create a pandas DataFrame
        3. Use plotly.express for visualization
        4. Include proper title and labels
        5. No string literals should be left unterminated
        
        Return only valid Python code.
        """
        
        response = self._get_llm_response(prompt, self.system_message)
        return response.strip()
    
    def execute_visualization_code(self, code: str, query: str) -> Tuple[str, Optional[str]]:
        """
        Executes the visualization code and saves the result.
        """
        try:
            # Create filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"data/visualizations/viz_{timestamp}.html"
            
            # Prepare the execution environment
            namespace = {
                'pd': pd,
                'px': plotly.express,
                'go': plotly.graph_objects,
                'data': self.current_data,  # Store data as class attribute
                'html_path': html_path
            }
            
            # Clean up the code and ensure proper string handling
            code = code.strip()
            if not code.endswith(';'):
                code += '\n'
                
            # Add save command if not present
            if "fig.write_html(" not in code:
                code += f"fig.write_html('{html_path}')\n"
            
            # Execute the code in the prepared namespace
            exec(code, namespace)
            
            if os.path.exists(html_path):
                return html_path, None
            else:
                return "", "Visualization file was not created"
                
        except Exception as e:
            logger.error(f"Error running visualization code: {e}")
            return "", f"Error creating visualization: {str(e)}"
    
    def generate_visualization_description(self, query: str, data: List[Dict[str, Any]], viz_path: str) -> str:
        """
        Generates a description of the created visualization.

        Args:
        query: The original query in natural language
        data: The data used for the visualization
        viz_path: The path to the visualization file

        Returns:
        A natural language description of the visualization
        """
       
        data_json = json.dumps(data[:5], ensure_ascii=False, default=str)  # Limité aux 5 premiers éléments
        
        prompt = f"""
        Here is a user query and the data used to create a visualization.
        Generate a clear and concise description of the visualization created.

        Query: "{query}"

        Sample data: {data_json}

        The visualization was saved in: {viz_path}

        Explain what the visualization shows, the key trends or insights,
        and how it answers the user query.
        """

        system_message = """
        You are an assistant specializing in pharmaceutical data analysis.
        Your task is to clearly explain inventory data visualizations.
        Use a professional and precise tone, appropriate for the pharmaceutical industry.
        """
        
        response = self._get_llm_response(prompt, system_message)
        
        return response
    
    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        """
        Processes a user query regarding the visualization.

        Args:
        query: The natural language query
        session: Database session

        Returns:
        Dictionary containing the response and associated data
        """
        # Using the inventory agent to get the data
        inventory_response = self.inventory_agent.process_query(query, session)
        data = inventory_response.get("data", [])
        
        # Preparing the response
        response = {
            "query": query,
            "data": data,
            "visualization_path": None,
            "error": None,
            "response": None
        }
        
        # If no data, return an error
        if not data:
            response["error"] = "No data available to create a visualization."
            response["response"] = "Sorry, I was unable to obtain data to create a visualization based on your query."
            return response
        
        # Generating the visualization code
        viz_code = self.generate_visualization_code(query, data)
        logger.info(f"Code de visualisation généré: {viz_code[:100]}...")
        
        # excuting the visualization code
        viz_path, error = self.execute_visualization_code(viz_code, query)
        
        if error:
            response["error"] = error
            response["response"] = f"Sorry, an error occurred while creating the visualization: {error}"
            return response
        
        # Adding the visualization path to the response
        response["visualization_path"] = viz_path
        
        # Generating the visualization description
        description = self.generate_visualization_description(query, data, viz_path)
        response["response"] = description
        
        return response
