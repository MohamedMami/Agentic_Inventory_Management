import sys
import os
# Add the src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from sqlalchemy.orm import Session

from src.database import engine
from src.integration import system
from logger import get_logger

logger = get_logger("query_page")

def query_page():
    """Displays the conversational query interface page."""
    logger.info("Loading query page")
    st.title("Conversational Query Interface")
    
    # Introduction et exemples
    st.write("""
             ## Ask your questions in natural language
            Our system uses artificial intelligence to understand your pharmaceutical inventory questions
            and provide you with precise answers, relevant visualizations, or forecasts based on historical data.
        """)
    
    # Exemples de requêtes
    with st.expander("Examples of queries", expanded=False):
        st.markdown("""
        ### Inventory Queries
        - Which products are below the alert threshold?
        - How many units of Amoxicillin do we have in stock?
        - Which products require a prescription?
        - What is the average unit price of antibiotics?

        ### Visualization Queries
        - Show me a graph of inventory by product category
        - Visualize the distribution of products by warehouse
        - Create a sales graph for the last 3 months
        - Display the best-selling products as a graph

        ### Forecasting Queries
        - Forecast antibiotic demand for the next 3 months
        - Which products are likely to be out of stock in the coming weeks?
        - Analyze vitamin sales trends
        - Predict inventory requirements for the next quarter
        """)
    
    # history
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
        logger.debug("Initializing query history")
    

    if st.session_state.query_history:
        with st.expander("Query history", expanded=False):
            for i, (past_query, timestamp) in enumerate(st.session_state.query_history):
                st.write(f"**{timestamp}**: {past_query}")
                if i < len(st.session_state.query_history) - 1:
                    st.divider()
    
    # query input
    query = st.text_input(
        "Your query:",
        placeholder="Enter your question about pharmaceutical inventory...",
        key="query_input"
    )
    
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("erase", use_container_width=False)
    
    
    if clear_button:
        logger.debug("Delete the query")
        st.session_state.query_input = ""
        st.rerun()
    
    
    if submit_button and query:
        logger.info(f"New request submitted: '{query}'")
        
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        st.session_state.query_history.append((query, timestamp))
        
        if len(st.session_state.query_history) > 10:
            st.session_state.query_history = st.session_state.query_history[-10:]
        
        session = Session(engine)
        
        try:
            st.divider()
            st.subheader(f"Requête: {query}")
            with st.spinner("Processing your request..."):
                logger.info("Processing the request via the integration module")
                result = system.process_query(query, session)
                
                # query type 
                query_type = result.get("query_type", "unkown")
                st.caption(f"Query type detected: **{query_type}**")
                logger.debug(f"Query type detected: {query_type}")
            
            # response 
            if "response" in result and result["response"]:
                st.write("### Answer")
                st.write(result["response"])
            
            # data response 
            if "data" in result and result["data"]:
                with st.expander("Detailed data", expanded=True):
                    st.dataframe(pd.DataFrame(result["data"]), use_container_width=True)
                    
                    # Option to download the data as CSV
                    csv = pd.DataFrame(result["data"]).to_csv(index=False)
                    st.download_button(
                        label="Download data (CSV)",
                        data=csv,
                        file_name=f"query_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # visualization response
            if "visualization_path" in result and result["visualization_path"]:
                st.write("### Visualization")
                
                # Checking that the file exists
                if os.path.exists(result["visualization_path"]):
                    st.image(result["visualization_path"])
                    
                    # Option to download the image
                    with open(result["visualization_path"], "rb") as file:
                        st.download_button(
                            label="Download visualization",
                            data=file,
                            file_name=os.path.basename(result["visualization_path"]),
                            mime="image/png"
                        )
                else:
                    st.error(f"View file not found: {result['visualization_path']}")
            
            # error handling
            if "error" in result and result["error"]:
                st.error(f"error: {result['error']}")
            
            # Displaying the SQL query (for inventory queries)
            if "sql_query" in result and result["sql_query"]:
                with st.expander("Generated SQL query", expanded=False):
                    st.code(result["sql_query"], language="sql")
        
        finally:
            session.close()
