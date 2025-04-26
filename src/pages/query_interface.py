import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session

from src.database import engine
from src.integration import system
from src.logger import get_logger

logger = get_logger("query_page")

# Set page config at the very start
st.set_page_config(
    page_title="Pharmaceutical Query Interface",
    page_icon="💊",
    layout="wide"
)

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Pharmaceutical Inventory Chat")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            # Handle different types of assistant responses
            if "response" in message:
                st.markdown(message["response"])
            
            if "data" in message and message["data"] is not None:
                with st.expander("View Data Details"):
                    st.dataframe(pd.DataFrame(message["data"]))
                    
                    # Download button for data
                    csv = pd.DataFrame(message["data"]).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Data (CSV)",
                        data=csv,
                        file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            if "visualization_path" in message and message["visualization_path"]:
                if os.path.exists(message["visualization_path"]):
                    st.image(message["visualization_path"])
                    
                    # Download button for visualization
                    with open(message["visualization_path"], "rb") as file:
                        st.download_button(
                            label="📥 Download Visualization",
                            data=file,
                            file_name=os.path.basename(message["visualization_path"]),
                            mime="image/png"
                        )

# Chat input
query = st.chat_input("Ask about your pharmaceutical inventory...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            session = Session(engine)
            try:
                result = system.process_query(query, session)
                
                assistant_message = {
                    "role": "assistant",
                    "response": result.get("response", "I couldn't process that query."),
                    "data": result.get("data"),
                    "visualization_path": result.get("visualization_path"),
                    "query_type": result.get("query_type", "unknown")
                }
                
                # Display response
                st.markdown(assistant_message["response"])
                
                # Display data if available
                if assistant_message["data"] is not None:
                    with st.expander("View Data Details"):
                        st.dataframe(pd.DataFrame(assistant_message["data"]))
                        
                        csv = pd.DataFrame(assistant_message["data"]).to_csv(index=False)
                        st.download_button(
                            label="📥 Download Data (CSV)",
                            data=csv,
                            file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                # Display visualization if available
                if assistant_message["visualization_path"] and os.path.exists(assistant_message["visualization_path"]):
                    st.image(assistant_message["visualization_path"])
                    
                    with open(assistant_message["visualization_path"], "rb") as file:
                        st.download_button(
                            label="📥 Download Visualization",
                            data=file,
                            file_name=os.path.basename(assistant_message["visualization_path"]),
                            mime="image/png"
                        )
                
                # Add assistant's response to chat history
                st.session_state.messages.append(assistant_message)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                session.close()

# Add a clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Show example queries in sidebar
with st.sidebar:
    st.markdown("### Example Queries")
    st.markdown("""
    **Inventory Queries:**
    - How many units of Amoxicillin do we have in stock?
    - Which products are below the alert threshold?
    
    **Visualization Queries:**
    - Show me a graph of inventory by category
    - Display sales trends for the last 3 months
    
    **Forecasting Queries:**
    - Predict inventory requirements for next quarter
    - Analyze vitamin sales trends
    """)
