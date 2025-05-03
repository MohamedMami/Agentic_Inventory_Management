import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import streamlit as st
import pandas as pd
from typing import Any, Dict
import base64
from datetime import datetime
from sqlalchemy.orm import Session
from src.database import engine
from src.integration import SystemIntegration
from src.logger import get_logger
import asyncio

logger = get_logger("query_page")

# Initialize SystemIntegration
system = SystemIntegration()

st.set_page_config(
    page_title="Pharmaceutical Query Interface",
    page_icon="💊",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

st.title("Pharmaceutical Inventory Chat")

# Chat history display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            st.markdown(message.get("response", "No response"))
            
            # Display data if available
            if message.get("data"):
                st.subheader("Data")
                display_data(message["data"])
                
            # Display visualization
            if message.get("visualization_base64"):
                st.image(
                    message["visualization_base64"], 
                    caption="Generated Visualization",
                    use_column_width=True
                )

# Chat input
query = st.chat_input("Ask about your pharmaceutical inventory...")

if query:
    async def process():
        session = Session(engine)
        try:
            result = await system.process_query(
                query, 
                session=session, 
                conversation_id=st.session_state.conversation_id
            )
            
            assistant_message = {
                "role": "assistant",
                "response": result.get("response", "No response"),
                "data": result.get("data"),
                "visualization_base64": result.get("visualization_base64"),
                "query_type": result.get("query_type", "unknown")
            }
            
            # Display response
            st.markdown(assistant_message["response"])
            
            # Handle data
            if assistant_message.get("data"):
                display_data(assistant_message["data"])
                
            # Handle visualization
            if assistant_message.get("visualization_base64"):
                st.image(
                    assistant_message["visualization_base64"],
                    use_column_width=True
                )
                # Download button
                img_data = base64.b64decode(
                    assistant_message["visualization_base64"].split(",")[1]
                )
                st.download_button(
                    label="📥 Download Visualization",
                    data=img_data,
                    file_name=f"viz_{datetime.now().strftime('%Y%m%d%H%M%S')}.png",
                    mime="image/png"
                )
            
            # Handle errors
            if result.get("error"):
                st.error(f"Error: {result['error']}")
            
            # Save to session state
            st.session_state.messages.append(assistant_message)
            
        except Exception as e:
            st.error(f"System Error: {str(e)}")
        finally:
            session.close()
    
    # Run the async function
    asyncio.run(process())

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    st.rerun()

# Example queries
with st.sidebar:
    st.markdown("### Example Queries")
    st.markdown("""
    **Inventory Queries:**
    - "Show all antibiotics expiring in next 30 days"
    - "Which products are below minimum stock levels?"
    
    **Visualization Queries:**
    - "Plot sales by category for Q2 2023"
    - "Compare sales of Paracetamol vs Ibuprofen"
    
    **Forecast Queries:**
    - "Forecast demand for vaccines in next 60 days"
    - "Predict sales of diabetes medications for Q4"
    
    **Composite Queries:**
    - "Analyze antibiotic stock and forecast next month's demand"
    - "Visualize sales trends and predict inventory needs for antivirals"
    """)

# Helper function for recursive data display
def display_data(data: Any, depth: int = 0, max_depth: int = 5):
    """
    Display data in a structured format with better type handling and depth control.
    
    Args:
        data: The data to display
        depth: Current recursion depth
        max_depth: Maximum recursion depth
    """
    try:
        if depth > max_depth:
            st.warning("Max depth reached. Some data may not be fully expanded.")
            st.write(str(data))
            return

        if isinstance(data, pd.DataFrame):
            st.dataframe(data)
            
        elif isinstance(data, dict):
            for k, v in data.items():
                with st.expander(f"**{k}**", expanded=(depth == 0)):
                    display_data(v, depth + 1, max_depth)
                    
        elif isinstance(data, (list, tuple)):
            try:
                # Try to convert to DataFrame if list contains dicts
                if data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                else:
                    # Display as regular list if conversion fails
                    st.write(data)
            except Exception:
                st.write(data)
                
        elif isinstance(data, (int, float)):
            # Show numerical data with metric
            st.metric(label="Value", value=data)
            
        elif isinstance(data, bool):
            # Show boolean as checkbox
            st.checkbox("Value", value=data, disabled=True)
            
        elif data is None:
            st.info("No data available")
            
        else:
            # Default fallback for other types
            st.write(data)
            
    except Exception as e:
        logger.error(f"Error displaying data: {str(e)}")
        st.error(f"Error displaying data: {str(e)}")