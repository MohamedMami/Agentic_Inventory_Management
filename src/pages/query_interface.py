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
from src.logger import get_logger
import asyncio


from src.agents.supervisorReAct import SupervisorReActAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.visualization_agent import VisualizationAgent

logger = get_logger("query_page")
visualization_logger = get_logger("visualization")

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

def check_environment():
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing required environment variable: GROQ_API_KEY")
        st.stop()
    if not os.getenv("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        logger.info("Using default Redis URL")

check_environment()

# Add these style definitions after the imports
def local_css():
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #2196f3;
        }
        .assistant-message {
            background-color: #f3f3f3;
            border-left: 5px solid #4caf50;
        }
        .message-content {
            margin: 0;
            padding: 0;
        }
        .message-timestamp {
            font-size: 0.8rem;
            color: #666;
            align-self: flex-end;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize SupervisorReActAgent
supervisor = SupervisorReActAgent(
    model_name="llama-3.3-70b-versatile",
    redis_url=os.getenv("REDIS_URL")
)

# Register sub-agents
supervisor.register_agent("inventory", InventoryAgent())
supervisor.register_agent("visualization", VisualizationAgent())
#supervisor.register_agent("forecast", ForecastAgent())

# Set up Streamlit page configuration
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
local_css()

# Create two columns for a better chat layout
col1, col2 = st.columns([3, 1])

with col1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().strftime("%H:%M")

            # Display message content
            st.markdown(f"**{message['role'].title()}** ({message['timestamp']})")
            if message["role"] == "user":
                st.markdown(f"_{message['content']}_")
            else:
                st.markdown(message.get("response", ""))
                
                # Display any additional data
                if message.get("data"):
                    with st.expander("📊 View Data Details"):
                        display_data(message["data"])
                        
                # Display visualization if available
                if message.get("visualization_base64"):
                    st.image(message["visualization_base64"], use_column_width=True)
                    img_data = base64.b64decode(
                        message["visualization_base64"].split(",")[1]
                    )
                    st.download_button(
                        label="📥 Download Visualization",
                        data=img_data,
                        file_name=f"viz_{datetime.now().strftime('%Y%m%d%H%M%S')}.png",
                        mime="image/png"
                    )
                
                # Display any errors
                if message.get("error"):
                    st.error(f"Error: {message['error']}")

with col2:
    # Add a sidebar with conversation info
    st.sidebar.title("Chat Info")
    st.sidebar.markdown(f"**Conversation ID:**  \n`{st.session_state.conversation_id}`")
    st.sidebar.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    # Add a clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat input and processing
query = st.chat_input("Ask about your pharmaceutical inventory...")

# Modify the process function to include logging
async def process():
    session = Session(engine)
    try:
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(user_message)
        
        # Add logging before processing
        logger.info(f"Processing query: {query}")
        
        result = await supervisor.process_query(
            query,
            session=session,
            conversation_id=st.session_state.conversation_id
        )
        
        # Add logging for visualization check
        if "visualization_base64" in result:
            logger.info("Visualization found in result")
        else:
            logger.warning("No visualization found in result")
            
        assistant_message = {
            "role": "assistant",
            "response": result.get("response", "No response"),
            "data": result.get("data"),
            "visualization_base64": result.get("visualization_base64"),
            "query_type": result.get("query_type", "unknown"),
            "error": result.get("error"),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        
        st.session_state.messages.append(assistant_message)
        st.rerun()
            
    except Exception as e:
        logger.error(f"Error in process function: {str(e)}")
        st.error(f"System Error: {str(e)}")
    finally:
        session.close()

if query:
    asyncio.run(process())
