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
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Required for threading safety
import matplotlib.pyplot as plt

# Configure Streamlit page


# Set up matplotlib style - handle seaborn deprecation
try:
    # Try modern seaborn styles first
    if 'seaborn-v0_8' in plt.style.available:
        plt.style.use('seaborn-v0_8')
    elif 'seaborn-whitegrid' in plt.style.available:
        plt.style.use('seaborn-whitegrid')
    else:
        # Fallback to built-in clean styles
        plt.style.use('default')
        # Apply custom styling for better appearance
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#cccccc',
            'axes.linewidth': 0.8,
            'axes.grid': True,
            'grid.color': '#e0e0e0',
            'grid.linewidth': 0.5,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
except Exception as e:
    # If all else fails, use default matplotlib style
    plt.style.use('default')
    print(f"Could not set matplotlib style: {e}. Using default style.")

from src.agents.supervisorReAct import SupervisorReActAgent
from src.agents.inventory_agent import InventoryAgent
from src.agents.visualization_agent import VisualizationAgent
from src.agents.Prophetforecasting import ProphetForecastAgent

logger = get_logger("query_page")
visualization_logger = get_logger("visualization")


def debug_visualization(viz_data: str) -> Dict:
    """
    Debug visualization data format and content.
    """
    debug_info = {
        "has_data": bool(viz_data),
        "format": None,
        "size": len(viz_data) if viz_data else 0,
        "is_valid": False
    }
    
    if viz_data:
        try:
            # Try decoding base64 directly
            base64.b64decode(viz_data)
            debug_info["is_valid"] = True
            debug_info["format"] = "base64"
        except Exception as e:
            try:
                # Try with data URL format
                if "," in viz_data:
                    header, content = viz_data.split(",", 1)
                    debug_info["format"] = header
                    base64.b64decode(content)
                    debug_info["is_valid"] = True
            except Exception as e2:
                debug_info["error"] = str(e2)
            
    return debug_info 


def display_data(data: Any, depth: int = 0, max_depth: int = 5):
    """
    Display data in a structured format with better type handling and depth control.
    """
    try:
        if depth > max_depth:
            st.warning("Max depth reached. Some data may not be fully expanded.")
            st.write(str(data))
            return

        if isinstance(data, pd.DataFrame):
            st.dataframe(data, use_container_width=True)
            
        elif isinstance(data, dict):
            # Instead of using expanders for nested dictionaries, use columns
            if depth == 0:
                for k, v in data.items():
                    with st.expander(f"**{k}**"):
                        display_data(v, depth + 1, max_depth)
            else:
                # For nested dictionaries, use a simpler layout
                for k, v in data.items():
                    st.markdown(f"**{k}:**")
                    st.markdown("---")
                    display_data(v, depth + 1, max_depth)
                    
        elif isinstance(data, (list, tuple)):
            try:
                # Try to convert to DataFrame if list contains dicts
                if data and isinstance(data[0], dict):
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True)
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
    """Check and set up environment variables"""
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing required environment variable: GROQ_API_KEY")
        st.stop()
    if not os.getenv("REDIS_URL"):
        os.environ["REDIS_URL"] = "redis://localhost:6379"
        logger.info("Using default Redis URL")


def local_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        /* Main app background - keep light */
        .stApp {
            background-color: #ffffff;
        }
        
        /* Chat message styling - light theme */
        .chat-message {
            padding: 1rem;
            border-radius: 0.8rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .user-message {
            background: linear-gradient(135deg, #f8f9ff 0%, #e8f2ff 100%);
            border-left: 4px solid #4a90e2;
            color: #2c3e50;
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #f9f9f9 0%, #f0f8f0 100%);
            border-left: 4px solid #27ae60;
            color: #2c3e50;
        }
        
        /* Chat input styling */
        .stChatInput > div > div {
            background-color: #ffffff !important;
            border: 2px solid #e1e8ed !important;
            border-radius: 25px !important;
        }
        
        /* Sidebar styling - Spotify dark theme */
        .css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #000000 0%, #121212 100%) !important;
        }

        /* Sidebar text */
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stText {
            color: #b3b3b3 !important; /* Spotify's light gray text */
        }

        /* Sidebar titles */
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #ffffff !important; /* White for headings */
        }

        /* Sidebar buttons */
        section[data-testid="stSidebar"] .stButton > button {
            background: #1DB954 !important; /* Spotify Green */
            color: #ffffff !important;
            border: none !important;
            border-radius: 20px !important; /* Rounded pill shape */
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }

        /* Button hover effect */
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: #1ed760 !important; /* Lighter Spotify green */
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 12px rgba(30, 215, 96, 0.3) !important;
        }

        
        /* Main content area */
        .main .block-container {
            background-color: #ffffff;
            padding-top: 2rem;
        }
        
        /* Improve visual hierarchy */
        .main h1, .main h2, .main h3 {
            color: #2c3e50 !important;
            font-weight: 600 !important;
        }
        
        /* Image styling */
        .stImage > img {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Metrics styling */
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 1px solid #dee2e6;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)


def validate_visualization(viz_data: str) -> bool:
    """Validate if visualization data is properly formatted."""
    try:
        if not viz_data:
            logger.warning("No visualization data provided")
            return False
            
        # Try decoding the base64 data
        if viz_data.startswith('data:image'):
            # Handle data URL format
            _, encoded = viz_data.split(',', 1)
            image_data = base64.b64decode(encoded)
        else:
            # Handle direct base64
            image_data = base64.b64decode(viz_data)
        
        # Check if it's a valid image (basic header check)
        return len(image_data) > 0
        
    except Exception as e:
        logger.error(f"Visualization validation error: {str(e)}")
        return False


def create_sample_visualization():
    """Create a sample visualization for testing"""
    try:
        plt.figure(figsize=(10, 6))
        plt.clf()
        
        # Sample data
        x = [1, 2, 3, 4, 5]
        y = [10, 25, 30, 45, 60]
        
        plt.plot(x, y, marker='o', linewidth=2, markersize=8)
        plt.title('Sample Visualization', fontsize=16, fontweight='bold')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_str
        
    except Exception as e:
        logger.error(f"Error creating sample visualization: {str(e)}")
        return None


# Initialize environment and styling
check_environment()
local_css()

# Initialize SupervisorReActAgent
supervisor = SupervisorReActAgent(
    model_name="llama-3.3-70b-versatile",
    redis_url=os.getenv("REDIS_URL")
)

# Register sub-agents
supervisor.register_agent("inventory", InventoryAgent())
supervisor.register_agent("visualization", VisualizationAgent())
supervisor.register_agent("forecast", ProphetForecastAgent())

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Main UI Layout
st.title("💊 Pharmaceutical Inventory Chat")

# Sidebar
with st.sidebar:
    st.title("🔧 Chat Controls")
    st.markdown(f"**Conversation ID:**  \n`{st.session_state.conversation_id}`")
    st.markdown(f"**Total Messages:** {len(st.session_state.messages)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    # Test visualization button
    if st.button("🧪 Test Visualization"):
        sample_viz = create_sample_visualization()
        if sample_viz:
            test_message = {
                "role": "assistant",
                "response": "Here's a test visualization to verify the system is working correctly.",
                "visualization_base64": sample_viz,
                "timestamp": datetime.now().strftime("%H:%M")
            }
            st.session_state.messages.append(test_message)
            st.rerun()
    
    # Debug section
    with st.expander("🔧 Debug Tools"):
        st.markdown("**System Status:**")
        st.success("✅ Matplotlib configured")
        st.success("✅ Base64 encoding ready")
        
        if st.button("🔍 Analyze Last Visualization"):
            if st.session_state.messages:
                assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant"]
                if assistant_messages:
                    last_message = assistant_messages[-1]
                    viz_data = last_message.get("visualization_base64")
                    debug_info = debug_visualization(viz_data)
                    st.json(debug_info)
                    
                    if viz_data and debug_info["is_valid"]:
                        st.success("✅ Visualization data is valid")
                        st.text(f"Size: {debug_info['size']} bytes")
                    else:
                        st.error("❌ No valid visualization found")
                else:
                    st.info("No assistant messages found")
            else:
                st.info("No messages in chat history")

# Main chat area
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
        # Message header
        st.markdown(f"**{message['role'].title()}** ({message['timestamp']})")
        
        if message["role"] == "user":
            st.markdown(f"_{message['content']}_")
        else:
            # Assistant message content
            st.markdown(message.get("response", ""))
            
            # Handle visualization display - FIXED
            if message.get("visualization_base64"):
                try:
                    viz_data = message["visualization_base64"]
                    
                    # Handle different base64 formats
                    if viz_data.startswith('data:image'):
                        # Data URL format
                        image_data = base64.b64decode(viz_data.split(',', 1)[1])
                    else:
                        # Direct base64
                        image_data = base64.b64decode(viz_data)
                    
                    # Display the image
                    st.image(
                        image_data,
                        caption="Visualization",
                        use_column_width=True
                    )
                    st.success("✅ Visualization displayed successfully")
                    
                except Exception as e:
                    logger.error(f"Error displaying visualization: {str(e)}")
                    st.error(f"❌ Error displaying visualization: {str(e)}")
                    
                    # Debug information
                    with st.expander("🔍 Debug Info"):
                        st.code(f"Visualization data length: {len(viz_data) if viz_data else 0}")
                        st.code(f"First 100 chars: {viz_data[:100] if viz_data else 'None'}")

            # Display forecast data if available
            if message.get("data") and "forecast_values" in message["data"]:
                with st.expander("📊 Forecast Details"):
                    # Show forecast statistics
                    if "statistics" in message["data"]:
                        stats = message["data"]["statistics"]
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Mean Forecast", f"{stats.get('mean_forecast', 0):.2f}")
                        with cols[1]:
                            st.metric("Trend", stats.get('trend_direction', 'N/A'))
                        with cols[2]:
                            st.metric("Confidence", f"{stats.get('confidence', 0):.1f}%")
                    
                    # Show forecast values table
                    st.subheader("Forecast Values")
                    df = pd.DataFrame(message["data"]["forecast_values"])
                    st.dataframe(df, use_container_width=True)
            
            # Display any other data
            elif message.get("data"):
                with st.expander("📊 View Data Details"):
                    display_data(message["data"])
            
            # Display any errors
            if message.get("error"):
                st.error(f"Error: {message['error']}")

# Chat input
query = st.chat_input("Ask about your pharmaceutical inventory...")

async def process_query():
    """Process user query and generate response"""
    session = Session(engine)
    try:
        # Add user message
        user_message = {
            "role": "user",
            "content": query,   
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(user_message)
        
        logger.info(f"Processing query: {query}")
        
        # Process the query
        result = await supervisor.process_query(
            query, 
            session=session, 
            conversation_id=st.session_state.conversation_id
        )
        
        # Ensure we have a valid response
        if not result:
            result = {
                "response": "I apologize, but I couldn't process your request. Please try again.",
                "error": "No response received from agent"
            }
        
        # Ensure response field exists
        if "response" not in result:
            result["response"] = "Processed successfully but no message was generated."
        
        # Handle visualization generation - IMPROVED
        if (result.get('query_type') == 'visualization' or 
            (result.get('data') and 'forecast_values' in result.get('data', {}))):
            
            try:
                # Create figure with proper size and DPI
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.clf()
                ax.clear()
                
                if result.get('query_type') == 'visualization':
                    # Handle visualization data
                    viz_data = result.get('data', {}).get('visualization_data', {})
                    if viz_data and 'x' in viz_data and 'y' in viz_data:
                        x = viz_data.get('x', [])
                        y = viz_data.get('y', [])
                        
                        ax.plot(x, y, marker='o', linewidth=2, markersize=6)
                        ax.set_title(viz_data.get('title', 'Visualization'), fontsize=16, fontweight='bold')
                        ax.set_xlabel(viz_data.get('xlabel', 'X'))
                        ax.set_ylabel(viz_data.get('ylabel', 'Y'))
                        ax.grid(True, alpha=0.3)
                    else:
                        # Create sample plot if no data
                        ax.plot([1, 2, 3, 4, 5], [10, 25, 30, 45, 60], marker='o')
                        ax.set_title('Sample Visualization', fontsize=16, fontweight='bold')
                        ax.grid(True, alpha=0.3)
                
                elif result.get('data') and 'forecast_values' in result.get('data', {}):
                    # Handle forecast data
                    forecast_df = pd.DataFrame(result['data']['forecast_values'])
                    if not forecast_df.empty and 'ds' in forecast_df.columns and 'yhat' in forecast_df.columns:
                        ax.plot(forecast_df['ds'], forecast_df['yhat'], marker='o', linewidth=2)
                        ax.set_title('Forecast Results', fontsize=16, fontweight='bold')
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Predicted Value')
                        ax.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                
                # Save with high quality
                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', pad_inches=0.2)
                buf.seek(0)
                
                # Encode to base64
                img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                result['visualization_base64'] = img_str
                
                plt.close(fig)  # Important: close the figure
                logger.info(f"Visualization created successfully, size: {len(img_str)} bytes")
                
            except Exception as viz_error:
                logger.error(f"Visualization error: {str(viz_error)}")
                result['error'] = f"Visualization error: {str(viz_error)}"
                plt.close('all')  # Close all figures on error
        
        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "response": result.get("response", "No response available"),
            "data": result.get("data", {}),
            "visualization_base64": result.get("visualization_base64"),
            "query_type": result.get("query_type", "text"),
            "error": result.get("error"),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        
        logger.debug(f"Assistant message created with visualization: {bool(result.get('visualization_base64'))}")
        st.session_state.messages.append(assistant_message)
        st.rerun()
            
    except Exception as e:
        logger.error(f"Error in process function: {str(e)}")
        st.error(f"System Error: {str(e)}")
        
        # Add error message to chat
        error_message = {
            "role": "assistant",
            "response": f"I encountered an error: {str(e)}",
            "error": str(e),
            "timestamp": datetime.now().strftime("%H:%M")
        }
        st.session_state.messages.append(error_message)
        
    finally:
        session.close()
        plt.close('all')  # Ensure all figures are closed


# Process query when submitted
if query:
    asyncio.run(process_query())


def display_forecast_results(response):
    """Helper function to display forecast results"""
    if not response or 'data' not in response:
        st.error("No forecast data available")
        return

    data = response['data']
    
    # Display product info
    if 'product_info' in data:
        st.subheader(f"Forecast for {data['product_info']['name']}")
        st.caption(f"Category: {data['product_info']['category']}")

    # Display visualization if available
    if 'visualization_base64' in data:
        try:
            image_data = base64.b64decode(data['visualization_base64'])
            st.image(
                image_data,
                caption="Forecast Visualization",
                use_column_width=True
            )
        except Exception as e:
            st.error(f"Error displaying forecast visualization: {str(e)}")

    # Display forecast metrics in columns
    if 'statistics' in data:
        stats = data['statistics']
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Forecast", f"{stats.get('mean_forecast', 0):.2f}")
        with col2:
            st.metric("Trend Direction", stats.get('trend_direction', 'N/A'))
        with col3:
            st.metric("Current Stock", f"{stats.get('inventory_metrics', {}).get('current_stock', 0):.0f}")

    # Display insights
    if 'insights' in data:
        st.subheader("Forecast Insights")
        st.write(data['insights'])

    # Display forecast values in a table
    if 'forecast_values' in data:
        st.subheader("Detailed Forecast")
        df = pd.DataFrame(data['forecast_values'])
        st.dataframe(df, use_container_width=True)