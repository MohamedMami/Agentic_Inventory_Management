# Main app entry point
import streamlit as st
import os
from datetime import datetime

from pages import dashboard, query_interface

# Ensure page config is first Streamlit command
st.set_page_config(
    page_title="Agentic Inventory Management",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """User interface for the system."""
    # Initialize default page if not set
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    # Sidebar setup
    st.sidebar.title("Smart Pharma Inventory")
    st.sidebar.image("https://img.icons8.com/color/96/000000/pharmacy-shop.png", width=100)
    
    # Sidebar info and date
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Smart Pharma Inventory** is a pharmaceutical inventory management system
    that uses artificial intelligence to process natural language queries
    and generate real-time insights.
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}")

    # Navigation menu with Dashboard as default
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Query Interface"], 
                           index=0,  # Explicitly set Dashboard as default
                           key="navigation")

    # Route based on selection
    if page == "Dashboard":
        dashboard()
    elif page == "Query Interface":
        query_interface()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")