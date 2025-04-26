# Main app entry point
import streamlit as st
import os
from datetime import datetime

from pages import dashboard, query_interface
from integration import system

st.set_page_config(
    page_title="Agentic Inventory Management",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"  # Ensures the sidebar is open by default
)

def main():
    """User interface for the system."""
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

    # Navigation menu with explicit default (dashboard first in options)
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Query Interface"])

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