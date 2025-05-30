import sys
import os
# Add the src directory to PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from src.database import engine
from src.models import Product, Inventory, Sale

# Spotify Color Palette
SPOTIFY_COLORS = {
    'primary': '#1DB954',      # Spotify Green
    'secondary': '#1ed760',    # Light Green
    'dark': '#191414',         # Almost Black
    'darker': '#121212',       # Pure Black
    'gray': '#535353',         # Medium Gray
    'light_gray': '#b3b3b3',  # Light Gray
    'white': '#ffffff',        # White
    'red': '#e22134',          # Spotify Red
    'orange': '#ff7300',       # Warning Orange
    'blue': '#2e77d0'          # Info Blue
}

# Custom CSS for Spotify Theme
SPOTIFY_CSS = f"""
<style>
    /* Main App Background */
    .stApp {{
        background: linear-gradient(135deg, {SPOTIFY_COLORS['darker']} 0%, {SPOTIFY_COLORS['dark']} 100%);
        color: {SPOTIFY_COLORS['white']};
    }}
    
    /* Hide Streamlit Menu and Footer */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Custom Title Styling */
    .main-title {{
        font-size: 3rem;
        font-weight: 900;
        color: {SPOTIFY_COLORS['primary']};
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(29, 185, 84, 0.3);
        font-family: 'Helvetica Neue', sans-serif;
    }}
    
    /* Metric Cards Styling */
    .metric-card {{
        background: linear-gradient(145deg, {SPOTIFY_COLORS['dark']} 0%, #252525 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(29, 185, 84, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(29, 185, 84, 0.2);
        border-color: {SPOTIFY_COLORS['primary']};
    }}
    
    .metric-value {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {SPOTIFY_COLORS['primary']};
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 1rem;
        color: {SPOTIFY_COLORS['light_gray']};
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* Section Headers */
    .section-header {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {SPOTIFY_COLORS['white']};
        margin: 2rem 0 1rem 0;
        padding-left: 1rem;
        border-left: 4px solid {SPOTIFY_COLORS['primary']};
    }}
    
    /* Chart Container */
    .chart-container {{
        background: rgba(25, 20, 20, 0.6);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(29, 185, 84, 0.1);
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }}
    
    /* Sidebar Styling */
    .css-1d391kg {{
        background: linear-gradient(180deg, {SPOTIFY_COLORS['darker']} 0%, {SPOTIFY_COLORS['dark']} 100%);
    }}
    
    /* Custom Button Styling */
    .stButton > button {{
        background: linear-gradient(145deg, {SPOTIFY_COLORS['primary']} 0%, {SPOTIFY_COLORS['secondary']} 100%);
        color: {SPOTIFY_COLORS['white']};
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(29, 185, 84, 0.4);
    }}
    
    /* Dataframe Styling */
    .dataframe {{
        background-color: rgba(25, 20, 20, 0.8) !important;
        border-radius: 8px;
    }}
    
    /* Plotly Chart Background Fix */
    .js-plotly-plot {{
        background: transparent !important;
    }}
</style>
"""

def get_spotify_plotly_theme():
    """Return Spotify-themed Plotly template"""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {'color': SPOTIFY_COLORS['white'], 'family': 'Helvetica Neue'},
            'colorway': [SPOTIFY_COLORS['primary'], SPOTIFY_COLORS['secondary'], 
                        SPOTIFY_COLORS['blue'], SPOTIFY_COLORS['orange'], 
                        SPOTIFY_COLORS['red'], SPOTIFY_COLORS['light_gray']],
            'title': {
                'font': {'size': 20, 'color': SPOTIFY_COLORS['white']},
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'gridcolor': 'rgba(83, 83, 83, 0.3)',
                'tickfont': {'color': SPOTIFY_COLORS['light_gray']},
                'title': {'font': {'color': SPOTIFY_COLORS['white']}}
            },
            'yaxis': {
                'gridcolor': 'rgba(83, 83, 83, 0.3)',
                'tickfont': {'color': SPOTIFY_COLORS['light_gray']},
                'title': {'font': {'color': SPOTIFY_COLORS['white']}}
            },
            'legend': {
                'font': {'color': SPOTIFY_COLORS['white']},
                'bgcolor': 'rgba(0,0,0,0)'
            }
        }
    }

def create_metric_card(title, value, icon="📊"):
    """Create a custom metric card with Spotify styling"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
    </div>
    """

def display_stock_by_category(session):
    """Display stock distribution by category with Spotify theme"""
    stock_by_category = session.query(
        Product.category, 
        func.sum(Inventory.current_quantity).label("total_quantity")
    ).join(Inventory).group_by(Product.category).all()
    
    df = pd.DataFrame(stock_by_category, columns=["Category", "Quantity"])
    
    fig = px.pie(
        df, 
        values="Quantity", 
        names="Category", 
        title="📦 Stock Distribution by Category",
        color_discrete_sequence=[SPOTIFY_COLORS['primary'], SPOTIFY_COLORS['secondary'], 
                               SPOTIFY_COLORS['blue'], SPOTIFY_COLORS['orange']]
    )
    
    fig.update_layout(get_spotify_plotly_theme()['layout'])
    fig.update_traces(
        textfont_size=14,
        marker=dict(line=dict(color=SPOTIFY_COLORS['white'], width=2))
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_expiry_tracking(session):
    """Display expiry tracking with enhanced styling"""
    st.markdown('<div class="section-header">⏰ Expiry Tracking</div>', unsafe_allow_html=True)
    
    expiring_products = session.query(
        Inventory.batch_id,
        Product.product_name,
        Inventory.expiry_date,
        Inventory.current_quantity,
        Inventory.warehouse
    ).join(Product).filter(
        Inventory.expiry_date <= pd.Timestamp.now() + pd.Timedelta(days=90)
    ).all()
    
    if expiring_products:
        df = pd.DataFrame(expiring_products, columns=["Batch ID", "Product", "Expiry Date", "Quantity", "Warehouse"])
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors="coerce")
        current_date = pd.Timestamp.now()
        df["Days Until Expiry"] = (df["Expiry Date"] - current_date).dt.days
        df["Days Until Expiry"] = df["Days Until Expiry"].clip(lower=-365)
        
        # Style the dataframe
        def color_expiry(val):
            if val <= 0:
                return f'background-color: {SPOTIFY_COLORS["red"]}; color: white; font-weight: bold'
            elif val <= 30:
                return f'background-color: {SPOTIFY_COLORS["orange"]}; color: white; font-weight: bold'
            elif val <= 60:
                return f'background-color: rgba(255, 115, 0, 0.3); color: {SPOTIFY_COLORS["white"]}'
            else:
                return f'color: {SPOTIFY_COLORS["light_gray"]}'
        
        styled_df = df.style.applymap(color_expiry, subset=["Days Until Expiry"])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.success("🎉 No products expiring in the next 90 days!")

def display_sales_trends(session):
    """Display sales trends with Spotify styling"""
    sales_trend = session.query(
        func.date_trunc('month', Sale.sale_date).label("month"),
        func.sum(Sale.total_value).label("total_sales")
    ).group_by(
        func.date_trunc('month', Sale.sale_date)
    ).order_by(
        func.date_trunc('month', Sale.sale_date)
    ).all()
    
    df = pd.DataFrame(sales_trend, columns=["Month", "Total Sales"])
    
    fig = px.line(
        df, 
        x="Month", 
        y="Total Sales", 
        title="📈 Monthly Sales Trend",
        line_shape="spline"
    )
    
    fig.update_layout(get_spotify_plotly_theme()['layout'])
    fig.update_traces(
        line_color=SPOTIFY_COLORS['primary'],
        line_width=4,
        marker=dict(size=8, color=SPOTIFY_COLORS['primary'])
    )
    
    # Add gradient fill
    fig.add_trace(go.Scatter(
        x=df["Month"], 
        y=df["Total Sales"],
        fill='tonexty',
        mode='none',
        fillcolor=f'rgba(29, 185, 84, 0.1)',
        showlegend=False
    ))
    
    st.plotly_chart(fig, use_container_width=True)

def display_top_products(session):
    """Display top products with enhanced bar chart"""
    top_products = session.query(
        Sale.product_name,
        func.sum(Sale.total_value).label("total_sales")
    ).group_by(
        Sale.product_name
    ).order_by(
        desc("total_sales")
    ).limit(10).all()
    
    df = pd.DataFrame(top_products, columns=["Product", "Total Sales"])
    
    fig = px.bar(
        df, 
        x="Product", 
        y="Total Sales", 
        title="🏆 Top 10 Products by Sales",
        color="Total Sales",
        color_continuous_scale=[[0, SPOTIFY_COLORS['dark']], 
                              [0.5, SPOTIFY_COLORS['primary']], 
                              [1, SPOTIFY_COLORS['secondary']]]
    )
    
    fig.update_layout(get_spotify_plotly_theme()['layout'])
    fig.update_traces(
        marker_line_color=SPOTIFY_COLORS['white'],
        marker_line_width=1
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig, use_container_width=True)

# Page Configuration (Must be first Streamlit command)
st.set_page_config(
    page_title="Pharma Dashboard", 
    page_icon="💊", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Main Dashboard
def main():
    # Apply custom CSS
    st.markdown(SPOTIFY_CSS, unsafe_allow_html=True)
    
    # Main title with custom styling
    st.markdown('<h1 class="main-title">💊 Pharma Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #b3b3b3; font-size: 1.2rem; margin-bottom: 3rem;">Real-time pharmaceutical inventory management</p>', unsafe_allow_html=True)
    
    session = Session(engine)
    
    try:
        # Calculate metrics
        total_stock_value = session.query(
            func.sum(Product.unit_price * Inventory.current_quantity)
        ).join(Inventory).scalar() or 0

        low_stock_count = session.query(Inventory).filter(
            Inventory.current_quantity < Product.min_stock_level
        ).join(Product).count()

        expiring_soon = session.query(Inventory).filter(
            Inventory.expiry_date <= datetime.now() + timedelta(days=30)
        ).count()

        total_products = session.query(Product).count()

        # Display metrics in custom cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card("Total Stock Value", f"€{total_stock_value:,.2f}", "💰"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card("Low Stock Alerts", str(low_stock_count), "⚠️"), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card("Expiring Soon", str(expiring_soon), "⏰"), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card("Total Products", str(total_products), "📦"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 1: Category Distribution + Sales Trends
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                display_stock_by_category(session)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                display_sales_trends(session)
                st.markdown('</div>', unsafe_allow_html=True)

        # Row 2: Top Products + Expiry Tracking
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                display_top_products(session)
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                display_expiry_tracking(session)
                st.markdown('</div>', unsafe_allow_html=True)

    finally:
        session.close()

if __name__ == "__main__":
    main()