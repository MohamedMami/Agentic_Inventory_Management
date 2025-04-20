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

def display_stock_by_category(session):
    stock_by_category = session.query(Product.category, func.sum(Inventory.current_quantity).label("total_quantity")).join(Inventory).group_by(Product.category).all()
    df = pd.DataFrame(stock_by_category, columns=["Category", "Quantity"])
    fig = px.pie(df, values="Quantity", names="Category", title="Stock by Category")
    st.plotly_chart(fig)

def display_expiry_tracking(session):
    expiring_products = session.query(
        Inventory.batch_id,
        Product.product_name,
        Inventory.expiry_date,
        Inventory.current_quantity,
        Inventory.warehouse
    ).join(Product).filter(Inventory.expiry_date <= pd.Timestamp.now() + pd.Timedelta(days=90)).all()
    
    df = pd.DataFrame(expiring_products, columns=["Batch ID", "Product", "Expiry Date", "Quantity", "Warehouse"])
    
    # Convert "Expiry Date" to pandas datetime
    df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors="coerce")
    
    # Current date as pandas Timestamp
    current_date = pd.Timestamp.now()
    
    # Calculate days until expiry
    df["Days Until Expiry"] = (df["Expiry Date"] - current_date).dt.days
    
    # Handle negative values (expired items)
    df["Days Until Expiry"] = df["Days Until Expiry"].clip(lower=-365)  # Optional
    
    # Apply styling
    st.dataframe(
        df.style.applymap(
            lambda x: "color: red" if x <= 30 else "color: orange" if 30 < x <= 60 else "",
            subset=["Days Until Expiry"]
        )
    )
def display_sales_trends(session):
    sales_trend = session.query(
        func.date_trunc('month', Sale.sale_date).label("month"),
        func.sum(Sale.total_value).label("total_sales")).group_by(func.date_trunc('month', Sale.sale_date)).order_by(func.date_trunc('month', Sale.sale_date)).all()
    df = pd.DataFrame(sales_trend, columns=["Month", "Total Sales"])
    fig = px.line(df, x="Month", y="Total Sales", title="Monthly Sales Trend")
    st.plotly_chart(fig)

def display_top_products(session):
    top_products = session.query(Sale.product_name,func.sum(Sale.total_value).label("total_sales")).group_by(Sale.product_name).order_by(desc("total_sales")).limit(10).all()
    df = pd.DataFrame(top_products, columns=["Product", "Total Sales"])
    fig = px.bar(df, x="Product", y="Total Sales", title="Top 10 Products by Sales")
    st.plotly_chart(fig)

# Main dashboard code
st.set_page_config(page_title="Pharmaceutical Dashboard", page_icon="📊", layout="wide")
st.title("Pharmaceutical Inventory Dashboard")

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

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stock Value", f"€{total_stock_value:,.2f}")
    col2.metric("Low Stock Products", low_stock_count)
    col3.metric("Expiring Soon", expiring_soon)

    # Row 1: Category Distribution + Expiry Tracking
    col1, col2 = st.columns(2)
    with col1:
        display_stock_by_category(session)
    with col2:
        display_expiry_tracking(session)

    # Row 2: Sales Trends + Top Products
    col1, col2 = st.columns(2)
    with col1:
        display_sales_trends(session)
    with col2:
        display_top_products(session)

finally:
    session.close()