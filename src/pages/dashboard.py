# Dashboard page (metrics, charts)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sqlalchemy import func, desc
from sqlalchemy.orm import Session
from src.database import engine
from src.models import product, inventory , sale 

def dashboard_page():
    st.title("Pharmaceutical Inventory Dashboard")
    session = Session(engine)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Stock Value", f"€{total_stock_value:,.2f}")
    col2.metric("Low Stock Products", low_stock_count)
    col3.metric("Expiring Soon", expiring_soon)
    # col4.metric("Total Sales", f"€{total_sales:,.2f}")

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

    # # Row 3: Warehouse Stock + Profitability
    # col1, col2 = st.columns(2)
    # with col1:
    #     display_warehouse_stock(session)
    # with col2:
    #     display_profitability(session)

    
    # Calculate total stock value
    total_stock_value = session.query(func.sum(product.unit_price * inventory.current_quantity)).join(inventory).scalar() or 0

    # Products below min stock level
    low_stock_count = session.query(inventory).filter(inventory.current_quantity < product.min_stock_level).join(product).count()

    # Expiring soon
    expiring_soon = session.query(inventory).filter(inventory.expiry_date <= datetime.now() + timedelta(days=30)).count()
    
    # Stock Distribution by Category
    def display_stock_by_category(session):
        stock_by_category = session.query(product.category, func.sum(inventory.current_quantity).label("total_quantity")).join(inventory).group_by(product.category).all()
        df = pd.DataFrame(stock_by_category, columns=["Category", "Quantity"])
        fig = px.pie(df, values="Quantity", names="Category", title="Stock by Category")
        st.plotly_chart(fig)
    def display_expiry_tracking(session):
        expiring_products = session.query(inventory.batch_id,product.product_name,inventory.expiry_date,inventory.current_quantity,inventory.warehouse).join(product).filter(inventory.expiry_date <= datetime.now() + timedelta(days=90)).all()
        df = pd.DataFrame(expiring_products, columns=["Batch ID", "Product", "Expiry Date", "Quantity", "Warehouse"])
        df["Days Until Expiry"] = (df["Expiry Date"] - datetime.now()).dt.days
        st.dataframe(df.style.applymap(
                lambda x: "color: red" if x <= 30 else "color: orange" if x <= 60 else "",
                subset=["Days Until Expiry"]
            ))
    def display_sales_trends(session):
        # Monthly sales trend
        sales_trend = session.query(
            func.date_trunc('month', sale.sale_date).label("month"),
            func.sum(sale.total_value).label("total_sales")
        ).group_by("month").order_by("month").all()

        df = pd.DataFrame(sales_trend, columns=["Month", "Total Sales"])
        fig = px.line(df, x="Month", y="Total Sales", title="Monthly Sales Trend")
        st.plotly_chart(fig)
    
    def display_top_products(session):
        # Top 10 products by sales
        top_products = session.query(
            sale.product_name,
            func.sum(sale.total_value).label("total_sales")
        ).group_by(sale.product_name).order_by(desc("total_sales")).limit(10).all()

        df = pd.DataFrame(top_products, columns=["Product", "Total Sales"])
        fig = px.bar(df, x="Product", y="Total Sales", title="Top 10 Products by Sales")
        st.plotly_chart(fig)
        
                
