# -*- coding: utf-8 -*-
from database import Session
from models import Product, Inventory, Sale
from sqlalchemy.exc import SQLAlchemyError
# Create a session
try:
    session = Session()
except SQLAlchemyError as e:
    print(f"Database connection error: {e}")
    exit(1)

# Check inventories
inventories = session.query(Inventory).all()
if inventories:
    print(inventories[0].warehouse)
else:
    print("No inventories found.")

# Check products
products = session.query(Product).all()
if products:
    print(products[0].product_name)
else:
    print("No products found.")

# Check sales
sales = session.query(Sale).all()
if sales:
    # Assuming Sale has a relationship to Product
    print(sales[0].product_name)  # Replace with sales[0].product.product_name if needed
    print(sales[0].sale_date)
else:
    print("No sales found.")