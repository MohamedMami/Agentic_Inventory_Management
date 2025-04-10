# -*- coding: utf-8 -*-
from database import Session
from models import Product, Inventory, Sale
from sqlalchemy.exc import SQLAlchemyError
import sys

def test_database_connection():
    try:
        session = Session()
        
        # Test Inventories
        try:
            inventories = session.query(Inventory).all()
            if inventories:
                print(f"Found {len(inventories)} inventories")
                print(f"First warehouse: {inventories[0].warehouse}")
            else:
                print("No inventories found")
        except Exception as e:
            print(f"Error querying inventories: {e}")

        # Test Products
        try:
            products = session.query(Product).all()
            if products:
                print(f"Found {len(products)} products")
                print(f"First product: {products[0].product_name}")
            else:
                print("No products found")
        except Exception as e:
            print(f"Error querying products: {e}")

        # Test Sales
        try:
            sales = session.query(Sale).all()
            if sales:
                print(f"Found {len(sales)} sales")
                print(f"First sale date: {sales[-1].sale_date}")
            else:
                print("No sales found")
        except Exception as e:
            print(f"Error querying sales: {e}")

    except SQLAlchemyError as e:
        print(f"Database connection error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        session.close()

if __name__ == "__main__":
    test_database_connection()