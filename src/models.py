from sqlalchemy import Column, String, Boolean, Numeric, Integer, Date, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class product(Base):
    __tablename__ = 'products'

    product_id = Column(String, primary_key=True)
    product_name = Column(String, nullable=False)
    generic_name = Column(String)
    strength_form = Column(String)
    category = Column(String)
    manufacturer = Column(String)
    atc_code = Column(String)
    storage_instructions = Column(String)
    prescription_required = Column(Boolean)
    controlled_substance_class = Column(String)
    requires_refrigeration = Column(Boolean)
    unit_price = Column(Numeric(10, 2))  # For currency
    min_stock_level = Column(Integer)
    reorder_lead_time_days = Column(Integer)
    registration_number = Column(String)
    approval_authority = Column(String)
    registration_date = Column(Date)
    renewal_date = Column(Date)
    package_size = Column(String)
    production_line = Column(String)
    production_capacity_units_per_day = Column(Integer)
    quality_control_sampling_rate = Column(Numeric(5, 2))
    barcode = Column(String)
    
    # Relationships
    inventories = relationship("Inventory", back_populates="product")
    sales = relationship("Sale", back_populates="product")

    def __repr__(self):
        return (
            f"<Product(product_id={self.product_id}, "
            f"product_name='{self.product_name}')>"
        )

class inventory(Base):
    __tablename__ = 'inventory'

    inventory_id = Column(String(100), primary_key=True, index=True)
    product_id = Column(String, ForeignKey("products.product_id"))
    product_name = Column(String, nullable=False)
    batch_id = Column(String(100), nullable=False)
    current_quantity = Column(Integer, nullable=False)
    manufacturing_date = Column(Date, nullable=True)
    expiry_date = Column(Date, nullable=True)
    warehouse = Column(String(100), nullable=False)
    location = Column(String(100), nullable=False)
    temperature_compliant = Column(Boolean, nullable=True)
    last_checked = Column(DateTime, nullable=True)
    received_date = Column(Date, nullable=True)
    quarantine_status = Column(String(100), nullable=False)
    
    product = relationship("Product", back_populates="inventories")

    def __repr__(self):
        return f"<Inventory(id={self.inventory_id}, product_id={self.product_id})>"

class sale(Base):
    __tablename__ = 'sales'

    sale_id = Column(String, primary_key=True)
    invoice_number = Column(String)
    product_id = Column(String, ForeignKey("products.product_id"))
    product_name = Column(String)
    category = Column(String)
    sale_date = Column(Date)
    quantity = Column(Integer)
    unit_price = Column(Numeric(10, 2))
    total_value = Column(Numeric(15, 2))
    facility_id = Column(Integer)
    facility_name = Column(String)
    facility_type = Column(String)
    region = Column(String)
    governorate = Column(String)
    is_weekend = Column(Boolean)
    is_holiday = Column(Boolean)
    month = Column(Integer)
    year = Column(Integer)
    day_of_week = Column(String)
    cost_per_unit = Column(Numeric(10, 2))
    total_cost = Column(Numeric(15, 2))
    profit = Column(Numeric(15, 2))
    
    product = relationship("Product", back_populates="sales")

    def __repr__(self):
        return (
            f"<Sale(sale_id={self.sale_id}, "
            f"product_id={self.product_id}, "
            f"quantity={self.quantity})>"
        )