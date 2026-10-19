-- ---------------------------------------------------------------------------
-- Database Schema Design for Retail Sales BI Dashboard
-- ---------------------------------------------------------------------------

-- Drop tables if they exist (for idempotency)
DROP TABLE IF EXISTS fact_sales;
DROP TABLE IF EXISTS dim_products;
DROP TABLE IF EXISTS dim_customers;
DROP TABLE IF EXISTS dim_regions;

-- 1. Dimension Table: Regions
CREATE TABLE dim_regions (
    region_id SERIAL PRIMARY KEY,
    region_name VARCHAR(50) UNIQUE NOT NULL
);

-- 2. Dimension Table: Customers
CREATE TABLE dim_customers (
    customer_id INT PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(100),
    region_id INT REFERENCES dim_regions(region_id),
    join_date DATE
);

-- 3. Dimension Table: Products
CREATE TABLE dim_products (
    product_id INT PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    unit_cost DECIMAL(10, 2) NOT NULL
);

-- 4. Fact Table: Sales
CREATE TABLE fact_sales (
    sale_id SERIAL PRIMARY KEY,
    order_id INT NOT NULL,
    order_date DATE NOT NULL,
    customer_id INT REFERENCES dim_customers(customer_id),
    product_id INT REFERENCES dim_products(product_id),
    quantity INT NOT NULL CHECK (quantity > 0),
    total_revenue DECIMAL(12, 2) NOT NULL,
    total_cost DECIMAL(12, 2) NOT NULL,
    profit DECIMAL(12, 2) NOT NULL
);

-- ---------------------------------------------------------------------------
-- Indexing Strategy for BI Performance
-- ---------------------------------------------------------------------------

-- Index on order_date for time-series analysis (Monthly Revenue)
CREATE INDEX idx_sales_date ON fact_sales(order_date);

-- Index on product_id and customer_id for join optimization
CREATE INDEX idx_sales_product ON fact_sales(product_id);
CREATE INDEX idx_sales_customer ON fact_sales(customer_id);

-- Index on category for filtering
CREATE INDEX idx_product_category ON dim_products(category);
