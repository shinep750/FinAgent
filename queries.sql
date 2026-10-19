-- ---------------------------------------------------------------------------
-- Business Intelligence Analytics Queries
-- ---------------------------------------------------------------------------

-- 1. Monthly Revenue and Growth
-- Calculates total revenue per month
SELECT
    DATE_TRUNC('month', order_date) AS sales_month,
    SUM(total_revenue) AS monthly_revenue,
    COUNT(DISTINCT order_id) AS total_orders
FROM fact_sales
GROUP BY 1
ORDER BY 1;

-- 2. Profit Margins by Product Category
-- Margin = (Revenue - Cost) / Revenue
SELECT
    p.category,
    SUM(s.total_revenue) AS total_revenue,
    SUM(s.profit) AS total_profit,
    ROUND((SUM(s.profit) / NULLIF(SUM(s.total_revenue), 0)) * 100, 2) AS profit_margin_pct
FROM fact_sales s
JOIN dim_products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY profit_margin_pct DESC;

-- 3. Top 10 Highest Selling Products
SELECT
    p.product_name,
    p.category,
    SUM(s.quantity) AS total_units_sold,
    SUM(s.total_revenue) AS total_revenue
FROM fact_sales s
JOIN dim_products p ON s.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. Regional Performance Analysis
SELECT
    r.region_name,
    SUM(s.total_revenue) AS total_revenue,
    SUM(s.profit) AS total_profit,
    COUNT(DISTINCT s.customer_id) AS unique_customers
FROM fact_sales s
JOIN dim_customers c ON s.customer_id = c.customer_id
JOIN dim_regions r ON c.region_id = r.region_id
GROUP BY r.region_name
ORDER BY total_revenue DESC;
