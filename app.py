import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import datetime

# Page Config
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for Professional Look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Database Connection
# In a real production app, these would be in st.secrets or environment variables
DB_URL = "postgresql://postgres:password@localhost:5432/retail_db"

@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

def load_data(engine, region_filter, date_range):
    """
    Fetches data from PostgreSQL based on sidebar filters.
    """
    # Construct SQL query dynamically
    query = """
    SELECT
        s.order_date, s.total_revenue, s.profit, s.quantity,
        p.category, p.product_name,
        r.region_name
    FROM fact_sales s
    JOIN dim_products p ON s.product_id = p.product_id
    JOIN dim_customers c ON s.customer_id = c.customer_id
    JOIN dim_regions r ON c.region_id = r.region_id
    WHERE s.order_date BETWEEN %s AND %s
    """

    params = [date_range[0], date_range[1]]

    if region_filter and region_filter != "All":
        if isinstance(region_filter, list):
            # Handle multi-select
            placeholder = ', '.join(['%s'] * len(region_filter))
            query += f" AND r.region_name IN ({placeholder})"
            params.extend(region_filter)
        else:
            query += " AND r.region_name = %s"
            params.append(region_filter)

    return pd.read_sql(query, engine, params=params)

# --- SIDEBAR ---
st.sidebar.header("Dashboard Filters")

# Date Range Filter
today = datetime.date.today()
start_date = today - datetime.timedelta(days=730) # 2 years
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(start_date, today),
    help="Filter the data based on the order date"
)

# Region Filter
# To get unique regions, we could query the DB, but for this demo we use a list
# In production: regions = pd.read_sql("SELECT region_name FROM dim_regions", engine)['region_name'].tolist()
regions_list = ['North America', 'Europe', 'Asia-Pacific', 'Latin America', 'Middle East']
selected_regions = st.sidebar.multiselect(
    "Select Regions",
    options=regions_list,
    default=regions_list
)

# --- MAIN CONTENT ---
st.title("🚀 Retail Sales Performance Dashboard")
st.markdown("Real-time business intelligence for retail operations.")

try:
    engine = get_db_engine()

    # Apply filters to data loading
    # If no regions selected, treat as 'All'
    region_filter = selected_regions if selected_regions else "All"

    if len(date_range) == 2:
        df = load_data(engine, region_filter, date_range)

        if df.empty:
            st.warning("No data found for the selected filters.")
        else:
            # 1. KPI METRICS
            total_rev = df['total_revenue'].sum()
            total_profit = df['profit'].sum()
            avg_order_val = df['total_revenue'].mean()
            margin_pct = (total_profit / total_rev) * 100 if total_rev != 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Revenue", f"${total_rev:,.2f}")
            with col2:
                st.metric("Total Profit", f"${total_profit:,.2f}")
            with col3:
                st.metric("Avg Order Value", f"${avg_order_val:,.2f}")
            with col4:
                st.metric("Profit Margin", f"{margin_pct:.2f}%")

            st.markdown("---")

            # 2. VISUALIZATIONS
            row1_col1, row1_col2 = st.columns(2)

            with row1_col1:
                st.subheader("Revenue Trend")
                # Group by date for time series
                trend_df = df.groupby('order_date')['total_revenue'].sum().reset_index()
                fig_trend = px.line(
                    trend_df, x='order_date', y='total_revenue',
                    title="Daily Revenue Over Time",
                    labels={'total_revenue': 'Revenue ($)', 'order_date': 'Date'},
                    template="plotly_white"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

            with row1_col2:
                st.subheader("Revenue by Category")
                cat_df = df.groupby('category')['total_revenue'].sum().reset_index()
                fig_cat = px.bar(
                    cat_df, x='category', y='total_revenue',
                    color='category',
                    title="Total Revenue by Product Category",
                    labels={'total_revenue': 'Revenue ($)', 'category': 'Category'},
                    template="plotly_white"
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            st.markdown("---")

            row2_col1, row2_col2 = st.columns([1, 2])

            with row2_col1:
                st.subheader("Regional Distribution")
                reg_df = df.groupby('region_name')['total_revenue'].sum().reset_index()
                fig_reg = px.pie(
                    reg_df, values='total_revenue', names='region_name',
                    title="Revenue Share by Region",
                    hole=0.4,
                    template="plotly_white"
                )
                st.plotly_chart(fig_reg, use_container_width=True)

            with row2_col2:
                st.subheader("Recent Transactions")
                st.dataframe(df[['order_date', 'product_name', 'category', 'total_revenue', 'profit']].sort_values('order_date', ascending=False).head(10), use_container_width=True)

except Exception as e:
    st.error(f"Database Connection Error: {e}")
    st.info("Please ensure PostgreSQL is running and the connection string in app.py is correct.")
