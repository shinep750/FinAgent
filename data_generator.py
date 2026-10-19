import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_retail_data(num_orders=1000, num_customers=200, num_products=50):
    """
    Generates a synthetic retail sales dataset using standard libraries.
    Returns a tuple of DataFrames: (customers, products, orders)
    """

    # --- Mock Data for generation ---
    first_names = ['James', 'Mary', 'Robert', 'Patricia', 'John', 'Jennifer', 'Michael', 'Linda', 'William', 'Elizabeth']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
    domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com']
    regions = ['North America', 'Europe', 'Asia-Pacific', 'Latin America', 'Middle East']

    # 1. Generate Products
    categories = {
        'Electronics': {'price_range': (50, 1000), 'cost_range': (30, 700)},
        'Clothing': {'price_range': (10, 100), 'cost_range': (5, 40)},
        'Home & Kitchen': {'price_range': (15, 300), 'cost_range': (10, 150)},
        'Beauty': {'price_range': (5, 150), 'cost_range': (2, 60)},
        'Books': {'price_range': (10, 50), 'cost_range': (4, 20)}
    }

    product_list = []
    for i in range(1, num_products + 1):
        cat = random.choice(list(categories.keys()))
        price = round(random.uniform(*categories[cat]['price_range']), 2)
        cost = round(random.uniform(*categories[cat]['cost_range']), 2)
        product_list.append({
            'product_id': i,
            'product_name': f"{cat} Product {i}",
            'category': cat,
            'unit_price': price,
            'unit_cost': cost
        })
    df_products = pd.DataFrame(product_list)

    # 2. Generate Customers
    customer_list = []
    for i in range(1, num_customers + 1):
        fn = random.choice(first_names)
        ln = random.choice(last_names)
        email = f"{fn.lower()}.{ln.lower()}{i}@{random.choice(domains)}"

        # Random join date within last 3 years
        days_ago = random.randint(0, 3 * 365)
        join_date = datetime.now() - timedelta(days=days_ago)

        customer_list.append({
            'customer_id': i,
            'customer_name': f"{fn} {ln}",
            'email': email,
            'region': random.choice(regions),
            'join_date': join_date.date()
        })
    df_customers = pd.DataFrame(customer_list)

    # 3. Generate Orders
    orders_list = []
    start_date = datetime(2023, 1, 1)

    for i in range(1, num_orders + 1):
        order_date = start_date + timedelta(days=random.randint(0, 600))
        customer_id = random.randint(1, num_customers)

        # Each order can have 1 to 5 items
        num_items = random.randint(1, 5)
        for _ in range(num_items):
            product = df_products.sample(1).iloc[0]
            qty = random.randint(1, 3)

            orders_list.append({
                'order_id': i,
                'order_date': order_date,
                'customer_id': customer_id,
                'product_id': product['product_id'],
                'quantity': qty,
                'unit_price': product['unit_price'],
                'unit_cost': product['unit_cost']
            })

    df_orders = pd.DataFrame(orders_list)

    # Data Cleaning & Transformation
    df_orders['total_revenue'] = df_orders['quantity'] * df_orders['unit_price']
    df_orders['total_cost'] = df_orders['quantity'] * df_orders['unit_cost']
    df_orders['profit'] = df_orders['total_revenue'] - df_orders['total_cost']

    # Simulate some missing values for cleaning demonstration
    mask = np.random.choice([True, False], size=len(df_customers), p=[0.01, 0.99])
    df_customers.loc[mask, 'email'] = np.nan

    return df_customers, df_products, df_orders

def clean_data(df_customers, df_products, df_orders):
    df_customers['email'] = df_customers['email'].fillna('unknown@example.com')
    df_orders['order_date'] = pd.to_datetime(df_orders['order_date'])
    return df_customers, df_products, df_orders

if __name__ == "__main__":
    customers, products, orders = generate_retail_data()
    customers, products, orders = clean_data(customers, products, orders)

    customers.to_csv('customers.csv', index=False)
    products.to_csv('products.csv', index=False)
    orders.to_csv('orders.csv', index=False)

    print(f"Successfully generated synthetic data:")
    print(f"- Customers: {len(customers)}")
    print(f"- Products: {len(products)}")
    print(f"- Order Items: {len(orders)}")
    print("\nCSV files saved: customers.csv, products.csv, orders.csv")
