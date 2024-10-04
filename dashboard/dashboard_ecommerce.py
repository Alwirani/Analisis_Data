import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/Alwirani/Analisis_Data/main/dashboard/final_dataset.csv", 
                encoding='utf-8', 
                on_bad_lines='skip')

# Convert datetime columns
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Sidebar for filtering (if needed)
st.sidebar.header('Filters')
state_filter = st.sidebar.multiselect("Select States", options=df['customer_state_y'].unique(), default=df['customer_state_y'].unique())

# Filter dataset based on state
df_filtered = df[df['customer_state_y'].isin(state_filter)]

# 1. Summary Statistics: Total Orders and Total Revenue
total_orders = df_filtered['order_id'].nunique()
total_revenue = df_filtered['price'].sum()

st.title("E-Commerce Dashboard")
st.subheader("Summary Statistics")
st.write(f"Total Orders: {total_orders}")
st.write(f"Total Revenue: ${total_revenue:.2f}")

# 2. Daily Orders
daily_orders = df_filtered.groupby(df_filtered['order_purchase_timestamp'].dt.date)['order_id'].nunique().reset_index()
daily_orders.columns = ['Date', 'Total Orders']
fig_daily_orders = px.line(daily_orders, x='Date', y='Total Orders', title='Daily Orders')
st.plotly_chart(fig_daily_orders)

# 3. Daily Revenue vs Orders
daily_revenue = df_filtered.groupby(df_filtered['order_purchase_timestamp'].dt.date).agg({'price': 'sum', 'order_id': 'nunique'}).reset_index()
daily_revenue.columns = ['Date', 'Total Revenue', 'Total Orders']
fig_revenue_orders = go.Figure()
fig_revenue_orders.add_trace(go.Scatter(x=daily_revenue['Date'], y=daily_revenue['Total Revenue'], mode='lines', name='Total Revenue'))
fig_revenue_orders.add_trace(go.Bar(x=daily_revenue['Date'], y=daily_revenue['Total Orders'], name='Total Orders'))
fig_revenue_orders.update_layout(title='Daily Revenue vs Orders', xaxis_title='Date', yaxis_title='Count')
st.plotly_chart(fig_revenue_orders)

# 4. Product Performance (quantity and total price)
product_performance = df_filtered.groupby('product_category_name_english').agg({'order_item_id': 'count', 'price': 'sum'}).reset_index()
product_performance.columns = ['Product Category', 'Total Sold', 'Total Revenue']
fig_product_performance = px.bar(product_performance, x='Product Category', y=['Total Sold', 'Total Revenue'], barmode='group', title='Product Performance')
st.plotly_chart(fig_product_performance)

# 5. Customers by State
customers_by_state = df_filtered.groupby('customer_state_y')['customer_id'].nunique().reset_index()
customers_by_state.columns = ['State', 'Total Customers']
fig_customers_state = px.bar(customers_by_state, x='State', y='Total Customers', title='Customers by State')
st.plotly_chart(fig_customers_state)

# 6. RFM Analysis
now = pd.Timestamp.now()

# Recency
recency = df_filtered.groupby('customer_id')['order_purchase_timestamp'].max().reset_index()
recency['Recency'] = (now - recency['order_purchase_timestamp']).dt.days

# Frequency
frequency = df_filtered.groupby('customer_id')['order_id'].nunique().reset_index()
frequency.columns = ['customer_id', 'Frequency']

# Monetary
monetary = df_filtered.groupby('customer_id')['price'].sum().reset_index()
monetary.columns = ['customer_id', 'Monetary']

# Merge RFM metrics
rfm = recency.merge(frequency, on='customer_id').merge(monetary, on='customer_id')

# Plot RFM
fig_rfm = px.scatter(rfm, x='Frequency', y='Monetary', size='Recency', color='Recency', title='RFM Analysis')
st.plotly_chart(fig_rfm)

# 8. Geospatial Analysis
fig_geo = px.scatter_geo(df_filtered, lat='geolocation_lat', lon='geolocation_lng', hover_name='customer_city_y', title='Geospatial Analysis')
st.plotly_chart(fig_geo)

# 9. Clustering (without ML)
cluster_data = df_filtered.groupby('customer_state_y').agg({'price': 'sum', 'order_id': 'nunique'}).reset_index()
cluster_data.columns = ['State', 'Total Revenue', 'Total Orders']
fig_cluster = px.scatter(cluster_data, x='Total Orders', y='Total Revenue', color='State', title='Customer Clustering by State')
st.plotly_chart(fig_cluster)
