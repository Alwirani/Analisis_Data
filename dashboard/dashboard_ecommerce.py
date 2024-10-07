import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------- #
#       Load Data       #
# --------------------- #

@st.cache_data
def load_data(url):
    """Load and preprocess the dataset."""
    df = pd.read_csv(url, encoding='utf-8', on_bad_lines='skip')
    
    # Convert datetime columns
    datetime_columns = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date', 'review_creation_date', 
        'review_answer_timestamp'
    ]
    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert numerical columns to appropriate types
    df['product_photos_qty'] = df['product_photos_qty'].astype('Int64')
    df['payment_sequential'] = df['payment_sequential'].astype('Int64')
    df['payment_installments'] = df['payment_installments'].astype('Int64')
    df['review_score'] = df['review_score'].astype('Int64')
    
    return df

# Load the dataset
DATA_URL = "https://raw.githubusercontent.com/Alwirani/Analisis_Data/main/dashboard/final_dataset.csv"
final_dataset = load_data(DATA_URL)

# --------------------- #
#      Sidebar Filters  #
# --------------------- #

st.sidebar.header('Filters')

# State Filter
state_filter = st.sidebar.multiselect(
    "Select States",
    options=final_dataset['customer_state_y'].unique(),
    default=final_dataset['customer_state_y'].unique()
)

# Apply State Filter
df_filtered = final_dataset[final_dataset['customer_state_y'].isin(state_filter)]

# --------------------- #
#    Dashboard Title    #
# --------------------- #

st.title("E-Commerce Dashboard")
st.markdown("### Business Intelligence PT Toktok")

# --------------------- #
#   Summary Statistics  #
# --------------------- #

st.header("Summary Statistics")

total_orders = df_filtered['order_id'].nunique()
total_revenue = df_filtered['price'].sum()

# Display Summary Metrics in Columns
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Orders", f"{total_orders}")
with col2:
    st.metric("Total Revenue ($)", f"${total_revenue:,.2f}")

# --------------------- #
#       Daily Metrics   #
# --------------------- #

st.header("Daily Metrics")

# Daily Orders
daily_orders = df_filtered.groupby(df_filtered['order_purchase_timestamp'].dt.date)['order_id'].nunique().reset_index()
daily_orders.columns = ['Date', 'Total Orders']

fig_daily_orders = px.line(
    daily_orders,
    x='Date',
    y='Total Orders',
    title='Daily Orders',
    labels={'Date': 'Date', 'Total Orders': 'Number of Orders'},
    template='plotly_white'
)
st.plotly_chart(fig_daily_orders, use_container_width=True)

# Daily Revenue vs Orders
daily_revenue = df_filtered.groupby(df_filtered['order_purchase_timestamp'].dt.date).agg({
    'price': 'sum',
    'order_id': 'nunique'
}).reset_index()
daily_revenue.columns = ['Date', 'Total Revenue', 'Total Orders']

fig_revenue_orders = go.Figure()

# Add Total Revenue as Line
fig_revenue_orders.add_trace(go.Scatter(
    x=daily_revenue['Date'],
    y=daily_revenue['Total Revenue'],
    mode='lines',
    name='Total Revenue ($)',
    yaxis='y1'
))

# Add Total Orders as Bar
fig_revenue_orders.add_trace(go.Bar(
    x=daily_revenue['Date'],
    y=daily_revenue['Total Orders'],
    name='Total Orders',
    yaxis='y2',
    marker_color='rgba(55, 83, 109, 0.7)'
))

# Update Layout for Dual Y-Axis
fig_revenue_orders.update_layout(
    title='Daily Revenue vs Orders',
    xaxis=dict(title='Date'),
    yaxis=dict(
        title='Total Revenue ($)',
        titlefont=dict(color='rgba(55, 83, 109, 1)'),
        tickfont=dict(color='rgba(55, 83, 109, 1)')
    ),
    yaxis2=dict(
        title='Total Orders',
        titlefont=dict(color='rgba(26, 118, 255, 1)'),
        tickfont=dict(color='rgba(26, 118, 255, 1)'),
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.01, y=1.05, orientation='h'),
    template='plotly_white'
)

st.plotly_chart(fig_revenue_orders, use_container_width=True)

# --------------------- #
#    General Overview   #
# --------------------- #

st.header("General Overview")

# Section: Top 10 Product Categories
st.subheader("Top 10 Product Categories")
top10_product_categories = df_filtered['product_category_name'].value_counts().head(10).reset_index()
top10_product_categories.columns = ['Product Category', 'Frequency']

fig_top10_products = px.bar(
    top10_product_categories,
    x='Product Category',
    y='Frequency',
    title='Top 10 Product Categories',
    labels={'Product Category': 'Product Category', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_top10_products, use_container_width=True)

# Section: Order Status Distribution
st.subheader("Order Status Distribution")
order_status_counts = df_filtered['order_status_x'].value_counts().reset_index()
order_status_counts.columns = ['Order Status', 'Frequency']

fig_order_status = px.bar(
    order_status_counts,
    x='Order Status',
    y='Frequency',
    title='Order Status Distribution',
    labels={'Order Status': 'Order Status', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_order_status, use_container_width=True)

# Section: Payment Type Distribution
st.subheader("Payment Type Distribution")
payment_type_counts = df_filtered['payment_type'].value_counts().reset_index()
payment_type_counts.columns = ['Payment Type', 'Frequency']

fig_payment_type = px.bar(
    payment_type_counts,
    x='Payment Type',
    y='Frequency',
    title='Payment Type Distribution',
    labels={'Payment Type': 'Payment Type', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_payment_type, use_container_width=True)

# Section: Review Score Distribution
st.subheader("Review Score Distribution")
review_score_counts = df_filtered['review_score'].value_counts().reset_index()
review_score_counts.columns = ['Review Score', 'Frequency']

fig_review_score = px.bar(
    review_score_counts,
    x='Review Score',
    y='Frequency',
    title='Review Score Distribution',
    labels={'Review Score': 'Review Score', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_review_score, use_container_width=True)

# Section: Top 10 Seller Cities
st.subheader("Top 10 Seller Cities")
top10_seller_cities = df_filtered['seller_city'].value_counts().head(10).reset_index()
top10_seller_cities.columns = ['Seller City', 'Frequency']

fig_seller_cities = px.bar(
    top10_seller_cities,
    x='Seller City',
    y='Frequency',
    title='Top 10 Seller Cities',
    labels={'Seller City': 'Seller City', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_seller_cities, use_container_width=True)

# Section: Top 10 Customer Cities
st.subheader("Top 10 Customer Cities")
top10_customer_cities = df_filtered['customer_city_x'].value_counts().head(10).reset_index()
top10_customer_cities.columns = ['Customer City', 'Frequency']

fig_customer_cities = px.bar(
    top10_customer_cities,
    x='Customer City',
    y='Frequency',
    title='Top 10 Customer Cities',
    labels={'Customer City': 'Customer City', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_customer_cities, use_container_width=True)

# Section: Top 10 Product Categories (English)
st.subheader("Top 10 Product Categories (English)")
top10_product_categories_en = df_filtered['product_category_name_english'].value_counts().head(10).reset_index()
top10_product_categories_en.columns = ['Product Category (EN)', 'Frequency']

fig_top10_products_en = px.bar(
    top10_product_categories_en,
    x='Product Category (EN)',
    y='Frequency',
    title='Top 10 Product Categories (English)',
    labels={'Product Category (EN)': 'Product Category (EN)', 'Frequency': 'Frequency'},
    template='plotly_white'
)
st.plotly_chart(fig_top10_products_en, use_container_width=True)

# Section: Purchase Time Distribution
st.subheader("Purchase Time Distribution")
df_filtered['order_purchase_hour'] = df_filtered['order_purchase_timestamp'].dt.hour

fig_purchase_time = px.histogram(
    df_filtered,
    x='order_purchase_hour',
    nbins=24,
    title='Purchase Time Distribution',
    labels={'order_purchase_hour': 'Hour of Purchase', 'count': 'Number of Orders'},
    template='plotly_white'
)
st.plotly_chart(fig_purchase_time, use_container_width=True)

# Section: Shipping Duration Distribution
st.subheader("Shipping Duration Distribution")
df_filtered['shipping_time'] = (df_filtered['order_delivered_customer_date'] - df_filtered['order_delivered_carrier_date']).dt.days

fig_shipping_duration = px.histogram(
    df_filtered,
    x='shipping_time',
    nbins=30,
    title='Shipping Duration Distribution',
    labels={'shipping_time': 'Shipping Duration (Days)', 'count': 'Number of Orders'},
    template='plotly_white'
)
st.plotly_chart(fig_shipping_duration, use_container_width=True)

# --------------------- #
#      RFM Analysis     #
# --------------------- #

st.header("RFM Analysis")

# Calculate RFM metrics
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
fig_rfm = px.scatter(
    rfm,
    x='Frequency',
    y='Monetary',
    size='Recency',
    color='Recency',
    title='RFM Analysis',
    labels={'Frequency': 'Frequency', 'Monetary': 'Monetary Value', 'Recency': 'Recency (Days)'},
    template='plotly_white',
    size_max=15
)
st.plotly_chart(fig_rfm, use_container_width=True)

# --------------------- #
#  Geospatial Analysis  #
# --------------------- #

st.header("Geospatial Analysis")

fig_geo = px.scatter_geo(
    df_filtered,
    lat='geolocation_lat',
    lon='geolocation_lng',
    hover_name='customer_city_y',
    title='Geospatial Analysis of Customers',
    projection='natural earth',
    template='plotly_white'
)

fig_geo.update_geos(
    scope='south america',  # Adjust based on your data's geographic location
    showcountries=True,
    countrycolor="Black",
    showland=True,
    landcolor="LightGray",
    showocean=True,
    oceancolor="LightBlue",
    fitbounds="locations"
)

st.plotly_chart(fig_geo, use_container_width=True)

# --------------------- #
#     Clustering        #
# --------------------- #

st.header("Customer Clustering by State")

# Aggregate data for clustering
cluster_data = df_filtered.groupby('customer_state_y').agg({
    'price': 'sum',
    'order_id': 'nunique'
}).reset_index()
cluster_data.columns = ['State', 'Total Revenue ($)', 'Total Orders']

fig_cluster = px.scatter(
    cluster_data,
    x='Total Orders',
    y='Total Revenue ($)',
    color='State',
    size='Total Revenue ($)',
    hover_name='State',
    title='Customer Clustering by State',
    labels={'Total Orders': 'Total Orders', 'Total Revenue ($)': 'Total Revenue ($)'},
    template='plotly_white'
)
st.plotly_chart(fig_cluster, use_container_width=True)

# --------------------- #
#  Advanced Analysis    #
# --------------------- #

st.header("Advanced Analysis")

# Ensure datetime conversion for advanced analysis
df_filtered['order_delivered_customer_date'] = pd.to_datetime(df_filtered['order_delivered_customer_date'], errors='coerce')
df_filtered['order_estimated_delivery_date'] = pd.to_datetime(df_filtered['order_estimated_delivery_date'], errors='coerce')

# Delivery Delay Analysis
df_filtered['delivery_delay'] = (df_filtered['order_delivered_customer_date'] - df_filtered['order_estimated_delivery_date']).dt.days

# Define Delivery Categories
conditions = [
    (df_filtered['delivery_delay'] <= 0),
    (df_filtered['delivery_delay'] > 0) & (df_filtered['delivery_delay'] <= 2),
    (df_filtered['delivery_delay'] > 2)
]
choices = ['On Time', 'Delayed 1-2 Days', 'Delayed More Than 2 Days']
df_filtered['delivery_category'] = pd.cut(
    df_filtered['delivery_delay'], 
    bins=[-float('inf'), 0, 2, float('inf')], 
    labels=choices
)

# Average Review Scores by Delivery Category
average_review_scores = df_filtered.groupby('delivery_category')['review_score'].mean().reset_index()

fig_delivery_delay = px.bar(
    average_review_scores,
    x='delivery_category',
    y='review_score',
    title='Average Review Score by Delivery Delay Category',
    labels={'review_score': 'Average Review Score', 'delivery_category': 'Delivery Delay Category'},
    template='plotly_white'
)
st.plotly_chart(fig_delivery_delay, use_container_width=True)

# Review Score Differences by Order Status and Payment Type
df_filtered_non_null_reviews = df_filtered[df_filtered['review_score'].notnull()]
review_summary = df_filtered_non_null_reviews.groupby(['order_status_x', 'payment_type'])['review_score'].mean().reset_index()

fig_review_summary = px.bar(
    review_summary,
    x='payment_type',
    y='review_score',
    color='order_status_x',
    title='Review Score Differences by Order Status and Payment Type',
    labels={'payment_type': 'Payment Type', 'review_score': 'Average Review Score', 'order_status_x': 'Order Status'},
    barmode='group',
    template='plotly_white'
)
st.plotly_chart(fig_review_summary, use_container_width=True)