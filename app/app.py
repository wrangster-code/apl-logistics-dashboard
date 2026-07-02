import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# --- 1. PAGE CONFIGURATION ---
# This must be the very first Streamlit command. It sets the browser tab title and expands the layout.
st.set_page_config(
    page_title="APL Logistics | Delivery Intelligence",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. DATA LOADING (WITH CACHING) ---
# @st.cache_data is a decorator. It tells Streamlit to remember the output of this function.
@st.cache_data
def load_data():
    # Note: Because we run the app from the root folder, the path is relative to the root, not the app folder.
    df = pd.read_csv('data/processed/cleaned_apl_logistics.csv')
    return df

# --- ADD THIS RIGHT AFTER YOUR load_data() FUNCTION ---

@st.cache_resource
def load_ml_assets():
    model = joblib.load('models/rf_delivery_model.joblib')
    encoders = joblib.load('models/label_encoders.joblib')
    return model, encoders

try:
    rf_model, label_encoders = load_ml_assets()
except FileNotFoundError:
    st.error("ML Model or Encoders not found. Ensure Phase 11 completed successfully.")
    st.stop()

# Load the dataframe
try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'cleaned_apl_logistics.csv' is in the data/processed/ folder.")
    st.stop()

# --- 3. SIDEBAR FILTERS ---
st.sidebar.title("🚢 APL Logistics")
st.sidebar.markdown("### Operational Filters")

# We extract unique values from our dataset to populate the dropdown menus
selected_mode = st.sidebar.multiselect("Shipping Mode", options=df['shipping_mode'].unique(), default=df['shipping_mode'].unique())
selected_market = st.sidebar.multiselect("Global Market", options=df['market'].unique(), default=df['market'].unique())
selected_segment = st.sidebar.multiselect("Customer Segment", options=df['customer_segment'].unique(), default=df['customer_segment'].unique())

# --- 4. APPLY FILTERS TO DATA ---
# This creates a "filtered" version of our dataframe based on what the user selects in the sidebar
filtered_df = df[
    (df['shipping_mode'].isin(selected_mode)) &
    (df['market'].isin(selected_market)) &
    (df['customer_segment'].isin(selected_segment))
]

# --- 5. MAIN DASHBOARD AREA ---
st.title("🚢 Global Supply Chain & Delivery Diagnostics")
st.markdown("Monitor delivery performance, delay risks, and logistics efficiency across all operational regions.")

# --- 6. KPI METRICS (SCORECARDS) ---
st.markdown("### 📊 Delivery Performance Overview")

# Calculate KPIs mathematically based on the dynamically filtered dataframe
total_orders = filtered_df.shape[0]
delayed_orders = filtered_df[filtered_df['delivery_classification'] == 'Delayed'].shape[0]

# Prevent division by zero if filters result in 0 orders
on_time_rate = ((total_orders - delayed_orders) / total_orders * 100) if total_orders > 0 else 0
avg_delay = filtered_df['delay_gap'].mean() if total_orders > 0 else 0

# Create 4 responsive columns for our scorecards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Orders", value=f"{total_orders:,}")
with col2:
    st.metric(label="On-Time Delivery Rate", value=f"{on_time_rate:.2f}%")
with col3:
    st.metric(label="Average Delay (Days)", value=f"{avg_delay:.2f}")
with col4:
    st.metric(label="Delayed Shipments", value=f"{delayed_orders:,}")

st.divider() # Renders a clean horizontal rule

# --- 7. CHARTS & VISUALIZATIONS ---
st.markdown("### 📈 Delay Risk Analysis & Shipping Comparison")

# Create 2 columns for a side-by-side chart layout
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # 1. Delivery Classification Distribution
    status_counts = filtered_df['delivery_classification'].value_counts().reset_index()
    status_counts.columns = ['Delivery Status', 'Count']
    
    # We use the exact Plotly code from our EDA phase
    fig1 = px.pie(status_counts, 
                  values='Count', 
                  names='Delivery Status',
                  title='Delivery Status Distribution',
                  hole=0.4,
                  color='Delivery Status',
                  color_discrete_map={'Delayed': '#EF553B', 'On-time': '#00CC96', 'Early': '#636EFA'})
    
    # Render the chart in Streamlit, forcing it to span the full width of its column container
    st.plotly_chart(fig1, use_container_width=True)

with chart_col2:
    # 2. Average Delay by Shipping Mode
    mode_delay = filtered_df.groupby('shipping_mode')['delay_gap'].mean().reset_index()
    mode_delay = mode_delay.sort_values(by='delay_gap', ascending=False)
    
    fig2 = px.bar(mode_delay, 
                  x='shipping_mode', 
                  y='delay_gap',
                  title='Average Delay Gap by Shipping Mode',
                  labels={'shipping_mode': '', 'delay_gap': 'Avg Delay (Days)'},
                  color='delay_gap', 
                  color_continuous_scale='Reds')
    
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- 8. REGIONAL & DEMOGRAPHIC ANALYSIS ---
st.markdown("### 🌍 Regional Risk & Customer Segment Impact")

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    # 3. Average Delay by Market
    market_delay = filtered_df.groupby('market')['delay_gap'].mean().reset_index()
    market_delay = market_delay.sort_values(by='delay_gap', ascending=False)
    
    fig3 = px.bar(market_delay, 
                  x='market', 
                  y='delay_gap',
                  title='Average Delivery Delay by Market',
                  labels={'market': '', 'delay_gap': 'Avg Delay (Days)'},
                  color='delay_gap',
                  color_continuous_scale='Viridis')
    
    st.plotly_chart(fig3, use_container_width=True)

with chart_col4:
    # 4. Average Delay by Customer Segment
    segment_delay = filtered_df.groupby('customer_segment')['delay_gap'].mean().reset_index()
    segment_delay = segment_delay.sort_values(by='delay_gap', ascending=False)
    
    fig4 = px.bar(segment_delay, 
                  x='customer_segment', 
                  y='delay_gap',
                  title='Average Delay by Customer Segment',
                  labels={'customer_segment': '', 'delay_gap': 'Avg Delay (Days)'},
                  color='customer_segment',
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    
    st.plotly_chart(fig4, use_container_width=True)


st.divider()

# --- 9. MACHINE LEARNING PREDICTION MODULE ---
st.markdown("### 🤖 Predictive Intelligence: Delay Risk Forecaster")
st.markdown("Enter the parameters for a new shipment to predict the likelihood of a delivery delay using our Random Forest model.")

# Create a clean form for user inputs
with st.form("prediction_form"):
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    
    with pred_col1:
        # We use the unique values from our dataset to populate the dropdowns
        p_mode = st.selectbox("Shipping Mode", options=df['shipping_mode'].unique())
        p_segment = st.selectbox("Customer Segment", options=df['customer_segment'].unique())
        
    with pred_col2:
        p_market = st.selectbox("Global Market", options=df['market'].unique())
        p_region = st.selectbox("Order Region", options=df['order_region'].unique())
        
    with pred_col3:
        p_category = st.selectbox("Product Category", options=df['category_name'].unique())
        
    # Every form needs a submit button
    submit_prediction = st.form_submit_button("Predict Delivery Risk")

# --- 10. PROCESS THE PREDICTION ---
if submit_prediction:
    # 1. Capture the user's inputs into a dictionary matching our feature columns
    input_data = {
        'shipping_mode': p_mode,
        'customer_segment': p_segment,
        'market': p_market,
        'order_region': p_region,
        'category_name': p_category
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # 3. Encode the text inputs into numbers using our saved label encoders!
    try:
        for col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])
            
        # 4. Make the prediction
        prediction = rf_model.predict(input_df)[0]
        
        # 5. Display the result beautifully
        if prediction == 1:
            st.error("🚨 **High Risk of Delay!** Based on historical data, this shipment configuration is likely to miss its target date. Consider upgrading the shipping mode or notifying the customer.")
        else:
            st.success("✅ **On-Time Prediction!** This shipment is projected to arrive within the scheduled timeline.")
            
    except Exception as e:
        st.warning(f"Prediction Error: Ensure the selected combination exists in historical data. Details: {e}")