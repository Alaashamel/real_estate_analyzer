"""
Streamlit Dashboard for Real Estate Price Prediction
Features: Upload image + description, get price prediction, analytics, recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import tempfile
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Real Estate AI Analyzer",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
        color: white;
    }
    .price-text {
        font-size: 3rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 30px;
        border-radius: 30px;
    }
    .stButton > button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Load Resources (Cached)
# ============================================================================

@st.cache_resource
def load_predictor():
    """Load the trained price predictor"""
    from multimodal.price_predictor import PricePredictor
    predictor = PricePredictor()
    model_path = 'models/price_predictor.pkl'
    if os.path.exists(model_path):
        if predictor.load(model_path):
            st.success("✅ PricePredictor loaded successfully")
            return predictor
        else:
            st.warning("⚠️ Failed to load model from {}, using untrained predictor".format(model_path))
    else:
        st.warning("⚠️ No model file at {}. Train model first: python multimodal/price_predictor.py".format(model_path))
    return predictor  # Always return instance


@st.cache_resource
def load_text_analyzer():
    """Load text analyzer"""
    try:
        from nlp.text_features import PropertyTextAnalyzer
        return PropertyTextAnalyzer()
    except:
        return None


@st.cache_data
def load_dataset():
    """Load the processed dataset"""
    data_path = 'data/processed/final_dataset.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return df
    return None


# ============================================================================
# Helper Functions
# ============================================================================

def extract_amenities_from_text(text_analyzer, description):
    """Extract amenities from description"""
    if text_analyzer and description:
        return text_analyzer.extract_amenities(description)
    return {}


def create_price_distribution_chart(df):
    """Create price distribution chart"""
    fig = px.histogram(df, x='price', nbins=50, 
                       title="🏠 Property Price Distribution",
                       labels={'price': 'Price (EGP)', 'count': 'Number of Properties'},
                       color_discrete_sequence=['#667eea'])
    fig.update_layout(bargap=0.1, showlegend=False)
    return fig


def create_price_by_type_chart(df):
    """Create price by property type chart"""
    if 'property_type' in df.columns:
        fig = px.box(df, x='property_type', y='price', 
                     title="💰 Price Distribution by Property Type",
                     labels={'property_type': 'Property Type', 'price': 'Price (EGP)'},
                     color='property_type')
        return fig
    return None


def create_price_vs_area_chart(df):
    """Create price vs area scatter plot"""
    fig = px.scatter(df, x='area_sqm', y='price', 
                     title="📐 Price vs Area",
                     labels={'area_sqm': 'Area (sqm)', 'price': 'Price (EGP)'},
                     trendline="ols",
                     color_discrete_sequence=['#667eea'])
    return fig


def create_amenity_chart(df):
    """Create amenity frequency chart"""
    amenity_cols = [col for col in df.columns if col.startswith('has_')]
    if amenity_cols:
        amenity_stats = {}
        for col in amenity_cols:
            amenity_name = col.replace('has_', '')
            count = df[col].sum()
            amenity_stats[amenity_name] = count
        
        amenity_df = pd.DataFrame(list(amenity_stats.items()), columns=['Amenity', 'Count'])
        amenity_df = amenity_df.sort_values('Count', ascending=True)
        
        fig = px.bar(amenity_df, x='Count', y='Amenity', orientation='h',
                     title="🔧 Amenities Frequency",
                     color='Count', color_continuous_scale='Blues')
        return fig
    return None


# ============================================================================
# Main App
# ============================================================================

def main():
    st.markdown('<div class="main-header">🏠 Real Estate AI Analyzer</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🔮 Price Predictor", "📊 Market Analytics", "🎯 Recommendations", "📈 Model Info"])
    
    # Load resources
    predictor = load_predictor()
    text_analyzer = load_text_analyzer()
    df = load_dataset()
    
    if page == "🔮 Price Predictor":
        price_predictor_page(predictor, text_analyzer)
    
    elif page == "📊 Market Analytics":
        market_analytics_page(df)
    
    elif page == "🎯 Recommendations":
        recommendations_page(df, predictor)
    
    elif page == "📈 Model Info":
        model_info_page(predictor)


def price_predictor_page(predictor, text_analyzer):
    st.header("🔮 Predict Property Price")
    
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    📝 Enter property details below to get an AI-powered price prediction based on similar properties.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Property Description")
        description = st.text_area(
            "Enter property description",
            height=200,
            placeholder="Example: Luxury apartment in Cairo with 3 bedrooms, sea view, fully furnished, elevator, parking, 150 sqm..."
        )
        
        st.subheader("📊 Property Details")
        
        col_a, col_b = st.columns(2)
        with col_a:
            area = st.number_input("📐 Area (sqm)", min_value=20, max_value=1000, value=120)
            bedrooms = st.number_input("🛏️ Bedrooms", min_value=0, max_value=10, value=2)
        
        with col_b:
            bathrooms = st.number_input("🚽 Bathrooms", min_value=1, max_value=8, value=2)
            property_type = st.selectbox("🏠 Property Type", ["Apartment", "Villa", "Townhouse", "Studio", "Duplex"])
        
        furnishing = st.selectbox("🪑 Furnishing", ["Unfurnished", "Semi-furnished", "Furnished"])
    
    with col2:
        st.subheader("🖼️ Property Images")
        st.write("Upload images of the property (optional)")
        
        uploaded_files = st.file_uploader(
            "Upload images",
            type=['jpg', 'jpeg', 'png', 'webp'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            cols = st.columns(min(len(uploaded_files), 3))
            for idx, file in enumerate(uploaded_files[:6]):
                with cols[idx % 3]:
                    img = Image.open(file)
                    st.image(img, caption=f"Image {idx+1}", use_container_width=True)
    
    # Analyze description
    if description:
        st.subheader("🔍 Text Analysis")
        
        amenities = extract_amenities_from_text(text_analyzer, description)
        
        if amenities:
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("📍 Total Amenities", sum(amenities.values()))
            with col_b:
                st.metric("🛡️ Security", "✅" if amenities.get('security', 0) else "❌")
            with col_c:
                st.metric("🅿️ Parking", "✅" if amenities.get('parking', 0) else "❌")
            with col_d:
                st.metric("🏊 Pool", "✅" if amenities.get('pool', 0) else "❌")
            
            # Show all amenities
            st.write("**Detected Amenities:**")
            amenity_cols = st.columns(4)
            for idx, (amenity, present) in enumerate(amenities.items()):
                if idx < 12:
                    with amenity_cols[idx % 4]:
                        st.write(f"{'✅' if present else '❌'} {amenity.capitalize()}")
    
    # Predict button
    if st.button("💰 Predict Price", type="primary", use_container_width=True):
        if not description:
            st.warning("⚠️ Please enter a property description")
        elif predictor.model is None:
            st.error("❌ Model not loaded/trained. Please run: python multimodal/price_predictor.py")
        else:
            with st.spinner("🔍 Analyzing property..."):
                try:
                    # Use the predictor's predict method
                    price = predictor.predict(
                        description=description,
                        area_sqm=area,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        property_type=property_type,
                        furnishing=furnishing
                    )
                    
                    # Display prediction
                    st.markdown("---")
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown('<h2 style="color: white;">🏠 Predicted Price</h2>', unsafe_allow_html=True)
                    st.markdown(f'<div class="price-text">{price:,.0f} EGP</div>', unsafe_allow_html=True)
                    
                    # Price range
                    lower_bound = price * 0.85
                    upper_bound = price * 1.15
                    st.write(f"**Estimated Range:** {lower_bound:,.0f} - {upper_bound:,.0f} EGP")
                    
                    # Confidence meter
                    confidence = min(0.95, max(0.5, 1 - (price / 10000000)))
                    st.progress(confidence)
                    st.caption(f"Confidence: {confidence:.0%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")


def market_analytics_page(df):
    st.header("📊 Market Analytics")
    
    if df is None:
        st.warning("⚠️ No data available. Please run the scraper and training first.")
        st.info("Run: python scraper/data_pipeline.py to process data")
        return
    
    # Summary metrics
    st.subheader("📈 Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🏠 Total Properties", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("💰 Average Price", f"{df['price'].mean():,.0f} EGP")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📐 Avg Area", f"{df['area_sqm'].mean():.0f} sqm")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("🛏️ Avg Bedrooms", f"{df['bedrooms'].mean():.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts
    st.subheader("📊 Price Distribution")
    fig1 = create_price_distribution_chart(df)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Price by Type")
        fig2 = create_price_by_type_chart(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("📐 Price vs Area")
        fig3 = create_price_vs_area_chart(df)
        st.plotly_chart(fig3, use_container_width=True)
    
    # Amenities
    st.subheader("🔧 Amenities Analysis")
    fig4 = create_amenity_chart(df)
    if fig4:
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No amenity data available in the dataset")


def recommendations_page(df, predictor):
    st.header("🎯 Property Recommendations")
    
    if df is None:
        st.warning("⚠️ No data available.")
        return
    
    st.subheader("🔍 Find Similar Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_price = st.number_input("💰 Max Price (EGP)", min_value=0, value=5_000_000, step=500_000)
        min_area = st.number_input("📐 Min Area (sqm)", min_value=0, value=80, step=10)
    
    with col2:
        if 'property_type' in df.columns:
            property_type = st.selectbox("🏠 Property Type", ["All"] + list(df['property_type'].unique()))
        else:
            property_type = "All"
        
        min_bedrooms = st.slider("🛏️ Min Bedrooms", 0, 5, 1)
    
    # Filter
    filtered_df = df[df['price'] <= max_price]
    filtered_df = filtered_df[filtered_df['area_sqm'] >= min_area]
    filtered_df = filtered_df[filtered_df['bedrooms'] >= min_bedrooms]
    
    if property_type != "All" and 'property_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['property_type'] == property_type]
    
    st.write(f"📊 Found **{len(filtered_df)}** matching properties")
    
    # Sort by price per sqm (best value)
    if 'price_per_sqm' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('price_per_sqm')
    
    # Display top recommendations
    for idx, row in filtered_df.head(10).iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                title = row.get('title', 'Property')
                st.write(f"**{title[:80]}**")
                if 'location' in row and row.get('location'):
                    st.caption(f"📍 {row['location']}")
            
            with col2:
                st.write(f"💰 **{row['price']:,.0f} EGP**")
                if 'area_sqm' in row:
                    st.caption(f"📐 {row['area_sqm']} sqm")
            
            with col3:
                if 'property_type' in row:
                    st.write(f"🏠 {row['property_type']}")
                st.caption(f"🛏️ {row['bedrooms']} beds | 🚽 {row['bathrooms']} baths")
            
            st.divider()


def model_info_page(predictor):
    st.header("📈 Model Information")
    
    st.markdown("""
    ### 🧠 Multimodal AI Model Architecture
    
### 📊 Features Used

| Modality | Features | Dimension |
|----------|----------|-----------|
| Text | TF-IDF from description | 500 |
| Image | ResNet18 features | 512 |
| Tabular | Area, bedrooms, bathrooms, property type, furnishing | 7+ |

### 🎯 Performance Metrics

""")

# if predictor.model is not None:
#     st.success("✅ Model loaded successfully!")
#     st.info("Ready for price predictions")
# else:
#     st.warning("⚠️ No trained model loaded")
#     st.info("**Status:** Untrained predictor available (predictions will fail)")
#     st.code("python multimodal/price_predictor.py", language="bash")


# ============================================================================
# Run App
# ============================================================================

if __name__ == "__main__":
        main()
else:
    print("ℹ️  Run with: streamlit run app/streamlit_app.py")
    print("   Direct python execution disabled to prevent Streamlit context errors.")
    
    