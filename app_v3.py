import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from dateutil.parser import parse

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RainCast | Clean Rain Forecast",
    page_icon="🌧️",
    layout="wide"
)

# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
.main-title { font-size: 2.5rem; font-weight: 800; color: #0f172a; }
.section-title { font-size: 1.4rem; font-weight: 700; color: #1e293b; margin: 1rem 0 0.5rem 0; }
.metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 1rem; border-radius: 12px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = "best_raincast_model.pkl"
SCALER_PATH = "raincast_scaler.pkl"
FEATURES_PATH = "raincast_feature_columns.pkl"

@st.cache_resource
def load_resources():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(FEATURES_PATH)

def check_files():
    for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            st.error(f"❌ Missing: {path}")
            st.stop()

# =========================================================
# LOAD
# =========================================================
check_files()
model, scaler, feature_columns = load_resources()
st.success(f"✅ Loaded: {len(feature_columns)} features")

# =========================================================
# UI
# =========================================================
st.markdown('<h1 class="main-title">🌧️ RainCast Predictor</h1>', unsafe_allow_html=True)
st.markdown("**Enter today's date & weather → Get tomorrow's rain forecast**")

# Metrics
col1, col2 = st.columns(2)
with col1: st.metric("Features", len(feature_columns))
with col2: st.metric("Model", "Ready")

st.divider()

# Form
with st.form("weather_form"):
    st.markdown("### 📅 Date")
    selected_date = st.date_input("Date (dd/mm/yy)", value=datetime.now().date())
    
    # Extract Y/M/D
    date_obj = pd.to_datetime(selected_date)
    year, month, day = date_obj.year, date_obj.month, date_obj.day
    
    st.markdown("### 🌡️ Weather Inputs (10 fields only)")
    
    # Organized sections
    st.markdown("**Temperature**")
    col1, col2 = st.columns(2)
    with col1: min_temp = st.number_input("Min Temp (°C)", value=20.0, step=0.1)
    with col2: max_temp = st.number_input("Max Temp (°C)", value=32.0, step=0.1)
    
    st.markdown("**Humidity**")
    col1, col2 = st.columns(2)
    with col1: humidity9am = st.number_input("Humidity 9AM (%)", 0.0, 100.0, 70.0)
    with col2: humidity3pm = st.number_input("Humidity 3PM (%)", 0.0, 100.0, 50.0)
    
    st.markdown("**Pressure**")
    col1, col2 = st.columns(2)
    with col1: pressure9am = st.number_input("Pressure 9AM (hPa)", 900.0, 1100.0, 1010.0, 0.1)
    with col2: pressure3pm = st.number_input("Pressure 3PM (hPa)", 900.0, 1100.0, 1005.0, 0.1)
    
    st.markdown("**Wind**")
    col1, col2 = st.columns(2)
    with col1: wind9am = st.number_input("Wind Speed 9AM (km/h)", 0.0, 100.0, 13.0)
    with col2: wind3pm = st.number_input("Wind Speed 3PM (km/h)", 0.0, 100.0, 15.0)
    
    rain_today = st.selectbox("Rain Today?", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    location_encoded = st.number_input("Location (0-50 encoded)", 0, 50, 0)
    
    predict = st.form_submit_button("🔮 Predict Tomorrow", use_container_width=True)

# =========================================================
# PREDICT
# =========================================================
if predict:
    try:
        # Build input matching model
        input_data = {
            feature_columns[0]: location_encoded,  # Location
            feature_columns[1]: min_temp,
            feature_columns[2]: max_temp,
            feature_columns[3]: humidity9am,
            feature_columns[4]: humidity3pm,
            feature_columns[5]: pressure9am,
            feature_columns[6]: pressure3pm,
            feature_columns[7]: wind9am,
            feature_columns[8]: wind3pm,
            feature_columns[9]: rain_today,
            feature_columns[10]: year,
            feature_columns[11]: month,
            feature_columns[12]: day
        }
        
        input_df = pd.DataFrame([input_data], columns=feature_columns)
        input_scaled = scaler.transform(input_df)
        
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        rain_prob = probs[1] * 100
        
        st.divider()
        st.markdown("### 📊 Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if pred == 1:
                st.error("🌧️ **Rain Tomorrow**")
            else:
                st.success("☀️ **Sunny Tomorrow**")
        with col2: st.metric("Rain Chance", f"{rain_prob:.1f}%")
        with col3: st.metric("Risk", "High" if rain_prob > 60 else "Low")
        
        st.progress(rain_prob / 100)
        st.balloons() if pred == 0 else st.warning("☔ Take umbrella!")
        
        with st.expander("Inputs Used"):
            st.json(input_data)
            
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.caption("Clean RainCast v3 | Model-Exact Inputs Only")
