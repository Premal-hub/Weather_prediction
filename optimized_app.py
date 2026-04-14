import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="RainCast | Rain Forecast Prediction",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS FOR PROFESSIONAL UI
# =========================================================
st.markdown("""
    <style>
        .main { background-color: #f8fafc; }
        .main-title { font-size: 2.3rem; font-weight: 800; color: #0f172a; margin-bottom: 0.2rem; }
        .sub-title { font-size: 1rem; color: #475569; margin-bottom: 1.5rem; }
        .metric-card { background: white; padding: 18px; border-radius: 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; }
        .section-title { font-size: 1.2rem; font-weight: 700; color: #1e293b; margin-top: 1rem; margin-bottom: 0.8rem; }
        .footer-text { text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# FILE PATHS (Model expects exactly these 13 features)
# =========================================================
MODEL_PATH = "best_raincast_model.pkl"
SCALER_PATH = "raincast_scaler.pkl"
FEATURES_PATH = "raincast_feature_columns.pkl"

# =========================================================
# LOAD MODEL FILES
# =========================================================
@st.cache_resource
def load_model_resources():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, scaler, feature_columns

def check_required_files():
    missing_files = []
    for path in [MODEL_PATH, SCALER_PATH, FEATURES_PATH]:
        if not os.path.exists(path):
            missing_files.append(path)
    return missing_files

# =========================================================
# MODEL FEATURES ONLY - Optimized Defaults
# =========================================================
DEFAULT_VALUES = {
    'Location': 0,  # Encoded integer
    'MinTemp': 20.0,
    'MaxTemp': 32.0,
    'Humidity9am': 70.0,
    'Humidity3pm': 50.0,
    'Pressure9am': 1010.0,
    'Pressure3pm': 1005.0,
    'WindSpeed9am': 13.0,
    'WindSpeed3pm': 15.0,
    'RainToday': 0,
    'Year': 2025,
    'Month': 6,
    'Day': 15
}

# =========================================================
# MODEL FEATURES ONLY - Clean Labels
# =========================================================
# FEATURE_LABELS = {
#     'Location': 'Location (Encoded)',
    'MinTemp': 'Minimum Temperature (°C)',
    'MaxTemp': 'Maximum Temperature (°C)',
    'Humidity9am': 'Humidity at 9 AM (%)',
    'Humidity3pm': 'Humidity at 3 PM (%)',
    'Pressure9am': 'Pressure at 9 AM (hPa)',
    'Pressure3pm': 'Pressure at 3 PM (hPa)',
    'WindSpeed9am': 'Wind Speed 9 AM (km/h)',
    'WindSpeed3pm': 'Wind Speed 3 PM (km/h)',
    'RainToday': 'Rain Today?',
    # 'Year': 'Year',
    # 'Month': 'Month (1-12)',
    # 'Day': 'Day (1-31)'
}

# =========================================================
# MODEL FEATURES ONLY - Input Ranges
# =========================================================
INPUT_CONFIG = {
    # 'Location': {'min': 0, 'max': 50, 'step': 1},  # Adjust based on unique locations
    'MinTemp': {'min': -10, 'max': 50, 'step': 0.1},
    'MaxTemp': {'min': 0, 'max': 60, 'step': 0.1},
    'Humidity9am': {'min': 0, 'max': 100, 'step': 1},
    'Humidity3pm': {'min': 0, 'max': 100, 'step': 1},
    'Pressure9am': {'min': 900, 'max': 1100, 'step': 0.1},
    'Pressure3pm': {'min': 900, 'max': 1100, 'step': 0.1},
    'WindSpeed9am': {'min': 0, 'max': 100, 'step': 1},
    'WindSpeed3pm': {'min': 0, 'max': 100, 'step': 1},
    # 'Year': {'min': 2020, 'max': 2030, 'step': 1},
    # 'Month': {'min': 1, 'max': 12, 'step': 1},
    # 'Day': {'min': 1, 'max': 31, 'step': 1}
}

# =========================================================
# HELPER FUNCTIONS - Optimized
# =========================================================
def create_input_widget(feature_name):
    label = FEATURE_LABELS.get(feature_name, feature_name)
    
    if feature_name == 'RainToday':
        return st.selectbox(label, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    if feature_name == 'Location':
        return st.number_input(label, min_value=0, max_value=50, step=1, value=0)
    
    config = INPUT_CONFIG.get(feature_name, {'min': -999, 'max': 999, 'step': 1})
    default = DEFAULT_VALUES.get(feature_name, 0)
    
    return st.number_input(label, 
                          min_value=float(config['min']), 
                          max_value=float(config['max']), 
                          value=float(default), 
                          step=float(config['step']))

def get_risk_level(probability):
    if probability >= 80: return "Very High"
    elif probability >= 60: return "High"
    elif probability >= 40: return "Moderate" 
    elif probability >= 20: return "Low"
    return "Very Low"

def get_recommendation(prediction, probability):
    if prediction == 1:
        if probability >= 80: return "Umbrella essential. Heavy rain likely."
        elif probability >= 60: return "Rain protection recommended."
        return "Rain possible. Stay prepared."
    return "Rain unlikely tomorrow."

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🌧️ RainCast")
    st.markdown("### ML Rain Prediction")
    st.markdown("---")
    st.info("📝 Enter **today's** weather → Predict **tomorrow**")
    st.markdown("---")
    st.caption("✅ Optimized: 13 model features only")

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown('<div class="main-title">🌧️ RainCast - Optimized Rain Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict tomorrow rain using 13 key features (no extras)</div>', unsafe_allow_html=True)

# =========================================================
# CHECK FILES
# =========================================================
missing_files = check_required_files()
if missing_files:
    st.error("❌ Missing files: " + ", ".join(missing_files))
    st.stop()

# =========================================================
# LOAD RESOURCES
# =========================================================
try:
    model, scaler, feature_columns = load_model_resources()
    st.success(f"✅ Model loaded: {len(feature_columns)} features")
except Exception as e:
    st.error(f"❌ Load error: {e}")
    st.stop()

# =========================================================
# METRICS
# =========================================================
col1, col2, col3 = st.columns(3)
with col1: st.metric("Features", len(feature_columns))
with col2: st.metric("Status", "Ready ✅")
with col3: st.metric("Time", datetime.now().strftime("%H:%M"))

st.markdown("---")

# =========================================================
# INPUT FORM - Model Features ONLY
# =========================================================
st.markdown('<div class="section-title">📊 Enter Today\'s Weather Data</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    user_input = {}
    
    # 3-column layout for 13 fields
    cols = st.columns(3)
    col_idx = 0
    
    for feature in feature_columns:
        with cols[col_idx]:
            user_input[feature] = create_input_widget(feature)
        col_idx = (col_idx + 1) % 3
    
    st.markdown("---")
    predict_btn = st.form_submit_button("🔮 Predict Tomorrow's Rain", use_container_width=True)

# =========================================================
# PREDICTION
# =========================================================
if predict_btn:
    try:
        # Create DataFrame WITH column names (fixes sklearn warning)
        input_df = pd.DataFrame([user_input], columns=feature_columns)
        
        # Scale
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Probability (RandomForest has predict_proba)
        probabilities = model.predict_proba(input_scaled)[0]
        # Class 1 (rain) probability
        rain_prob = probabilities[1] * 100 if len(probabilities) > 1 else (probabilities[0] * 100 if prediction == 1 else 0)
        
        risk = get_risk_level(rain_prob)
        rec = get_recommendation(prediction, rain_prob)
        
        # Results
        st.markdown("---")
        st.markdown('<div class="section-title">📈 Prediction Results</div>', unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            color = "inverse" if prediction == 1 else "normal"
            st.metric("Tomorrow", "🌧️ Rain" if prediction == 1 else "☀️ Clear", delta=None)
        with c2: st.metric("Probability", f"{rain_prob:.1f}%")
        with c3: st.metric("Risk", risk)
        
        # Progress bar
        st.progress(min(rain_prob / 100, 1.0))
        
        # Message
        st.balloons() if prediction == 0 else st.error("☔ Rain likely!")
        st.info(rec)
        
        # Input summary
        with st.expander("📋 Input Summary"):
            st.dataframe(input_df.T, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown('<div class="footer-text">Optimized RainCast v2.0 | 13 Model Features Only | No Unused Fields</div>', unsafe_allow_html=True)

