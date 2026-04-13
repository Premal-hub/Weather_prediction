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
        .main {
            background-color: #f8fafc;
        }
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: white;
            padding: 18px;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }
        .section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }
        .footer-text {
            text-align: center;
            color: #64748b;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# FILE PATHS
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

    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        missing_files.append(SCALER_PATH)
    if not os.path.exists(FEATURES_PATH):
        missing_files.append(FEATURES_PATH)

    return missing_files

# =========================================================
# DEFAULT INPUT VALUES
# (Safe fallback defaults if feature exists in model)
# =========================================================
DEFAULT_VALUES = {
    "MinTemp": 15.0,
    "MaxTemp": 25.0,
    "Rainfall": 0.0,
    "Humidity9am": 60.0,
    "Humidity3pm": 55.0,
    "Pressure9am": 1015.0,
    "Pressure3pm": 1012.0,
    "Cloud9am": 4.0,
    "Cloud3pm": 5.0,
    "Temp9am": 18.0,
    "Temp3pm": 24.0,
    "RainToday": 0
}

# =========================================================
# HUMAN-READABLE LABELS
# =========================================================
FEATURE_LABELS = {
    "MinTemp": "Minimum Temperature (°C)",
    "MaxTemp": "Maximum Temperature (°C)",
    "Rainfall": "Rainfall (mm)",
    "Evaporation": "Evaporation (mm)",
    "Sunshine": "Sunshine (hours)",
    "WindGustSpeed": "Wind Gust Speed (km/h)",
    "WindSpeed9am": "Wind Speed at 9 AM (km/h)",
    "WindSpeed3pm": "Wind Speed at 3 PM (km/h)",
    "Humidity9am": "Humidity at 9 AM (%)",
    "Humidity3pm": "Humidity at 3 PM (%)",
    "Pressure9am": "Pressure at 9 AM (hPa)",
    "Pressure3pm": "Pressure at 3 PM (hPa)",
    "Cloud9am": "Cloud Cover at 9 AM (oktas)",
    "Cloud3pm": "Cloud Cover at 3 PM (oktas)",
    "Temp9am": "Temperature at 9 AM (°C)",
    "Temp3pm": "Temperature at 3 PM (°C)",
    "RainToday": "Did it rain today?"
}

# =========================================================
# INPUT RANGES FOR BETTER UX
# =========================================================
INPUT_CONFIG = {
    "MinTemp": {"min": -20.0, "max": 50.0, "step": 0.1},
    "MaxTemp": {"min": -10.0, "max": 60.0, "step": 0.1},
    "Rainfall": {"min": 0.0, "max": 500.0, "step": 0.1},
    "Evaporation": {"min": 0.0, "max": 100.0, "step": 0.1},
    "Sunshine": {"min": 0.0, "max": 24.0, "step": 0.1},
    "WindGustSpeed": {"min": 0.0, "max": 200.0, "step": 1.0},
    "WindSpeed9am": {"min": 0.0, "max": 150.0, "step": 1.0},
    "WindSpeed3pm": {"min": 0.0, "max": 150.0, "step": 1.0},
    "Humidity9am": {"min": 0.0, "max": 100.0, "step": 1.0},
    "Humidity3pm": {"min": 0.0, "max": 100.0, "step": 1.0},
    "Pressure9am": {"min": 900.0, "max": 1100.0, "step": 0.1},
    "Pressure3pm": {"min": 900.0, "max": 1100.0, "step": 0.1},
    "Cloud9am": {"min": 0.0, "max": 8.0, "step": 1.0},
    "Cloud3pm": {"min": 0.0, "max": 8.0, "step": 1.0},
    "Temp9am": {"min": -20.0, "max": 50.0, "step": 0.1},
    "Temp3pm": {"min": -10.0, "max": 60.0, "step": 0.1},
}

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def create_input_widget(feature_name):
    """
    Dynamically create input widgets based on feature name.
    """
    label = FEATURE_LABELS.get(feature_name, feature_name)

    if feature_name == "RainToday":
        return st.selectbox(
            label,
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes",
            index=DEFAULT_VALUES.get(feature_name, 0)
        )

    config = INPUT_CONFIG.get(feature_name, {"min": -9999.0, "max": 9999.0, "step": 0.1})
    default_value = float(DEFAULT_VALUES.get(feature_name, 0.0))

    return st.number_input(
        label,
        min_value=float(config["min"]),
        max_value=float(config["max"]),
        value=default_value,
        step=float(config["step"])
    )

def get_risk_level(probability):
    """
    Return a text risk level based on rain probability.
    """
    if probability >= 80:
        return "Very High"
    elif probability >= 60:
        return "High"
    elif probability >= 40:
        return "Moderate"
    elif probability >= 20:
        return "Low"
    else:
        return "Very Low"

def get_recommendation(prediction, probability):
    """
    Return a user-friendly recommendation.
    """
    if prediction == 1:
        if probability >= 80:
            return "Carry an umbrella and plan for possible rain. Outdoor activities may be affected."
        elif probability >= 60:
            return "Rain is likely. It is recommended to carry rain protection."
        else:
            return "There is a chance of rain. Keep an eye on weather conditions."
    else:
        if probability <= 20:
            return "Weather appears stable. Rain is unlikely tomorrow."
        else:
            return "Rain is not predicted, but slight changes in conditions are possible."

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("🌧️ RainCast")
    st.markdown("### Machine Learning Based Rain Prediction")
    st.markdown("---")

    st.info(
        """
        **How it works:**
        - Enter weather parameters
        - Click **Predict Rain Forecast**
        - The trained ML model predicts whether it will rain tomorrow
        """
    )

    st.markdown("---")
    st.markdown("### 📁 Required Files")
    st.caption("Keep these files in the same folder:")
    st.code(
        "app.py\nbest_raincast_model.pkl\nraincast_scaler.pkl\nraincast_feature_columns.pkl",
        language="bash"
    )

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown('<div class="main-title">🌧️ RainCast - Rain Forecast Prediction System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A professional machine learning application that predicts whether it is likely to rain tomorrow based on weather parameters.</div>',
    unsafe_allow_html=True
)

# =========================================================
# CHECK FILES
# =========================================================
missing_files = check_required_files()

if missing_files:
    st.error("❌ Some required files are missing.")
    st.write("Please make sure the following files exist in the same folder as `app.py`:")

    for file in missing_files:
        st.write(f"- `{file}`")

    st.stop()

# =========================================================
# LOAD RESOURCES
# =========================================================
try:
    model, scaler, feature_columns = load_model_resources()
except Exception as e:
    st.error(f"❌ Failed to load model resources: {e}")
    st.stop()

# =========================================================
# TOP METRICS / INFO
# =========================================================
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Model Status", "Loaded ✅")

with col_b:
    st.metric("Input Features", f"{len(feature_columns)}")

with col_c:
    st.metric("Session Time", datetime.now().strftime("%H:%M"))

st.markdown("---")

# =========================================================
# INPUT FORM
# =========================================================
st.markdown('<div class="section-title">📝 Enter Weather Parameters</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    user_input = {}

    # Split inputs into 3 columns for professional layout
    col1, col2, col3 = st.columns(3)

    for idx, feature in enumerate(feature_columns):
        if idx % 3 == 0:
            with col1:
                user_input[feature] = create_input_widget(feature)
        elif idx % 3 == 1:
            with col2:
                user_input[feature] = create_input_widget(feature)
        else:
            with col3:
                user_input[feature] = create_input_widget(feature)

    st.markdown("")
    predict_button = st.form_submit_button("🔍 Predict Rain Forecast", use_container_width=True)

# =========================================================
# PREDICTION SECTION
# =========================================================
if predict_button:
    try:
        # Create dataframe from user input
        input_df = pd.DataFrame([user_input])

        # Ensure exact column order
        input_df = input_df[feature_columns]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        # Predict probability if available
        rain_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]

            # Try to safely detect class index for class 1
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    class_1_index = classes.index(1)
                    rain_probability = probabilities[class_1_index] * 100
                else:
                    # fallback
                    rain_probability = probabilities[-1] * 100
            else:
                rain_probability = probabilities[-1] * 100
        else:
            # fallback if no predict_proba
            rain_probability = 100.0 if prediction == 1 else 0.0

        risk_level = get_risk_level(rain_probability)
        recommendation = get_recommendation(prediction, rain_probability)

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Prediction Results</div>', unsafe_allow_html=True)

        result_col1, result_col2, result_col3 = st.columns(3)

        # Result
        with result_col1:
            if prediction == 1:
                st.error("🌧️ **Rain Predicted**")
            else:
                st.success("☀️ **No Rain Predicted**")

        # Probability
        with result_col2:
            st.metric("Chance of Rain", f"{rain_probability:.2f}%")

        # Risk Level
        with result_col3:
            st.metric("Risk Level", risk_level)

        # Progress bar for probability
        st.markdown("### Rain Probability")
        st.progress(min(int(rain_probability), 100))

        # Friendly message
        if prediction == 1:
            st.warning("⚠️ The model predicts that it is likely to rain tomorrow.")
        else:
            st.info("ℹ️ The model predicts that rain is unlikely tomorrow.")

        # Recommendation box
        st.markdown("### 📌 Recommendation")
        st.success(recommendation)

        # Show input summary (optional but professional)
        with st.expander("🔎 View Submitted Input Summary"):
            summary_df = pd.DataFrame({
                "Feature": input_df.columns,
                "Value": input_df.iloc[0].values
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer-text">
        Built with ❤️ using <b>Streamlit</b>, <b>Scikit-learn</b>, and <b>Machine Learning</b><br>
        Project: <b>RainCast - Rain Forecast Prediction System</b>
    </div>
    """,
    unsafe_allow_html=True
)