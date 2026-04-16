import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Paths
DATA_PATH = "d:/Premal/weather_prediction/weather_prediction/rain_forecasting.csv"
MODEL_PATH = "d:/Premal/weather_prediction/weather_prediction/best_raincast_model.pkl"
SCALER_PATH = "d:/Premal/weather_prediction/weather_prediction/raincast_scaler.pkl"
FEATURES_PATH = "d:/Premal/weather_prediction/weather_prediction/raincast_feature_columns.pkl"

if not os.path.exists(DATA_PATH):
    print("No data found!")
    exit(1)

# Load resources
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
except Exception as e:
    print("Could not load model:", e)
    exit(1)

# Load data
df = pd.read_csv(DATA_PATH)
print("Data Shape:", df.shape)
print("Columns:", df.columns.tolist())

# The target is likely RainTomorrow
target_col = "RainTomorrow"
if target_col not in df.columns:
    print("Could not find target column!")
    exit(1)

# Drop missing values for simplicity of baseline
df = df.dropna(subset=[target_col])
y = df[target_col].map({'Yes': 1, 'No': 0}) if df[target_col].dtype == 'object' else df[target_col]

X = df.drop(columns=[target_col])

# Force features to float, keeping what model expects
X_prepared = X.copy()
for col in features:
    if col not in X_prepared.columns:
        X_prepared[col] = 0
    else:
        X_prepared[col] = pd.to_numeric(X_prepared[col], errors='coerce').fillna(0)

X_final = X_prepared[features]

try:
    X_scaled = scaler.transform(X_final)
    y_pred = model.predict(X_scaled)
    # This evaluates on entire dataset (just to see what it's capable of)
    print("Current Model Training Data Accuracy:", accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))
except Exception as e:
    print("Evaluation failed:", e)
