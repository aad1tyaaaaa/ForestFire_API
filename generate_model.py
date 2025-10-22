import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# --- Step 1: Loading Dataset from CSV (Hardcoded Dataset for Uttarakhand Forest Fire Prediction) ---

print("--- Step 1: Loading Hardcoded Dataset from CSV ---")

# Load the hardcoded dataset
df_ml = pd.read_csv('dataset.csv')

print(f"✅ Dataset loaded from CSV. Total ML Samples: {len(df_ml)}")
print("Dataset columns:", list(df_ml.columns))
print("Sample data:")
print(df_ml.head())

# --- Step 2: ML Model Training (Random Forest - Faster than XGBoost) ---

print("\n--- Step 2: Training Random Forest Model with ALL 8 Factors ---")

# AB FEATURES MEIN SAARE 8 FACTORS SHAMIL HAIN
features = ['X_frp', 'slope', 'temp', 'humidity', 'wind_speed', 'fuel_dryness', 'pop_density', 'dist_to_road']
target = 'Y_is_burned'
X = df_ml[features].fillna(0)
Y = df_ml[target].astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Using Random Forest instead of XGBoost for much faster loading and similar performance
rf_model = RandomForestClassifier(
    n_estimators=10, max_depth=5, random_state=42, n_jobs=-1
)

rf_model.fit(X_train, Y_train)

Y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print(f"Model AUC on Test Data: {roc_auc_score(Y_test, Y_pred_proba):.4f}")

joblib.dump(rf_model, 'uttarakhand_fire_model_ultimate.pkl')
print("\n✅ Ultimate Model Trained and saved as 'uttarakhand_fire_model_ultimate.pkl' (Random Forest for faster loading).")
