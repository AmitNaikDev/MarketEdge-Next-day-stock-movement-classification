import joblib
from tensorflow.keras.models import load_model as tf_load_model
from src.config import RF_MODEL_PATH, LSTM_MODEL_PATH, SCALER_PATH

app_state = {
    "scaler": None,
    "rf_model": None,
    "lstm_model": None
}

def load_models():
    print("[STARTUP] Loading models ...")
    
    # Load Scaler
    try:
        app_state["scaler"] = joblib.load(SCALER_PATH)
        print(f"[SUCCESS] Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load scaler: {e}")

    # Load RF Model
    try:
        app_state["rf_model"] = joblib.load(RF_MODEL_PATH)
        print(f"[SUCCESS] RF Model loaded from {RF_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load RF Model: {e}")

    # Load LSTM Model
    try:
        app_state["lstm_model"] = tf_load_model(LSTM_MODEL_PATH)
        print(f"[SUCCESS] LSTM Model loaded from {LSTM_MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] Failed to load LSTM Model: {e}")

def clear_models():
    app_state["scaler"] = None
    app_state["rf_model"] = None
    app_state["lstm_model"] = None
    print("[SHUTDOWN] App state cleared")

def get_scaler():
    return app_state.get("scaler")

def get_rf_model():
    return app_state.get("rf_model")

def get_lstm_model():
    return app_state.get("lstm_model")