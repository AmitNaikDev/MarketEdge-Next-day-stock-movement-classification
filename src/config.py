# src/config.py

import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model Paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Supported Tickers
SUPPORTED_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN",
    "NVDA", "META", "NFLX", "JPM", "SPY"
]

# Inference Parameters
DEFAULT_WINDOW = 20
DEFAULT_THRESHOLD = 0.5
WARMUP_DAYS = 120 # Enough for 50-day SMA and LSTM window
