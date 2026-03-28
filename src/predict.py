# src/predict.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import DEFAULT_THRESHOLD, DEFAULT_WINDOW, WARMUP_DAYS
from src.data_loader import load_stock_data, create_target
from src.features import add_technical_indicators, get_feature_columns

def prepare_inference_data(ticker: str) -> pd.DataFrame:
    """Fetches and processes data for a single ticker."""
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")

    df = load_stock_data(ticker, start=start, end=end)
    # df = create_target(df) # Target not strictly needed for inference
    df = add_technical_indicators(df)
    return df

# ── 1. RF Prediction ──────────────────────────────────────────────────────────

def predict_rf(ticker: str, model, df: pd.DataFrame = None, 
               threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Predicts next-day movement using a pre-loaded RF model.
    """
    if df is None:
        df = prepare_inference_data(ticker)

    feature_cols = get_feature_columns(df)
    X_latest = df[feature_cols].iloc[[-1]]

    prob = model.predict_proba(X_latest)[0][1]   # P(Up)
    direction = "UP" if prob >= threshold else "DOWN"

    return {
        "ticker":     ticker,
        "date":       df.index[-1].strftime("%Y-%m-%d"),
        "prediction": direction,
        "confidence": round(float(prob), 4),
        "model":      "RandomForest"
    }


# ── 2. LSTM Prediction ────────────────────────────────────────────────────────

def predict_lstm(ticker: str, model, scaler, df: pd.DataFrame = None,
                 window: int = DEFAULT_WINDOW, 
                 threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Predicts next-day movement using a pre-loaded LSTM model and scaler.
    """
    if df is None:
        df = prepare_inference_data(ticker)

    feature_cols = get_feature_columns(df)
    
    # Scale using the pre-loaded scaler
    X_scaled = scaler.transform(df[feature_cols].values)

    if len(X_scaled) < window:
        raise ValueError(f"Not enough data: need {window} rows, got {len(X_scaled)}")

    X_seq = X_scaled[-window:].reshape(1, window, -1)   # (1, window, features)

    prob = float(model.predict(X_seq, verbose=0)[0][0])
    direction = "UP" if prob >= threshold else "DOWN"

    return {
        "ticker":     ticker,
        "date":       df.index[-1].strftime("%Y-%m-%d"),
        "prediction": direction,
        "confidence": round(prob, 4),
        "model":      "LSTM"
    }


# ── 3. Ensemble Prediction ────────────────────────────────────────────────────

def predict_ensemble(ticker: str, rf_model, lstm_model, scaler, 
                     window: int = DEFAULT_WINDOW) -> dict:
    """
    Averages RF and LSTM probabilities using a single data fetch.
    """
    # Fetch data ONCE for both models
    df = prepare_inference_data(ticker)

    rf_result = predict_rf(ticker, model=rf_model, df=df)
    lstm_result = predict_lstm(ticker, model=lstm_model, scaler=scaler, df=df, window=window)

    avg_prob  = (rf_result["confidence"] + lstm_result["confidence"]) / 2
    direction = "UP" if avg_prob >= 0.5 else "DOWN"

    return {
        "ticker":          ticker,
        "date":            rf_result["date"],
        "prediction":      direction,
        "confidence":      round(avg_prob, 4),
        "rf_confidence":   rf_result["confidence"],
        "lstm_confidence": lstm_result["confidence"],
        "model":           "Ensemble (RF + LSTM)"
    }


if __name__ == "__main__":
    # Smoke test requires models and data. 
    # In a real scenario, we'd mock these or use a test model.
    pass