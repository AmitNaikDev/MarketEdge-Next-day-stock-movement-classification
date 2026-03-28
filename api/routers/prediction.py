from fastapi import APIRouter, HTTPException, Query
from api.schemas import PredictionResponse, EnsembleResponse, HealthResponse
from api.dependencies import get_scaler, get_rf_model, get_lstm_model
from src.predict import predict_rf, predict_lstm, predict_ensemble
from src.config import SUPPORTED_TICKERS

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Meta"])
def health_check():
    return {
        "status":        "ok",
        "scaler_loaded": get_scaler() is not None
    }


@router.get("/predict/rf", response_model=PredictionResponse, tags=["Prediction"])
def predict_random_forest(
    ticker:    str   = Query(..., description="Ticker e.g. AAPL"),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    model = get_rf_model()
    if model is None:
        raise HTTPException(503, "RF model not loaded. Check server logs.")
    
    try:
        return predict_rf(ticker.upper(), model=model, threshold=threshold)
    except Exception as e:
        raise HTTPException(500, f"RF Prediction failed: {str(e)}")


@router.get("/predict/lstm", response_model=PredictionResponse, tags=["Prediction"])
def predict_lstm_endpoint(
    ticker:    str   = Query(..., description="Ticker e.g. AAPL"),
    window:    int   = Query(20, ge=5, le=60),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    model = get_lstm_model()
    scaler = get_scaler()
    
    if model is None or scaler is None:
        raise HTTPException(503, "LSTM model or Scaler not loaded.")
        
    try:
        return predict_lstm(ticker.upper(), model=model, scaler=scaler,
                            window=window, threshold=threshold)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"LSTM Prediction failed: {str(e)}")


@router.get("/predict/ensemble", response_model=EnsembleResponse, tags=["Prediction"])
def predict_ensemble_endpoint(
    ticker: str = Query(..., description="Ticker e.g. AAPL"),
    window: int = Query(20, ge=5, le=60)
):
    rf_model = get_rf_model()
    lstm_model = get_lstm_model()
    scaler = get_scaler()
    
    if any(m is None for m in [rf_model, lstm_model, scaler]):
        raise HTTPException(503, "One or more models/scaler not loaded.")
        
    try:
        return predict_ensemble(ticker.upper(), rf_model=rf_model, 
                                lstm_model=lstm_model, scaler=scaler, window=window)
    except Exception as e:
        raise HTTPException(500, f"Ensemble Prediction failed: {str(e)}")


@router.get("/tickers", tags=["Meta"])
def supported_tickers():
    return {
        "tickers": SUPPORTED_TICKERS
    }
