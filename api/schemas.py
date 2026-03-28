from pydantic import BaseModel

class PredictionResponse(BaseModel):
    ticker:     str
    date:       str
    prediction: str
    confidence: float
    model:      str

class EnsembleResponse(PredictionResponse):
    rf_confidence:   float
    lstm_confidence: float

class HealthResponse(BaseModel):
    status:        str
    scaler_loaded: bool