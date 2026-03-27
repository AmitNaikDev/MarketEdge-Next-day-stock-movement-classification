# 📈 Stock Price Movement Classifier

An end-to-end ML pipeline that predicts **next-day stock price direction (UP/DOWN)**
using a dual model approach — Random Forest + LSTM — served via a REST API.

---

## 🧠 Models

| Model         | Approach         | Input Type              |
|---------------|------------------|-------------------------|
| Random Forest | Classical ML     | Flat feature vector     |
| LSTM          | Deep Learning    | 20-day sequence window  |
| Ensemble      | RF + LSTM avg    | Both                    |

---

## 🏗️ Project Structure
```
stock_classifier/
├── src/
│   ├── data_loader.py      # yfinance data fetch + target engineering
│   ├── features.py         # technical indicators (RSI, MACD, BB, ATR)
│   ├── model_ml.py         # Random Forest pipeline + TimeSeriesSplit CV
│   ├── model_dl.py         # Stacked LSTM with Dropout + BatchNorm
│   ├── evaluate.py         # metrics, ROC/PR curves, backtesting
│   └── predict.py          # RF / LSTM / Ensemble inference
├── api/
│   ├── main.py             # FastAPI app init (wiring only)
│   ├── dependencies.py     # model loading at startup
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/
│       └── prediction.py   # all route handlers
├── models/                 # saved .pkl and .keras files
├── outputs/                # charts, backtest plots
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup
```bash
# 1. Clone the repo
git clone https://github.com/your-username/stock-classifier.git
cd stock-classifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create output directories
mkdir -p models outputs
```

---

## 🚀 Usage

### Step 1 — Train the Random Forest model
```bash
python src/model_ml.py
```

### Step 2 — Train the LSTM model
```bash
python src/model_dl.py
```

### Step 3 — Run the API
```bash
uvicorn api.main:app --reload --port 8000
```

### Step 4 — Test a prediction
```bash
curl "http://localhost:8000/predict/ensemble?ticker=AAPL"
```

Expected response:
```json
{
  "ticker": "AAPL",
  "date": "2024-06-10",
  "prediction": "UP",
  "confidence": 0.6132,
  "rf_confidence": 0.5900,
  "lstm_confidence": 0.6364,
  "model": "Ensemble (RF + LSTM)"
}
```

---

## 📊 Features Used

| Category   | Indicators                                      |
|------------|-------------------------------------------------|
| Trend      | EMA(20), SMA(50), MACD, MACD Signal, MACD Diff  |
| Momentum   | RSI(14), Stochastic K, Stochastic D             |
| Volatility | Bollinger Bands (High/Low/Width), ATR(14)       |
| Lag        | 1, 2, 3, 5-day returns                          |

---

## 🔁 API Endpoints

| Endpoint                  | Description                        |
|---------------------------|------------------------------------|
| `GET /health`             | API status + model load check      |
| `GET /predict/rf`         | Random Forest prediction           |
| `GET /predict/lstm`       | LSTM prediction                    |
| `GET /predict/ensemble`   | Ensemble prediction (recommended)  |
| `GET /tickers`            | Sample supported tickers           |
| `GET /docs`               | Auto-generated Swagger UI          |

---

## 📉 Backtesting

The `evaluate.py` module includes a simple **long-only backtest** that compares
the model's strategy returns against a Buy & Hold benchmark.
```
--- Backtest Results: Ensemble ---
Initial Capital  : $10,000.00
Buy & Hold Final : $14,230.00  (+42.30%)
Strategy Final   : $16,880.00  (+68.80%)
Total Trades     : 312
```

> ⚠️ This is a simplified backtest. It does not account for transaction costs,
> slippage, or market impact. Do not use for real trading.

---

## 🛡️ Key ML Practices Applied

- **No data leakage** — scaler fit on train set only, `TimeSeriesSplit` for CV
- **Class imbalance** — handled via `class_weight="balanced"` in Random Forest
- **Regularization** — Dropout + BatchNormalization in LSTM
- **Early stopping** — prevents LSTM overfitting via `EarlyStopping` callback
- **Modular codebase** — each file has a single responsibility

---

## 🧪 Experiment Tracking (Optional)

This project supports **MLflow** for tracking experiments.
```bash
# Start MLflow UI
mlflow ui

# Open in browser
http://localhost:5000
```

---

## 📦 Tech Stack

`yfinance` · `ta` · `scikit-learn` · `TensorFlow/Keras` · `FastAPI` · `MLflow` · `matplotlib`

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.
Predictions are not financial advice.