import requests
import time

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health ...", end=" ")
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    print("PASSED")

def test_tickers():
    print("Testing /tickers ...", end=" ")
    response = requests.get(f"{BASE_URL}/tickers")
    assert response.status_code == 200
    data = response.json()
    assert "AAPL" in data["tickers"]
    print("PASSED")

def test_predict_rf():
    print("Testing /predict/rf?ticker=AAPL ...", end=" ")
    response = requests.get(f"{BASE_URL}/predict/rf", params={"ticker": "AAPL"})
    if response.status_code == 200:
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "prediction" in data
        print("PASSED")
    elif response.status_code == 503:
        print("SKIPPED (Model not loaded)")
    else:
        print(f"FAILED (Status {response.status_code}: {response.text})")

def test_predict_lstm():
    print("Testing /predict/lstm?ticker=AAPL ...", end=" ")
    response = requests.get(f"{BASE_URL}/predict/lstm", params={"ticker": "AAPL"})
    if response.status_code == 200:
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "prediction" in data
        print("PASSED")
    elif response.status_code == 503:
        print("SKIPPED (Model/Scaler not loaded)")
    else:
        print(f"FAILED (Status {response.status_code}: {response.text})")

def test_predict_ensemble():
    print("Testing /predict/ensemble?ticker=AAPL ...", end=" ")
    response = requests.get(f"{BASE_URL}/predict/ensemble", params={"ticker": "AAPL"})
    if response.status_code == 200:
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "rf_confidence" in data
        assert "lstm_confidence" in data
        print("PASSED")
    elif response.status_code == 503:
        print("SKIPPED (Models not loaded)")
    else:
        print(f"FAILED (Status {response.status_code}: {response.text})")

if __name__ == "__main__":
    try:
        test_health()
        test_tickers()
        test_predict_rf()
        test_predict_lstm()
        test_predict_ensemble()
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API. Is 'python main.py' running?")
