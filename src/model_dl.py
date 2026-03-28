import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

def create_sequences(X: np.ndarray, Y: np.ndarray, window: int = 20):
    """Create sliding window sequences for LSTM input"""
    Xs, Ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        Ys.append(Y[i])
    return np.array(Xs), np.array(Ys)

def prepare_lstm_data(dataset: pd.DataFrame, feature_cols: list, window: int = 20, test_size: float = 0.2):
    """Prepare data for LSTM model"""
    
    split_idx = int(len(dataset) * (1 - test_size))
    X = dataset[feature_cols].values
    Y = dataset["Target"].values
    
    scaler = StandardScaler()
    # Scale based on training data only to avoid leakage
    X_train_raw = X[:split_idx]
    X_test_raw = X[split_idx:]
    
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # Build Sequences
    X_train, Y_train = create_sequences(X_train_scaled, Y[:split_idx], window)
    X_test, Y_test = create_sequences(X_test_scaled, Y[split_idx:], window)
    
    print(f"[INFO] X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"[INFO] Class balance (train): {np.bincount(Y_train)}")

    return X_train, X_test, Y_train, Y_test, scaler

def build_lstm_model(input_shape: tuple) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        BatchNormalization(),

        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),

        Dense(32, activation="relu"),
        Dropout(0.3),

        Dense(1, activation="sigmoid")   # P(price goes up)
    ]) 

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model 

def train_lstm(model: Sequential, X_train, Y_train, X_test, Y_test, epochs: int = 50) -> tuple:
    """
    Trains with EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint.
    """
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint("models/lstm_best.keras", monitor="val_loss", save_best_only=True, verbose=1)
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    return model, history

def evaluate_lstm(model: Sequential, X_test, Y_test, threshold: float = 0.5) -> None:
    Y_prob = model.predict(X_test).flatten()
    Y_pred = (Y_prob >= threshold).astype(int)

    print(f"\n--- LSTM Evaluation (threshold={threshold}) ---")
    print(classification_report(Y_test, Y_pred, target_names=["Down", "Up"]))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, Y_pred))

def save_lstm(model, path: str = "models/lstm_model.keras") -> None:
    model.save(path)
    print(f"[INFO] LSTM saved to {path}")

def load_lstm(path: str = "models/lstm_model.keras"):
    return load_model(path)

if __name__ == "__main__":
    import sys
    import os
    # Allow running directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import load_stock_data, create_target
    from src.features import add_technical_indicators, get_feature_columns

    df = load_stock_data("AAPL", "2018-01-01", "2024-01-01")
    df = create_target(df)
    df = add_technical_indicators(df)

    feature_cols = get_feature_columns(df)

    X_train, X_test, Y_train, Y_test, scaler = prepare_lstm_data(df, feature_cols)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model, history = train_lstm(model, X_train, Y_train, X_test, Y_test, epochs=5) # Reduced epochs for testing

    evaluate_lstm(model, X_test, Y_test)
    save_lstm(model)
    
    # Save scaler as well
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    print("[INFO] Scaler saved to models/scaler.pkl")

