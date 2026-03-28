# src/evaluate.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve
)


# ── 1. Full Evaluation Report ─────────────────────────────────────────────────

def full_evaluation(y_true, y_pred, y_prob, model_name: str = "Model") -> dict:
    """
    Prints a complete evaluation report.
    Returns metrics as a dict for easy comparison between RF and LSTM.
    """
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["Down", "Up"]))

    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    auc = roc_auc_score(y_true, y_prob)
    print(f"\n--- ROC-AUC Score: {auc:.4f} ---")

    return {
        "model":    model_name,
        "roc_auc":  auc,
        "confusion_matrix": cm
    }


# ── 2. Plots ──────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true, y_prob_rf, y_prob_lstm) -> None:
    """
    Overlays ROC curves for both models on one plot.
    Makes for a great README chart.
    """
    fpr_rf,   tpr_rf,   _ = roc_curve(y_true, y_prob_rf)
    fpr_lstm, tpr_lstm, _ = roc_curve(y_true, y_prob_lstm)

    auc_rf   = roc_auc_score(y_true, y_prob_rf)
    auc_lstm = roc_auc_score(y_true, y_prob_lstm)

    plt.figure(figsize=(8, 5))
    plt.plot(fpr_rf,   tpr_rf,   label=f"Random Forest (AUC={auc_rf:.3f})",  lw=2)
    plt.plot(fpr_lstm, tpr_lstm, label=f"LSTM         (AUC={auc_lstm:.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — RF vs LSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.show()
    print("[INFO] ROC curve saved to outputs/roc_curve.png")


def plot_precision_recall(y_true, y_prob_rf, y_prob_lstm) -> None:
    """
    Precision-Recall curve — more informative than ROC for imbalanced classes.
    """
    prec_rf,   rec_rf,   _ = precision_recall_curve(y_true, y_prob_rf)
    prec_lstm, rec_lstm, _ = precision_recall_curve(y_true, y_prob_lstm)

    plt.figure(figsize=(8, 5))
    plt.plot(rec_rf,   prec_rf,   label="Random Forest", lw=2)
    plt.plot(rec_lstm, prec_lstm, label="LSTM",          lw=2)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — RF vs LSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/pr_curve.png", dpi=150)
    plt.show()
    print("[INFO] PR curve saved to outputs/pr_curve.png")


def plot_training_history(history) -> None:
    """
    Plots LSTM training vs validation loss and accuracy.
    Helps visually confirm no overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train Acc")
    axes[1].plot(history.history["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("outputs/lstm_training.png", dpi=150)
    plt.show()
    print("[INFO] Training history saved to outputs/lstm_training.png")


# ── 3. Backtesting ────────────────────────────────────────────────────────────

def backtest(df_test: pd.DataFrame, y_pred: np.ndarray,
             initial_capital: float = 10000.0,
             model_name: str = "Model") -> pd.DataFrame:
    """
    Simple long-only backtest:
    - If model predicts UP  → buy (hold for 1 day)
    - If model predicts DOWN → stay in cash

    Compares strategy returns vs Buy-and-Hold benchmark.

    Args:
        df_test:         Test portion of the dataframe (must have 'Close' column)
        y_pred:          Model predictions (0 or 1) aligned with df_test
        initial_capital: Starting capital in USD
        model_name:      Label for the plot

    Returns:
        results dataframe with cumulative returns
    """
    df = df_test.copy().iloc[-len(y_pred):]   # align lengths
    df["Prediction"]    = y_pred
    df["Daily_Return"]  = df["Close"].pct_change().fillna(0)

    # Strategy: only take return on days we predicted UP
    df["Strategy_Return"] = df["Daily_Return"] * df["Prediction"]

    # Cumulative returns
    df["BuyHold_Cum"]  = (1 + df["Daily_Return"]).cumprod() * initial_capital
    df["Strategy_Cum"] = (1 + df["Strategy_Return"]).cumprod() * initial_capital

    # Summary stats
    total_trades   = df["Prediction"].sum()
    final_bh       = df["BuyHold_Cum"].iloc[-1]
    final_strategy = df["Strategy_Cum"].iloc[-1]

    print(f"\n--- Backtest Results: {model_name} ---")
    print(f"Initial Capital  : ${initial_capital:,.2f}")
    print(f"Buy & Hold Final : ${final_bh:,.2f}  ({((final_bh/initial_capital)-1)*100:.2f}%)")
    print(f"Strategy Final   : ${final_strategy:,.2f}  ({((final_strategy/initial_capital)-1)*100:.2f}%)")
    print(f"Total Trades     : {int(total_trades)}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["BuyHold_Cum"],  label="Buy & Hold", lw=2, linestyle="--")
    plt.plot(df.index, df["Strategy_Cum"], label=f"{model_name} Strategy", lw=2)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.title(f"Backtest: {model_name} vs Buy & Hold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"outputs/backtest_{model_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    return df


# ── 4. Model Comparison ───────────────────────────────────────────────────────

def compare_models(results: list) -> None:
    """
    Takes a list of dicts from full_evaluation() and prints a comparison table.
    Usage: compare_models([rf_results, lstm_results])
    """
    print("\n--- Model Comparison ---")
    print(f"{'Model':<20} {'ROC-AUC':<12}")
    print("-" * 32)
    for r in results:
        print(f"{r['model']:<20} {r['roc_auc']:<12.4f}")