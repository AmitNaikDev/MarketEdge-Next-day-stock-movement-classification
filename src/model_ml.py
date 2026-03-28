import pandas as pd 
import numpy as np 
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import TimeSeriesSplit , GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 

def build_ml_pipeline() -> Pipeline:
    # Building a sklearn Pipeline 
    pipeline = Pipeline([
        ("scaler",StandardScaler()),
        ("model",RandomForestClassifier(n_estimators=200,max_depth=10,class_weight="balanced",random_state=42,n_jobs=-1))
    ])
    return pipeline 

def time_series_split(dataset: pd.DataFrame,feature_cols:list,test_size:float = 0.2):
    # Splits data respecting time order - No random shuffling
    split_idx = int(len(dataset)*(1-test_size))

    X = dataset[feature_cols]
    Y = dataset["Target"]

    X_train,X_test  = X.iloc[:split_idx],X.iloc[split_idx:]
    Y_train,Y_test = Y.iloc[:split_idx],Y.iloc[split_idx:]

    print(f"Train Shape : {X_train.shape}")
    print(f"Test Shape : {X_test.shape}")
    print(f"Train Target Distribution : \n{Y_train.value_counts()}")
    print(f"Test Target Distribution : \n{Y_test.value_counts()}")

    return X_train,X_test,Y_train,Y_test 

def tune_hyperparameters(X_train,Y_train) -> Pipeline:
    # Hyperparameter tuning using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "model__n_estimators" : [100,200],
        "model__max_depth"  :  [5,10,None],
        "model__min_samples_split" : [2,5]
    }

    pipeline = build_ml_pipeline()

    search = GridSearchCV(pipeline,param_grid,cv=tscv,scoring="f1",n_jobs=-1,verbose=1)
    search.fit(X_train, Y_train)

    print(f"[INFO] Best params: {search.best_params_}")
    print(f"[INFO] Best CV F1:  {search.best_score_:.4f}")

    return search.best_estimator_

def evaluate_ml_model(model,X_test,Y_test) -> None:
    Y_pred = model.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(Y_test, Y_pred, target_names=["Down", "Up"]))

    print("--- Confusion Matrix ---")
    print(confusion_matrix(Y_test, Y_pred))

import os

def save_model(model, path: str = "models/rf_model.pkl") -> None:
    # Make path absolute relative to the project root (StockClassifier directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_path = os.path.join(project_root, path)
    
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    joblib.dump(model, abs_path)
    print(f"[INFO] Model saved to {abs_path}")

if __name__ == "__main__":
    import sys
    import os
    # Allow running directly via VS Code play button
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import load_stock_data, create_target
    from src.features import add_technical_indicators, get_feature_columns

    df = load_stock_data("AAPL", "2018-01-01", "2024-01-01")
    df = create_target(df)
    df = add_technical_indicators(df)

    feature_cols = get_feature_columns(df)

    X_train, X_test, y_train, y_test = time_series_split(df, feature_cols)
    model = tune_hyperparameters(X_train, y_train)
    evaluate_ml_model(model, X_test, y_test)
    save_model(model)

    