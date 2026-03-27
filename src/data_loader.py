import yfinance as yf 
import pandas as pd 

def load_stock_data(ticker:str,start:str,end:str) -> pd.DataFrame:
    # Load available data from Yfinance
    dataset = yf.download(ticker, start=start, end=end, auto_adjust=True)

    # Flatten MultiIndex columns returned by yfinance >= 0.2
    # e.g. ('Close', 'AAPL') -> 'Close'
    if isinstance(dataset.columns, pd.MultiIndex):
        dataset.columns = dataset.columns.get_level_values(0)

    # Validation
    assert not dataset.empty, f"No data fetched for {ticker}"
    print(f"Shape of the Dataset : {dataset.shape}")
    print(f"Total Null Values:\n{dataset.isnull().sum()}")
    print(f"Data range : {dataset.index.min()} -> {dataset.index.max()}")

    # Dropping rows with null values
    dataset.dropna(inplace=True)

    return dataset

def create_target(dataset:pd.DataFrame) -> pd.DataFrame:
    #Binary target : 1 if next day's close > today's close ,else 0 . This willbe the key label for classification 
    dataset = dataset.copy()
    dataset["Target"] = (dataset["Close"].shift(-1) > dataset["Close"]).astype(int)
    dataset.dropna(inplace=True)
    return dataset

if __name__ == "__main__":
    dataset = load_stock_data("AAPL","2020-01-01","2026-03-24") 
    dataset = create_target(dataset)
    print(dataset.head())   