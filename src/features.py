import pandas as pd 
from ta.trend import MACD, EMAIndicator , SMAIndicator
from ta.momentum import RSIIndicator , StochasticOscillator
from ta.volatility import BollingerBands , AverageTrueRange

def add_technical_indicators(dataset:pd.DataFrame) -> pd.DataFrame:
    # Adding technical indicators asfeatures to the OHLCV dataset
    dataset = dataset.copy()

    close = dataset["Close"]
    high = dataset["High"]
    low = dataset["Low"]

    # Trend 
    dataset["EMA_20"] = EMAIndicator(close,window=20).ema_indicator()
    dataset["SMA_50"] = SMAIndicator(close,window=20).sma_indicator()

    macd = MACD(close)
    dataset["MACD"] = macd.macd()
    dataset["MACD_Signal"] = macd.macd_signal()
    dataset["MACD_diff"] = macd.macd_diff()

    #Momentum 
    dataset["RSI_14"] = RSIIndicator(close,window=14).rsi()

    stoch = StochasticOscillator(high,low,close)
    dataset["Stoch_K"] = stoch.stoch()
    dataset["Stoch_D"] = stoch.stoch_signal()

    # Volatility 
    bb = BollingerBands(close)
    dataset["BB_High"]  = bb.bollinger_hband()
    dataset["BB_Low"]   = bb.bollinger_lband()
    dataset["BB_Width"] = bb.bollinger_wband()

    dataset["ATR_14"]  = AverageTrueRange(high,low,close,window=14).average_true_range()

    # Feature Engineering
    dataset["Price_Change"] = dataset["Close"].diff()
    dataset["Return"] = dataset["Close"].pct_change()
    dataset["Vol_Change"] = dataset["Volume"].pct_change()

    # Interaction Features
    dataset["RSI_x_Vol"] = dataset["RSI_14"] * dataset["Vol_Change"]
    dataset["BB_Width_x_Return"] = dataset["BB_Width"] * dataset["Return"]

    # Lag Features
    for lag in [1,2,3,5]:
        dataset[f"Close_Lag_{lag}"] = dataset["Close"].shift(lag)
        dataset[f"Return_Lag_{lag}"] = dataset["Return"].shift(lag)

    # Rolling Features
    for window in [5,10,20]:
        dataset[f"SMA_{window}_Rolling"] = dataset["Close"].rolling(window=window).mean()
        dataset[f"Std_{window}_Rolling"] = dataset["Close"].rolling(window=window).std()

    # Drop initial rows with NaN values
    dataset.dropna(inplace=True)

    return dataset

def get_feature_columns(dataset: pd.DataFrame) -> list:
    """Returns all columns except the Target to be used as features."""
    return [col for col in dataset.columns if col != "Target"]
