import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from ta.trend import EMAIndicator, MACD
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# Data loading
ticker = 'AAPL' 
today = dt.datetime.now()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - dt.timedelta(days=5*365)).strftime("%Y-%m-%d")
print(f"Loading data for {ticker} from {start_date} to {end_date}...")
raw_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
dates = raw_data.index.strftime('%Y-%m-%d').tolist()

df = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
df.insert(0, 'Date', dates)
print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

def add_VIX(df):
    """Adds VIX Close price to the DataFrame."""
    today = dt.datetime.now()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - dt.timedelta(days=5*365)).strftime("%Y-%m-%d")
    raw_data = yf.download('^VIX', start=start_date, end=end_date, interval='1d')
    dates = raw_data.index.strftime('%Y-%m-%d').tolist()
    df_vix = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    df_vix.insert(0, 'Date', dates)
    df['VIX_Close'] = df_vix['Close']
    return df

def add_sp500_close(df): 
    """Adds S&P 500 Close price to the DataFrame."""
    today = dt.datetime.now()
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - dt.timedelta(days=5*365)).strftime("%Y-%m-%d")
    raw_data = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
    dates = raw_data.index.strftime('%Y-%m-%d').tolist()
    df_sp500 = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    df_sp500.insert(0, 'Date', dates)
    df['SP500_Close'] = df_sp500['Close']
    return df

def add_VWAP(df):
    """Calculates the Volume Weighted Average Price (VWAP)."""

    required_cols = ['High', 'Low', 'Close', 'Volume', 'Date']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain columns: {required_cols}")
        return pd.DataFrame()  # Return an empty DataFrame to signal an error
    # Calculate the typical price
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    # Calculate the product of typical price and volume
    price_volume = typical_price * df['Volume']
    # Calculate the cumulative sum of price_volume and volume
    cumulative_price_volume = price_volume.cumsum()
    cumulative_volume = df['Volume'].cumsum()
    # Calculate VWAP
    df['VWAP'] = cumulative_price_volume / cumulative_volume
    return df

# Add indicators
def add_indicators(df):
    """Adds technical indicators to the DataFrame."""

    # Close Change
    df['Close_Change'] = df['Close'].pct_change()

    # Exponential Moving Averages (EMA)
    df['EMA_5'] = EMAIndicator(df['Close'], window=5).ema_indicator()
    df['EMA_9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
    df['EMA_20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['EMA_50'] = EMAIndicator(df['Close'], window=50).ema_indicator()
    df['EMA_100'] = EMAIndicator(df['Close'], window=100).ema_indicator()
    # RSI 
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # Moving Average Convergence Divergence (MACD)
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    
    # Lagged Close Change
    df['Close_Change_Lag1'] = df['Close_Change'].shift(1)
    df['Close_Change_Lag2'] = df['Close_Change'].shift(2)
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # Average True Range (ATR)
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    
    # On-Balance Volume (OBV)
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # Direction (1 for up, 0 for down)
    df['Direction'] = (df['Close_Change'] > 0).astype(int)
    
    # Handle NaN values (e.g., due to indicator windows)
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

df = add_VIX(df)
df = add_sp500_close(df)
df = add_VWAP(df)
df = add_indicators(df)

# Drop NaN (due to indicator windows)
df = df.dropna()

print(df.tail())

y = df['Direction']
X = df.drop(columns=['Date', 'Direction'])

X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, shuffle=False)

model = XGBClassifier()
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
model.fit(X_train_flat, y_train)
predictions = model.predict(X_test_flat)
accuracy = accuracy_score(y_test, predictions)
print(f"XGBoost Directional Accuracy: {accuracy:.2%}")
#joblib.dump(model, 'model.pkl')

# heatmap for XGBoost
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(cm, index=['Down', 'Up'], columns=['Down', 'Up'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# plot chart for the last 30 days
plt.figure(figsize=(14, 7)) 
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.plot(df['Date'], df['BB_Upper'], label='Bollinger Upper Band', color='red', linestyle='--') 
plt.plot(df['Date'], df['BB_Lower'], label='Bollinger Lower Band', color='green', linestyle='--')
plt.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], color='lightgrey', alpha=0.5)
plt.title(f'{ticker} Price with Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# save image of chart 
plt.savefig('static'/'bollinger_bands.png')

# Extract the last row of features (same as X)
last_data = X.iloc[-1:].values  # Last row as a 2D array
print(f"Last data for prediction: {last_data}")
last_data_flat = last_data.reshape(1, -1)  # Flatten for XGBoost

# Make prediction
pred_prob = model.predict_proba(last_data_flat)[0, 1]  # Probability of class 1 (Up)
pred_direction = "Up" if pred_prob > 0.5 else "Down"
tomorrow_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
print(f"Predicted Direction for {tomorrow_date.strftime('%Y-%m-%d')}: {pred_direction} (Probability: {pred_prob*100:.2f}%)")