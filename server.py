import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt
from datetime import datetime
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
from flask import Flask, render_template, request
from sklearn.metrics import accuracy_score
from waitress import serve
import os
from flask import Flask, render_template, request
import matplotlib.ticker as mticker 
import warnings
from finbert_function import sentiment_function, analyze_sentiment_finbert, company_name
from alpha_advantage import get_company_name 
warnings.filterwarnings("ignore")

app = Flask(__name__)

def is_valid_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 day of historical data to check if ticker exists
        hist = stock.history(period='1d', interval='1d')
        print(hist)
        if hist.empty:
            return False
        return True
    except Exception:
        return False

def get_ipo_date(ticker):
    try:
        # Create a Ticker object
        ticker = yf.Ticker(ticker)
        
        # Fetch historical data with the maximum period
        history = ticker.history(period="max")
        
        # Check if data is available
        if history.empty:
            return False
        
        # Get the earliest date (first row's index)
        ipo_date = history.index[0]
        
        return ipo_date
    
    except Exception as e:
        return f"Error fetching data for {ticker}: {str(e)}"

def fetch_and_preprocess_data(ticker):
    today = dt.datetime.now() + dt.timedelta(days=1)
    end_date = today.strftime("%Y-%m-%d")
    start_date = (today - dt.timedelta(days=5*365)).strftime("%Y-%m-%d")
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    raw_data = yf.download(ticker, start=start_date, end=end_date, interval='1d')
    dates = raw_data.index.strftime('%Y-%m-%d').tolist()


    df = pd.DataFrame(raw_data.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    
    df.insert(0, 'Date', dates)

    df['Date'] = pd.to_datetime(df['Date'])

    # Add Vix Close
    raw_data_vix = yf.download('^VIX', start=start_date, end=end_date, interval='1d')
    df_vix = pd.DataFrame(raw_data_vix.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    df['VIX_Close'] = df_vix['Close']

    # Add S&P 500 Close
    raw_data_sp500 = yf.download('^GSPC', start=start_date, end=end_date, interval='1d')
    df_sp500 = pd.DataFrame(raw_data_sp500.values, columns=['Close', 'High', 'Low', 'Open', 'Volume'])
    df['SP500_Close'] = df_sp500['Close']

    # Add VWAP
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

    # Add indicators
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
    print(df.head())

    # plot chart for the last 30 days
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]}, # Price chart 3x taller than volume
                               figsize=(14, 9)) # Adjust figure size as needed

    # --- Plotting on the Top Subplot (ax1) - Price and Bollinger Bands ---
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    ax1.plot(df['Date'], df['BB_Upper'], label='Bollinger Upper Band', color='red', linestyle='--')
    ax1.plot(df['Date'], df['BB_Lower'], label='Bollinger Lower Band', color='green', linestyle='--')
    ax1.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], color='lightgrey', alpha=0.5)

    ax1.set_title(f'{ticker} Price with Bollinger Bands', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left') # Adjust legend location as needed
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Move Y-axis to the right for the price chart
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=10)) # Adjust number of ticks

    # --- Plotting on the Bottom Subplot (ax2) - Volume ---
    ax2.bar(df['Date'], df['Volume'], label='Volume', color='grey', alpha=0.6, width=0.8) # Adjust width
    ax2.set_ylabel('Volume')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5)) # Adjust number of ticks

    # Format the x-axis for dates
    fig.autofmt_xdate(rotation=45) # Rotate date labels for better readability

    plt.xlabel('Date') # Set a common x-label at the bottom

    # Use tight_layout to prevent labels from overlapping
    plt.tight_layout()
    # Adjust layout to make space for the right-aligned y-axis if needed
    plt.subplots_adjust(right=0.92) # Adjust based on your plot margins

    # --- Save the image ---
    # Ensure the 'static' directory exists
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig(os.path.join('static', 'bollinger_bands.png'))

    try: 
        close_price = df['Close'].iloc[-1]  # Last close price
        change = df['Close_Change'].iloc[-1]*100  # Last close change
    except IndexError:
        print("No data available.")
        pred_prob, close_price, change, last_date = 0, 0, 0, 0
        return pred_prob, close_price, change, last_date
    
    last_date = df['Date'].iloc[-1]

    df = df.dropna()
    print(df[['Date', 'Close', 'Close_Change', 'Close_Change_Lag1', 'Close_Change_Lag2', 'Direction']])

    y = df['Direction']
    X = df.drop(columns=['Date', 'Direction'])


    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.05, shuffle=False)

    model = XGBClassifier()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    model.fit(X_train_flat, y_train)
    predictions = model.predict(X_test_flat)
    accuracy = accuracy_score(y_test, predictions)
    print(f"XGBoost Directional Accuracy: {accuracy:.2%}")
    
    last_data = X.iloc[-1:].values  # Last row for prediction
    last_data_flat = last_data.reshape(1, -1)  # Flatten for XGBoost
    pred_prob = model.predict(last_data_flat)[0]  # Predict the next day direction
    tomorrow_date = last_date + dt.timedelta(days=1)
    print(f"Predicted direction for {tomorrow_date.strftime('%Y-%m-%d')}: {'Up' if pred_prob == 1 else 'Down'}")
    return pred_prob, close_price, change, last_date

@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].strip().upper()
    # Fetch company details
    # Get IPO date
    ipo_date = get_ipo_date(ticker)
    if ipo_date == False:
        return render_template('error.html', error=f"No historical data available for {ticker}")

    # Check if the IPO date is old enough
    print('ipo_date:', type(ipo_date))

    #if is_valid_ticker(ticker) == False:
        #return render_template('error.html', error="Invalid ticker symbol. Please try again.")

    # Fetch company name
    company = get_company_name(ticker)
    
    try: 
        pred_prob, close_price, change, end_date = fetch_and_preprocess_data(ticker)
        if end_date == 0: 
            return render_template('error.html', error="No data available for prediction.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return render_template('error.html', error="Error fetching data. Please try a different ticker.")

    # Make prediction
    print(f"Prediction for {ticker}: {pred_prob}")
    pred_direction = "Up" if pred_prob > 0.5 else "Down"
    if pred_prob < 0.5:
        probaility = 1 - pred_prob
    else: 
        probaility = pred_prob

    def tmr_date(end_date): 
        if end_date.day_name() == 'Friday':
            tomorrow_date = end_date + pd.Timedelta(days=3)
        else: 
            tomorrow_date = end_date + pd.Timedelta(days=1)
        return tomorrow_date

    tomorrow_date = tmr_date(end_date)
    #news_sentiment = sentiment_function(ticker)
    news_sentiment ='N/A'
    # Format the date for display  

    return render_template('result.html', 
                          ticker=ticker,
                          today = end_date.strftime('%Y-%m-%d'),
                          company_name=company,
                          change = change, 
                          direction=pred_direction,
                          probability=probaility,
                          news_sentiment = news_sentiment,
                          close_price=close_price,
                          date=tomorrow_date.strftime('%Y-%m-%d'))


if __name__ == '__main__':
    serve(app, host = '0.0.0.0', port = 8000)