# Stock Price Prediction and Sentiment Analysis
## Overview

This repository contains a Flask web application that predicts the next day's stock price direction (Up or Down) for a given stock ticker using historical price data and technical indicators, and performs sentiment analysis on recent news articles. The application uses the yfinance library to fetch stock data, the ta library for technical indicators, XGBoost for price direction prediction, and FinBERT for news sentiment analysis. The results are displayed via a web interface, including a Bollinger Bands plot for the stock's recent price history.

## Features
- Stock Price Prediction: Uses XGBoost to predict whether a stock's price will go up or down the next trading day based on historical price data and technical indicators.
- Technical Indicators: Includes EMA, RSI, MACD, Bollinger Bands, ATR, OBV, VIX, S&P 500, VWAP, and lagged price changes.
- News Sentiment Analysis: Fetches recent news articles using yfinance and applies FinBERT to classify their sentiment as Positive, Negative, or Neutral.
- Web Interface: Built with Flask, allowing users to input a stock ticker and view predictions, sentiment analysis, and a Bollinger Bands chart.
- Data Visualization: Generates and displays a Bollinger Bands plot with volume for the stock's recent price history.
- IPO Date Check: Validates ticker and retrieves IPO date to ensure sufficient historical data.

## Prerequisites
- Python 3.8 or higher
- Git
- A GitHub account (for cloning and contributing)
- A stable internet connection for fetching stock and news data

## Installation
1. Clone the Repository:

        git clone https://github.com/your-username/your-repo.git
        cd your-repo
2. Set Up a Virtual Environment (recommended):

        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies:

        pip install -r requirements.txt
   The requirements.txt file includes:

        pandas>=2.0.0
        numpy>=1.24.0
        matplotlib>=3.7.0
        yfinance>=0.2.40
        ta>=0.10.2
        seaborn>=0.12.0
        scikit-learn>=1.2.0
        xgboost>=2.0.0
        flask>=2.2.0
        waitress>=2.1.0
        transformers>=4.30.0
        torch>=2.0.0

4. Directory Setup: Ensure a static directory exists in the project root to store generated plots:

        mkdir static
5. Template Files: Create a templates directory and ensure it contains index.html, result.html, and error.html for the Flask web interface. Example templates can be found in the Templates section below.

## Usage
1. Run the Application: Start the Flask server using Waitress:

        python app.py

    The application will be available at http://localhost:8000.
   
2. Access the Web Interface:
   - Open a web browser and navigate to http://localhost:8000.
   - Enter a valid stock ticker (e.g., AAPL for Apple Inc.).
   - Submit the form to view:
     - Predicted price direction (Up or Down) for the next trading day.
     - Probability of the prediction.
     - Recent closing price and percentage change.
     - News sentiment analysis (Positive, Negative, Neutral).
     - A Bollinger Bands plot saved in the static directory.
       
3. Example Code: The core functionality is implemented in app.py. Key components include:
    - sentiment_function: Fetches and processes news data (from sentiment_function.py).
    - fetch_and_preprocess_data: Downloads stock data, computes technical indicators, trains an XGBoost model, and generates a Bollinger Bands plot.
    - predict: Handles web form submissions and renders results.

    Example of running the prediction manually:

        from app import fetch_and_preprocess_data, sentiment_function
        ticker = "AAPL"
        pred_prob, close_price, change, last_date = fetch_and_preprocess_data(ticker)
        news_sentiment = sentiment_function(ticker)
        print(f"Prediction: {'Up' if pred_prob > 0.5 else 'Down'}")
        print(f"Close Price: {close_price:.2f}, Change: {change:.2%}")
        print(f"News Sentiment:\n{news_sentiment}")

## File Structure
  - srever.py: Main Flask application script with data fetching, preprocessing, prediction, and web routes.
  - sentiment_function.py: Script for fetching and processing news data with sentiment analysis.
  - requirements.txt: List of Python dependencies.
  - static/: Directory for storing generated plots (e.g., bollinger_bands.png).
  - templates/: Directory containing HTML templates (index.html, result.html, error.html).
  - README.md: This documentation file.



## Console:

    Loading data for AAPL from 2020-06-06 to 2025-06-06...
    XGBoost Directional Accuracy: 78.45%
    Predicted direction for 2025-06-07: Up

## Contributing
1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a Pull Request.

## Notes
  - Data Availability: The application requires sufficient historical data (at least 100 days) for reliable predictions. Invalid tickers or recent IPOs may return errors.
  - Performance: The XGBoost model is trained on 95% of the data and tested on the last 5%, with no shuffling to preserve time-series order.
  - Sentiment Analysis: Requires a stable internet connection for fetching news data and a GPU (optional) for faster FinBERT processing.
  - Plot Storage: Plots are saved in the static directory and overwritten for each new ticker.

## Contact

For questions or suggestions, open an issue or contact bazdiozt@gmail.com.
