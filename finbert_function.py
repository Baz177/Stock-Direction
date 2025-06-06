from transformers import pipeline
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from finvizfinance.quote import finvizfinance


def analyze_sentiment_finbert(text):
    classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
    result = classifier(text)[0]
    sentiment = result['label'].capitalize()  # e.g., 'Positive', 'Negative', 'Neutral'
    confidence = result['score']
    return sentiment, confidence

def sentiment_function(ticker: str) -> pd.DataFrame:
    """
    Fetches and processes news data for a given stock ticker.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
        pd.DataFrame: DataFrame containing news data with columns for date, links, title, summary, and time.
                      Returns empty DataFrame with correct columns if no data is available.

    Raises:
        ValueError: If ticker is empty or invalid.
        Exception: For other unexpected errors during data retrieval or processing.
    """
    # Input validation
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")

    try:
        # Initialize yfinance Ticker object
        stock = yf.Ticker(ticker.strip().upper())
        news_data = stock.news

        # Return empty DataFrame if no news data
        if not news_data:
            return f"There haven't been any news within the last 24hrs"

        # Process news data
        all_news = []
        links = []
        
        for item in news_data:
            # Verify item is a dictionary with expected structure
            if not isinstance(item, dict) or 'content' not in item:
                continue
                
            content = item['content']
            all_news.append({
                'Date': content.get('displayTime'),
                'title': content.get('title', ''),
                'summary': content.get('summary', '')
            })
            # Handle thumbnail links safely
            links.append(content.get('thumbnail', 'N/A'))

        # Process thumbnail links
        urls = []
        for link in links:
            if isinstance(link, dict) and link:
                urls.append(list(link.values())[0])
            else:
                urls.append('N/A')

        # Create and format DataFrame
        df = pd.DataFrame(all_news)
        if df.empty:
            return f"There haven't been any news within the last 24hrs"

        # Convert and adjust dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date'] = df['Date'] - timedelta(hours=5)
        
        # Extract date and time components
        df['tDate'] = df['Date'].dt.date
        df['time'] = df['Date'].dt.time
        
        # Add links column
        df.insert(1, 'links', urls)
        
        # Sort by time (most recent first) and reset index
        df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        print(df)

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        df_today = df[df['tDate'].isin([today, yesterday])]
        if df_today.empty: 
            return f'There havent been any news within the last 24hrs' 
        df_today['finbert_sentiment'] = df_today['summary'].apply(lambda x: analyze_sentiment_finbert(x)[0])
        df_today['finbert_confidence'] = df_today['summary'].apply(lambda x: analyze_sentiment_finbert(x)[1])
        neg_sentiment = df_today[df_today['finbert_sentiment'] == 'Negative']['finbert_confidence'].sum()
        pos_sentiment = df_today[df_today['finbert_sentiment'] == 'Positive']['finbert_confidence'].sum()
        print(df_today)
        if neg_sentiment > pos_sentiment: 
            return f'Generally News have been Negative over the last 24hrs'
        else: 
            return f'Generally News have Positive over the last 24hrs'
    except Exception as e:
        # Log error and return empty DataFrame with correct structure
        print(f"Error processing data for {ticker}: {str(e)}")
        return f"There haven't been any news within the last 24hrs"

#print(sentiment_function('MBOT'))

def company_name(ticker):
    stock = finvizfinance(ticker)
    stock_fundament = stock.ticker_fundament()
    company = stock_fundament['Company']
    return company

#print(company_name('TSLA'))    
