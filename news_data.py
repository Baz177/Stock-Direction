import yfinance as yf
import pandas as pd
from datetime import datetime
from datetime import date
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from transformers import pipeline

def stock_news(ticker):
    try:
        stock = yf.Ticker(ticker)
        news_data = stock.news
        #print("Raw news_data:", news_data)  # Debug raw response
        all_news = []
        links = []
        # Selecting data from the news_data
        for item in news_data:
            if isinstance(item, dict):
                all_news.append({
                    'Date': item['content']['displayTime'],
                    'title': item['content']['title'],
                    'summary': item['content']['summary'],
                })
                # Some items may not have 'thumbnail' key
                if 'thumbnail' in item['content']:
                    links.append(item['content']['thumbnail'])
                else: 
                    links.append('N/A')
            else:
                print(f"Item is not a dictionary: {item}")  # Debug non-dict item
        
        if not news_data:
            print(f"No news data returned for {ticker}")
            return None
        
        urls = []
        for item in links: 
            if isinstance(item, dict):
                urls.append(item[list(item.keys())[0]])
            else:
                urls.append('N/A')
        
        df = pd.DataFrame(all_news)
        df['Date'] = pd.to_datetime(df['Date']) - pd.Timedelta(hours=5)
        df.insert(1, 'links', urls)

        # Convert 'Date' to datetime and extract date and time
        df["tDate"] = df['Date'].dt.date 
        df['time'] = df['Date'].dt.time

        # Sorting the DataFrame by date
        df = df.sort_values(by="time", ascending=False).reset_index(drop=True)

        #df['sentiment_textblob'] = df['summary'].apply(lambda x: TextBlob(x).sentiment.polarity) 
        #df['sentiment'] = df['sentiment_textblob'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
        
        # using the finbert sentiment analysis model
        def analyze_sentiment_finbert(text):
            classifier = pipeline('sentiment-analysis', model='ProsusAI/finbert')
            result = classifier(text)[0]
            sentiment = result['label'].capitalize()  # e.g., 'Positive', 'Negative', 'Neutral'
            confidence = result['score']
            return sentiment, confidence
        
        sentiment_data = df['summary'].apply(lambda x: analyze_sentiment_finbert(x))

        # Unpack the tuples into separate columns
        df['finbert_sentiment'] = sentiment_data.apply(lambda x: x[0])
        df['finbert_confidence'] = sentiment_data.apply(lambda x: x[1])
        #df['finbert_confidence'] = df['summary'].apply(lambda x: analyze_sentiment_finbert(x)[1])

        sentiment_row = df[df['finbert_confidence'] == df['finbert_confidence'].max()]  # Debug sentiment counts
        sentiment = sentiment_row['finbert_sentiment'].to_string(index=False)

        try:
        # If sentiment_row['tDate'] is a string, convert pandas Timestamp or datetime object
            sentiment_date = sentiment_row['tDate'].to_string(index=False)
            time = sentiment_row['time'].to_string(index=False)
            t_date_obj = pd.to_datetime(sentiment_date).date()  
            if t_date_obj == date.today() or t_date_obj == (date.today() - pd.Timedelta(days=1)):
                print(f"The date in sentiment_row ({t_date_obj}) is either today or yesterday.")
                sentiment = sentiment_row['finbert_sentiment'].to_string(index=False)
                sentiment_title = sentiment_row['title'].to_string(index=False)
                sentiment_link = sentiment_row['links'].to_string(index=False)
            else:
                print(f"The date in sentiment_row ({t_date_obj}) is neither today nor yesterday.")
                sentiment = 'No recent News'
                sentiment_title = 'No recent News'
                sentiment_link = 'No recent News'
                t_date_obj = date.today()  # Default to today if no recent news
        except AttributeError:
        # If sentiment_row['tDate'] is a string, convert it to datetime and then get the date part
            t_date_obj = sentiment_row['tDate'].date()
        return sentiment, sentiment_title, sentiment_link, t_date_obj, time
    
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return e 






