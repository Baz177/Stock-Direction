from polygon import RESTClient
from polygon import RESTClient
from datetime import datetime, timedelta
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get('API_KEY')
client = RESTClient(api_key)


def get_aggregates(ticker):
    # --- Fetch the data ---
    multiplier = 1  # 1 unit of time
    timespan = "day" # daily bars
    from_date = ((datetime.now() + timedelta(days=1)) - timedelta(days=730)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')# End date
    api_key = os.environ.get('API_KEY')
    client = RESTClient(api_key)

    try:
        # The list_aggs method returns an iterator, so we convert it to a list
        aggs = []
        for a in client.list_aggs(ticker=ticker,
                              multiplier=multiplier,
                              timespan=timespan,
                              from_=from_date, # 'from' is a Python keyword, so use 'from_'
                              to=to_date,
                              limit=50000 # Max limit for aggregates
                             ):
            aggs.append(a)
            None

        if aggs:
            # We need to extract the relevant fields: timestamp (t), open (o), high (h), low (l), close (c), volume (v), and transactions (n)
            data = []
            for agg in aggs:
                # Polygon timestamps are in milliseconds, convert to seconds for datetime
                dt_object = datetime.fromtimestamp(agg.timestamp / 1000)
                data.append({
                    'Date': dt_object,
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume,
                    'transaction': agg.transactions
                })

            df = pd.DataFrame(data)
            #df.set_index('Date', inplace=True) # Set Date as index
            df.sort_index(inplace=True) # Ensure chronological order

            print(f"Successfully fetched {len(df)} daily bars for {ticker}.")
            print(df.tail()) # Display the first few rows of the DataFrame
            return df
        else:
            print(f"No data found for {ticker} between {from_date} and {to_date}.")
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def ticker_details(ticker):
    # --- Fetch the data ---
    try:
        ticker_details = client.get_ticker_details(ticker=ticker)
        details = {} 
        if ticker_details:
            details['name'] = ticker_details.name
            details['description'] = ticker_details.description
            details['homepage_url'] = ticker_details.homepage_url
            details['market_cap'] = ticker_details.market_cap
            details['primary_exchange'] = ticker_details.primary_exchange

            print(f"Ticker details for {ticker}:")
            print(f"Name: {details['name']}")   
            print(f"Description: {details['description']}")
            print(f"Homepage URL: {details['homepage_url']}")
            print(f"Market Cap: {details['market_cap']}")
        else:
            print(f"No details found for {ticker}.")
    except Exception as e:
        print(f"Error getting ticker details: {e}")

    return details

print(ticker_details('AAPL'))

