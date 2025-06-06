from dotenv import load_dotenv
import os 
import requests

load_dotenv()

def get_company_name(ticker):
    api_key = os.environ.get('ALPHA_ADVANTAGE_API_KEY')
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "Name" in data:
            return data["Name"]
        else:
            return f"No company found for ticker {ticker}"
    except requests.RequestException as e:
        return f"Error fetching data: {e}"

# Example usage
#ticker = "AAPL"
#print(get_company_name(ticker))