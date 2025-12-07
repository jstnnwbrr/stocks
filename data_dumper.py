# data_dumper.py
# Run this script periodically (e.g., via cron job) to update your local data cache.
# Pre-requisites: Set the TIINGO_API_KEY environment variable.
# pip install pandas tiingo yfinance requests nltk

import os
import pandas as pd
import requests
import warnings
import time
import nltk

from datetime import datetime, timedelta
from tiingo import TiingoClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Configuration ---
# List of tickers for which you want to maintain a historical data cache.
TICKER_LIST = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "JPM", "V", "JNJ", "WMT"]

# Local file paths for the cache
PRICES_CACHE_FILE = "historical_prices.csv"
NEWS_CACHE_FILE = "news_sentiment.csv"

# Tiingo API setup
TIINGO_API_KEY = os.environ.get("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError("TIINGO_API_KEY environment variable not set. Cannot proceed.")

CONFIG = {'session': True, 'api_key': TIINGO_API_KEY}
CLIENT = TiingoClient(CONFIG)

# NLTK setup for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')
VADER = SentimentIntensityAnalyzer()

# --- Helper Functions ---

def load_or_create_cache(file_path, index_col=None, dtype=None):
    """Loads existing data from a CSV file or returns an empty DataFrame."""
    if os.path.exists(file_path):
        try:
            # Check file size before reading
            if os.path.getsize(file_path) > 0:
                print(f"Loading existing data from {file_path}...")
                df = pd.read_csv(file_path, index_col=index_col, parse_dates=True, dtype=dtype)
                if index_col:
                    df = df.sort_index()
                return df
            else:
                print(f"Cache file {file_path} is empty. Creating new cache.")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error loading {file_path}: {e}. Starting with an empty cache.")
            return pd.DataFrame()
    return pd.DataFrame()

def determine_start_date(df, ticker, default_days_ago=730):
    """Determines the last recorded date for a ticker to fetch new data."""
    today = datetime.now().date()
    
    # NEW: Check if required columns are present in the loaded DataFrame
    required_cols = ['ticker', 'date']
    if not all(col in df.columns for col in required_cols):
        # If essential columns are missing (which caused the KeyError),
        # treat it as no existing cache data and fetch the default history.
        start = today - timedelta(days=default_days_ago)
        print(f"  > WARNING: Cache file structure is incomplete (missing {', '.join([col for col in required_cols if col not in df.columns])}). Fetching full history ({default_days_ago} days) from {start}.")
        return start.strftime('%Y-%m-%d')

    # Ensure 'date' is present in the DataFrame for comparison
    # Added .copy() to avoid potential SettingWithCopyWarning
    df_filtered = df[df['ticker'] == ticker].copy()
    
    if not df_filtered.empty:
        # Convert date column to datetime objects if it's not already
        df_filtered['date'] = pd.to_datetime(df_filtered['date']).dt.date
        last_date = df_filtered['date'].max()
        
        # Start fetching from the day after the last recorded date
        start = last_date + timedelta(days=1)
        print(f"  > Found data up to {last_date}. Fetching from {start}.")
        return start.strftime('%Y-%m-%d')
    
    # If no data or date column is missing, fetch the default history
    start = today - timedelta(days=default_days_ago)
    print(f"  > No existing cache data for {ticker}. Fetching full history ({default_days_ago} days) from {start}.")
    return start.strftime('%Y-%m-%d')

# --- Data Fetching Logic (Modified from App) ---

def fetch_historical_prices(ticker, start_date, end_date):
    """Fetches historical price data from Tiingo."""
    print(f"  > Fetching price data for {ticker} from {start_date} to {end_date}...")
    try:
        # Fetch daily data
        data = CLIENT.get_dataframe(ticker, startDate=start_date, endDate=end_date, metric_name='close', frequency='daily')
        
        if data.empty:
            print(f"  > No new price data found for {ticker} in range.")
            return pd.DataFrame()
        
        # Tiingo returns 'close', 'ticker' columns. Rename index to 'date'.
        data = data.reset_index().rename(columns={'index': 'date'})
        data['date'] = data['date'].dt.strftime('%Y-%m-%d')
        print(f"  > Successfully fetched {len(data)} new price records.")
        return data
    except Exception as e:
        print(f"  > Error fetching price data for {ticker}: {e}")
        return pd.DataFrame()

def fetch_news_and_sentiment(ticker, start_date, end_date, interval_days=30):
    print(f"[{ticker}] Fetching news sentiment (Tiingo News)...")

    all_news_data = []

    # Split the time range into intervals
    # We will assume that if it's a string, we parse it, otherwise, we convert it.
    if isinstance(start_date, str):
        current_start = datetime.strptime(start_date, '%Y-%m-%d')
    else: # Handle datetime.date objects by combining them with a time component (00:00:00)
        current_start = datetime.combine(start_date, datetime.time())

    # The end_date also needs to be converted if it's a string
    if isinstance(end_date, str):
        parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d')
    else:
        parsed_end_date = datetime.combine(end_date, datetime.time())

    current_end = min(current_start + timedelta(days=interval_days - 1), parsed_end_date)

    CHUNK_SIZE_DAYS = interval_days

    while current_start <= current_end: 
        # Determine the end of the current chunk
        chunk_end = min(current_start + timedelta(days=CHUNK_SIZE_DAYS - 1), current_end)
        
        chunk_start_str = current_start.strftime('%Y-%m-%d')
        chunk_end_str = chunk_end.strftime('%Y-%m-%d')

        try:
            # Fetch news articles for the current interval
            news_chunk = CLIENT.get_news(
                tickers=[ticker],
                startDate=chunk_start_str,
                endDate=chunk_end_str,
                limit=1000
            )
            all_news_data.extend(news_chunk)

        except Exception as e:
            # Print error but continue to the next chunk
            print(f"      -> ERROR fetching chunk {chunk_start_str} to {chunk_end_str}: {e}")
            
        # Move the start date to the day after the chunk ended
        current_start = chunk_end + timedelta(days=1)
        time.sleep(0.5) # API courtesy pause

    if not all_news_data:
        print(f"  > No new news articles found for {ticker} in range.")
        return pd.DataFrame()

    articles = []
    for article in all_news_data:
        # Perform sentiment analysis on the article's text
        sentiment = VADER.polarity_scores(article.get('title', '') + ' ' + article.get('description', ''))
        
        # The Tiingo news API often returns the ticker in the 'tags' list, not directly as 'ticker'.
        # We safely get the first tag, falling back to the requested ticker if tags are missing.
        article_ticker = article.get('tags', [ticker])[0] if article.get('tags') else ticker

        articles.append({
            'date': pd.to_datetime(article['publishedDate']).strftime('%Y-%m-%d'),
            'ticker': article_ticker, # Using the fixed ticker retrieval
            'title': article['title'],
            'description': article['description'],
            'source': article['source'],
            'Avg_Sentiment': sentiment['compound']
        })

    news_df = pd.DataFrame(articles)
    
    # Aggregate sentiment by day (use the mean compound score)
    sentiment_summary = news_df.groupby(['date', 'ticker'])['Avg_Sentiment'].mean().reset_index()
    sentiment_summary['num_articles'] = news_df.groupby(['date', 'ticker']).size().reset_index(name='num_articles')['num_articles']
    
    print(f"  > Successfully fetched and summarized {len(all_news_data)} news articles into {len(sentiment_summary)} daily sentiment records.")
    return sentiment_summary

# --- Main Execution ---

def run_data_dumper():
    """Main function to run the data fetching process."""
    print(f"Starting data dumping process for {len(TICKER_LIST)} tickers...")
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Load existing cache files
    # Note: Using 'object' for date/ticker columns to prevent mixed type warnings
    prices_cache = load_or_create_cache(PRICES_CACHE_FILE, dtype={'ticker': 'object', 'date': 'object'})
    news_cache = load_or_create_cache(NEWS_CACHE_FILE, dtype={'ticker': 'object', 'date': 'object'})
    
    # --- FIX: Ensure caches have required column schemas if they were loaded empty or corrupted ---
    # This prevents KeyError during concatenation and drop_duplicates later.
    PRICE_COLS = ['date', 'ticker', 'close'] 
    NEWS_COLS = ['date', 'ticker', 'Avg_Sentiment', 'num_articles']

    if not all(col in prices_cache.columns for col in PRICE_COLS):
        # Initialize an empty DataFrame with the required column structure
        prices_cache = pd.DataFrame(columns=PRICE_COLS)

    if not all(col in news_cache.columns for col in NEWS_COLS):
        # Initialize an empty DataFrame with the required column structure
        news_cache = pd.DataFrame(columns=NEWS_COLS)
    # -----------------------------------------------------------------------------------------

    for ticker in TICKER_LIST:
        new_prices_data = []
        new_news_data = []

        print(f"\nProcessing Ticker: {ticker}")
        
        # 1. Determine start date for price data
        prices_start_date = determine_start_date(prices_cache, ticker, default_days_ago=1825) # 5 years
        
        # 2. Determine start date for news data
        # News is often sparse, so we might want to check for a slightly longer history if missing
        news_start_date = determine_start_date(news_cache, ticker, default_days_ago=1825) # 5 years
        
        # 3. Fetch and collect new data
        price_df = fetch_historical_prices(ticker, prices_start_date, end_date)
        if not price_df.empty:
            new_prices_data.append(price_df)
        
        news_df = fetch_news_and_sentiment(ticker, news_start_date, end_date)
        if not news_df.empty:
            new_news_data.append(news_df)
        
        time.sleep(0.5) # Be kind to the API

    # --- Combine and Save ---
    
    # 4. Save Prices
    if new_prices_data:
        new_prices_df = pd.concat(new_prices_data, ignore_index=True)

        # Ensure 'date' is in the correct format before concatenation
        new_prices_df['date'] = pd.to_datetime(new_prices_df['date']).dt.strftime('%Y-%m-%d')

        # prices_cache now guaranteed to have the correct columns, preventing KeyError
        final_prices_df = pd.concat([prices_cache, new_prices_df], ignore_index=True)

        # Remove duplicates (in case of partial overlap from start_date logic)
        final_prices_df = final_prices_df.drop_duplicates(subset=['date', 'ticker'], keep='last')
        
        # Organize the order of columns
        final_prices_df = final_prices_df[['date', 'ticker', 'close']]
        
        final_prices_df.to_csv(PRICES_CACHE_FILE, index=False)
        print(f"\n✅ Price cache updated! Total records: {len(final_prices_df)}")
    else:
        print("\nSkipping price cache update: No new price data fetched.")

    # 5. Save News
    if new_news_data:
        new_news_df = pd.concat(new_news_data, ignore_index=True)

        # Ensure 'date' is in the correct format before concatenation
        print(new_news_df)
        new_news_df['date'] = pd.to_datetime(new_news_df['date']).dt.strftime('%Y-%m-%d')
        
        # news_cache now guaranteed to have the correct columns, preventing KeyError
        final_news_df = pd.concat([news_cache, new_news_df], ignore_index=True)

        # Remove duplicates
        final_news_df = final_news_df.drop_duplicates(subset=['date', 'ticker'], keep='last').sort_values(by='date')
        
        # Organize the order of columns
        final_news_df = final_news_df[NEWS_COLS + ['Avg_Sentiment']]


        final_news_df.to_csv(NEWS_CACHE_FILE, index=False)
        print(f"✅ News cache updated! Total records: {len(final_news_df)}")
    else:
        print("Skipping news cache update: No new news data fetched.")

    print("\nData dumping process completed.")

if __name__ == "__main__":
    run_data_dumper()