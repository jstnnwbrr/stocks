import os
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tiingo import TiingoClient
from datetime import datetime, timedelta

# IMPORTANT: Ensure these are installed:
# pip install tiingo pandas numpy nltk
# Also, run nltk.download('vader_lexicon') once in your environment

# --- 1. Tiingo API Setup and Data Fetching ---

def fetch_and_analyze_news(api_key, ticker, start_date, end_date):
    """
    Fetches Tiingo news data, performs VADER sentiment analysis, and 
    aggregates the daily average sentiment score.
    """
    
    print(f"--- Fetching Tiingo News for {ticker} from {start_date} to {end_date} ---")

    # 1.1 Configure and Initialize Tiingo Client
    config = {'api_key': api_key}
    client = TiingoClient(config)

    # 1.2 Fetch News Articles
    try:
        # Fetch news articles for the specified ticker and date range
        articles = client.get_news(
            tickers=[ticker],
            startDate=start_date,
            endDate=end_date,
            limit=1000  # Adjust limit as needed
        )
        
        if not articles:
            print(f"No articles found for {ticker} in the specified range.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error fetching data from Tiingo: {e}")
        return pd.DataFrame()

    # 1.3 Convert to DataFrame and Pre-process
    news_df = pd.DataFrame(articles)
    
    # FIX: Explicitly use format='ISO8601' to handle the date string variations 
    # like "YYYY-MM-DDTHH:MM:SSZ" robustly.
    news_df['date'] = pd.to_datetime(news_df['publishedDate'], format='ISO8601').dt.date
    
    # We will analyze the sentiment of the article 'title' and 'description'
    news_df['text_to_analyze'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
    print(news_df[['date', 'title', 'description']].head())
    news_df.to_csv('fetched_news.csv', index=False)

    print(f"Successfully fetched {len(news_df)} articles.")

    # --- 2. Sentiment Analysis with VADER ---
    
    # 2.1 Initialize VADER Analyzer
    try:
        # Check if lexicon is downloaded, download if necessary
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK VADER lexicon...")
        nltk.download('vader_lexicon')
        
    sid = SentimentIntensityAnalyzer()

    # 2.2 Define the Sentiment Scoring Function
    def get_vader_score(text):
        """Returns the compound score from VADER analysis."""
        return sid.polarity_scores(text)['compound']

    # 2.3 Apply the scoring function to the text
    news_df['sentiment_score'] = news_df['text_to_analyze'].apply(get_vader_score)
    
    # --- 3. Aggregate Daily Sentiment ---
    
    # Aggregate sentiment by date, taking the mean of all articles published that day.
    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Avg_Sentiment']
    
    # Convert 'Date' column to datetime objects for easy merging later
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])

    return daily_sentiment

def fetch_stock_prices(api_key, ticker, start_date, end_date):
    """
    Fetches historical End-of-Day (EOD) stock prices using Tiingo. 
    Renames Tiingo's lowercase columns to the traditional capitalized format.
    """
    print(f"--- Fetching Market Data for {ticker} from Tiingo ---")
    
    config = {'api_key': api_key}
    client = TiingoClient(config)

    try:
        # Fetch EOD data (frequency='daily') as a Pandas DataFrame
        stock_df = client.get_dataframe(
            ticker, 
            frequency='daily', 
            startDate=start_date, 
            endDate=end_date
        )
        
        # Tiingo's get_dataframe returns 'date' as the index. Reset index.
        stock_df.reset_index(inplace=True)
        
        # --- FIX: Rename all Tiingo lowercase columns to expected capitalized format ---
        stock_df.rename(columns={
            'date': 'Date',          
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'adjOpen': 'Adj Open',
            'adjHigh': 'Adj High',
            'adjLow': 'Adj Low',
            'adjClose': 'Adj Close', 
            'volume': 'Volume',
            'adjVolume': 'Adj Volume',
        }, inplace=True)

        # Ensure the 'Date' column is converted to timezone-naive datetime objects
        # This is the fix for the TZ-aware error.
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)

        # Return the essential columns. Now, 'Open', 'High', etc., are guaranteed to exist.
        return stock_df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    except Exception as e:
        # Catches API connection errors or other unexpected issues
        print(f"Error fetching stock prices from Tiingo: {e}")
        return pd.DataFrame()

# --- 4. Main Execution and Merging ---

if __name__ == '__main__':
    # --- Configuration ---
    # !! IMPORTANT: Replace this with your actual Tiingo API Key !!
    TIINGO_API_KEY = os.getenv('TIINGO_API_KEY', None) 

    TICKER = 'AAPL'
    DAYS_TO_FETCH = 90
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=DAYS_TO_FETCH)).strftime('%Y-%m-%d')
    # ---------------------

    # 1. Get News Sentiment Feature
    sentiment_df = fetch_and_analyze_news(TIINGO_API_KEY, TICKER, START_DATE, END_DATE)

    if sentiment_df.empty:
        print("Sentiment analysis failed or returned no data. Exiting.")
    else:
        # Check for data sparsity which causes the all-zero issue
        if len(sentiment_df) < (DAYS_TO_FETCH * 0.1): # If less than 10% of days have sentiment
             print(f"\n! WARNING: Only {len(sentiment_df)} unique days of sentiment data were returned by Tiingo.")
             print("This may be due to API limits. Proceeding with imputation.")

        # 2. Get Historical Price Data (EOD prices) from Tiingo
        price_df = fetch_stock_prices(TIINGO_API_KEY, TICKER, START_DATE, END_DATE)

        if price_df.empty:
             print("Price data fetch failed or returned no data. Exiting.")
        else:
            # 3. Merge the DataFrames
            # Align the daily sentiment score with the corresponding day's OHLC data.
            final_df = pd.merge(price_df, sentiment_df, on='Date', how='left')
            
            # --- 4. Handle Missing Sentiment (Days with no news or non-trading days) ---
            
            # 4a. Forward Fill (FFILL): Carry the last known sentiment score forward to the next trading days.
            # This handles gaps *between* known data points (e.g., weekends).
            final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].fillna(method='ffill')
            
            # 4b. Anchor Check: If the very first value is still NaN, set it to 0.0.
            # This creates a neutral starting point for the subsequent linear interpolation.
            if final_df['Avg_Sentiment'].isna().iloc[0]:
                final_df.loc[0, 'Avg_Sentiment'] = 0.0
            
            # 4c. Linear Interpolation: Fill any remaining gaps (including the ramp-up from the 0.0 anchor) 
            # by linearly interpolating between two known points.
            final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].interpolate(method='linear')
            
            # 4d. Final Fallback: Fill any remaining NaNs (if the entire column was empty or a single leading NaN wasn't caught) with 0.0.
            final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].fillna(0.0)

            # Display the result
            print("\n--- FINAL FEATURE SET (Ready for Modeling) ---")
            print(f"Total rows in final DataFrame (trading days in {DAYS_TO_FETCH} calendar days): {len(final_df)}")
            print(final_df[['Date', 'Adj Close', 'Volume', 'Avg_Sentiment']])
            
            # Save to CSV
            final_df.to_csv(f'{TICKER}_sentiment_features.csv', index=False)
            print(f"\nData saved to {TICKER}_sentiment_features.csv")

