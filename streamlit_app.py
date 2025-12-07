# forecasting_app.py
# To run this app, save it as a python file (e.g., app.py) and run: streamlit run app.py
# Make sure to set the TIINGO_API_KEY environment variable before running.
# Make sure to install all necessary libraries:
# pip install streamlit pandas numpy scikit-learn statsmodels optuna yfinance requests matplotlib xlsxwriter openpyxl

import streamlit as st
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import polars as pl
import pandas as pd
import random
import requests
import statsmodels.api as sm
import time
import warnings
import yfinance as yf
import io
import re
import nltk

from datetime import timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tiingo import TiingoClient
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, pacf

# --- Initial Setup ---
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
st.set_page_config(layout="wide", page_title="Stock Forecasting App")

# Download VADER lexicon on startup (if not already present)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    st.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

# --- Helper Functions from Original Script ---

# Throttling for API requests
MAX_REQUESTS_PER_HOUR = 10000
requests_made = 0
start_time = datetime.datetime.now()

def throttle_request():
    global requests_made, start_time
    requests_made += 1
    if requests_made > MAX_REQUESTS_PER_HOUR:
        time_elapsed = datetime.datetime.now() - start_time
        if time_elapsed.total_seconds() < 3600:
            wait_time = 3600 - time_elapsed.total_seconds()
            st.warning(f"Rate limit likely exceeded. Sleeping for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
        requests_made = 1
        start_time = datetime.datetime.now()

def create_date_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Creates date-based features using Polars expressions.
    Assumes 'Date' column exists and is of type Date or Datetime.
    """
    # Polars dt.weekday() is 1 (Mon) - 7 (Sun). Pandas is 0-6.
    # We subtract 1 to maintain compatibility with the original logic if needed, though strictly not required for trees/regression.
    
    ###KEEP JUST IN CASE###
    # Ensure Date index is clean and timezone-naive before feature creation
    #if 'Date' in df.columns:
    #    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    #else:
    #    # If 'Date' is the index, reset it temporarily.
    #    df = df.reset_index(names=['Date'])
    
    return df.with_columns([
        pl.col("Date").dt.month().alias("month"),
        pl.col("Date").dt.year().alias("year"),
        pl.col("Date").dt.day().alias("day"),
        (pl.col("Date").dt.weekday() - 1).alias("day_of_week"),
        
        # Approximate is_month_end/start logic
        (pl.col("Date").dt.day() == 1).cast(pl.Int64).alias("is_month_start"),
        
        # For month end, we check if the next day is day 1
        (pl.col("Date").dt.offset_by("1d").dt.day() == 1).cast(pl.Int64).alias("is_month_end"),
        
        # Quarter calculations
        (pl.col("Date").dt.quarter().is_in([1, 2, 3, 4]) & (pl.col("Date").dt.day() == 1) & (pl.col("Date").dt.month().is_in([1, 4, 7, 10]))).cast(pl.Int64).alias("is_quarter_start"),
        
        # Quarter end (simple approximation for standard quarters)
        (pl.col("Date").dt.offset_by("1d").dt.day() == 1).cast(pl.Int64).alias("is_quarter_end_check") 
        # Note: accurate quarter end is complex without a calendar, but this suffices for ML features usually
    ]).drop("is_quarter_end_check") # Cleanup temp column if needed

# --- Function to parse and clean ticker inputs ---
def parse_and_clean_tickers(input_data):
    """
    Parses messy or clean pasted text into a clean list of stock tickers.
    Filters out financial figures, numbers, and non-ticker entries.
    """
    if isinstance(input_data, list):
        text_data = ' '.join(map(str, input_data))
    else:
        text_data = str(input_data)

    # Clean split of all tokens
    tokens = re.split(r'[\s,;\t\n]+', text_data)

    # Keep only short uppercase strings that are likely tickers (1–5 chars, all caps)
    cleaned_tickers = [
        token.strip().upper()
        for token in tokens
        if re.fullmatch(r'[A-Z]{1,5}', token.strip())  # 1-5 uppercase letters only
    ]

    # Remove duplicates while preserving order
    seen = set()
    unique_tickers = []
    for ticker in cleaned_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)

    return unique_tickers

@st.cache_data(ttl=3600) # Cache data for 1 hour
def get_top_200_active_tickers(tiingo_api_key):
    url = "https://api.tiingo.com/iex"
    headers = {
        "Authorization": f"Token {tiingo_api_key}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        df = pl.DataFrame(data)
        
        # Sort by volume descending and take top 200
        # Ensure 'volume' is numeric
        if df.height > 0:
            df = df.with_columns(pl.col("volume").cast(pl.Float64))
            df = df.sort("volume", descending=True).head(200)
            
            top_tickers = ['SPY'] + df['ticker'].to_list()
            return parse_and_clean_tickers(top_tickers)
        else:
             return ["SPY", "AAPL", "MSFT", "GOOG", "AMZN"]

    except Exception as e:
        st.warning(f"Failed to fetch top active tickers: {e}")
        return ["SPY", "AAPL", "MSFT", "GOOG", "AMZN"]

def fill_missing_business_days(df: pl.DataFrame, start_date, end_date) -> pl.DataFrame:
    """
    Simulates Pandas .asfreq('B').
    Generates a full range of business days and joins the data to it.
    """
    # 1. Generate full date range
    # Polars date_range is eager by default in recent versions or via 'eager=True'
    full_dates = pl.date_range(
        start=start_date,
        end=end_date,
        interval="1d",
        eager=True
    ).to_frame("Date")
    
    # 2. Filter for Weekdays (Monday=1 to Friday=5)
    business_dates = full_dates.filter(pl.col("Date").dt.weekday() <= 5)
    
    # 3. Join original data
    # Ensure original df has Date as just Date (not datetime with time) for join
    df = df.with_columns(pl.col("Date").cast(pl.Date))
    
    # Perform Left Join to keep all business days, introducing Nulls where data is missing
    merged = business_dates.join(df, on="Date", how="left")
    
    # 4. Drop rows where data is still missing (equivalent to .dropna() after asfreq)
    # The original script does .dropna(), which implies we only keep days that actually have data
    # OR perfectly filled days. If the intention of asfreq('B') was to insert rows to fill them,
    # we would need ffill. The original script does .asfreq('B').dropna(), which paradoxically
    # removes the gaps created unless they were filled. 
    # However, strictly speaking, asfreq('B') aligns the index.
    
    merged = merged.drop_nulls()
    return merged

@st.cache_data(ttl=3600)
def get_data(stock_name, end_date, tiingo_api_key):
    # Use Tiingo API as primary data source
    try:
        st.info(f"[{stock_name}] Sourcing data from Tiingo...")
        throttle_request()
        url = f"https://api.tiingo.com/tiingo/daily/{stock_name}/prices"
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
        params = {'startDate': '2019-01-01', 'endDate': end_date.strftime('%Y-%m-%d'), 'resampleFreq': 'daily'}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Tiingo error: {response.status_code} - {response.text}")
        
        data = response.json()
        if not data:
            raise ValueError("Tiingo returned empty data.")
            
        # Polars Load
        df = pl.DataFrame(data)
        
        # Select and Rename
        df = df.select([
            pl.col('date').alias('Date'),
            pl.col('adjOpen').alias('Open'),
            pl.col('adjHigh').alias('High'),
            pl.col('adjLow').alias('Low'),
            pl.col('adjClose').alias('Close'),
            pl.col('adjVolume').alias('Volume')
        ])
        
        # Date Conversion
        df = df.with_columns(
            pl.col('Date').str.to_datetime(time_zone='UTC')
            .dt.replace_time_zone(None)
            .cast(pl.Date)
        )
        
        # Fill business days (aligning to 'B' frequency)
        df = fill_missing_business_days(df, datetime.date(2019, 1, 1), end_date.date())
        
        # Create Features
        df = create_date_features(df)
        
        return df
    
    except Exception as e:
        st.warning(f"Tiingo failed for {stock_name}: {e}. Trying yfinance as backup.")
        try:
            st.info(f"[{stock_name}] Attempting to source data from yfinance...")
            pandas_df = yf.download(stock_name, start='2019-01-01', end=end_date, progress=False)
            
            if not pandas_df.empty:
                # Convert Pandas -> Polars
                pandas_df = pandas_df.reset_index()
                # Handle MultiIndex columns if present (yfinance update)
                if isinstance(pandas_df.columns, pd.MultiIndex):
                    pandas_df.columns = pandas_df.columns.get_level_values(0)
                    
                df = pl.from_pandas(pandas_df)
                
                # Normalize column names
                # yfinance usually gives Date, Open, High, Low, Close, Volume
                df = df.rename({'Date': 'Date', 'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
                
                # Standardize Date
                df = df.with_columns(pl.col('Date').cast(pl.Date))
                
                df = fill_missing_business_days(df, datetime.date(2019, 1, 1), end_date.date())
                df = create_date_features(df)
                return df
            else:
                raise ValueError("yfinance returned empty data.")
        except Exception as yf_e:
            st.error(f"[{stock_name}] Attempt with yfinance failed: {yf_e}")
    
    st.error(f"[{stock_name}] All data sources failed.")
    return None

@st.cache_data(ttl=3600) # Cache news for 1 hour
def fetch_and_analyze_sentiment_tiingo(api_key, ticker, start_date, end_date, interval_days=45):
    """
    Fetches news, calculates daily sentiment, and identifies the most recent article.
    """
    
    st.info(f"[{ticker}] Fetching news sentiment (Tiingo News)...")
    throttle_request()

    all_articles = []
    most_recent_article = None

    try:
        # Initialize Tiingo Client
        client = TiingoClient({'api_key': api_key})

        # Split the time range into intervals
        # We will assume that if it's a string, we parse it, otherwise, we convert it.
        if isinstance(start_date, str):
            current_start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        else: # Handle datetime.date objects by combining them with a time component (00:00:00)
            current_start = datetime.datetime.combine(start_date, datetime.time())

        # The end_date also needs to be converted if it's a string
        if isinstance(end_date, str):
            parsed_end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        else:
            parsed_end_date = datetime.datetime.combine(end_date, datetime.time())

        current_end = min(current_start + timedelta(days=interval_days - 1), parsed_end_date)
    
        while current_start <= parsed_end_date: 
            # Fetch news articles for the current interval
            articles = client.get_news(
                tickers=[ticker],
                startDate=current_start.strftime('%Y-%m-%d'),
                endDate=current_end.strftime('%Y-%m-%d'),
                limit=1000 
            )
            all_articles.extend(articles)
        
            # Move to the next interval
            current_start = current_end + timedelta(days=1)
            current_end = min(current_start + timedelta(days=interval_days - 1), parsed_end_date)

        if not all_articles:
            st.warning(f"[{ticker}] No articles found for sentiment analysis.")
            return pl.DataFrame(), most_recent_article

    except Exception as e:
        st.error(f"[{ticker}] Error fetching Tiingo News for sentiment: {e}")
        return pl.DataFrame(), most_recent_article
    
    # Convert to DataFrame and pre-process
    news_df = pl.DataFrame(all_articles)
    
    # Remove duplicates based on 'title' and 'description'
    news_df = news_df.unique(subset=['title', 'description'], keep='first')

    # Process Dates
    # Robust ISO 8601 parsing for news as well
    news_df = news_df.with_columns(
        pl.col('publishedDate').str.to_datetime(time_zone='UTC')
        .dt.replace_time_zone(None)
        .dt.cast_time_unit("us")
        .alias('publishedDate_dt')
    )
    
    news_df = news_df.with_columns(pl.col('publishedDate_dt').dt.date().alias('date'))
    
    # Identify most recent article
    if news_df.height > 0:
        most_recent_row = news_df.sort('publishedDate_dt', descending=True).row(0, named=True)
        most_recent_article = {
            'title': most_recent_row['title'],
            'description': most_recent_row['description'],
            'url': most_recent_row['url'],
            'date': str(most_recent_row['publishedDate_dt']),
            'tickers': most_recent_row['tickers']
        }
        
    # Analyze Sentiment
    # Concatenate title and description
    news_df = news_df.with_columns(
        (pl.col('title').fill_null('') + ' ' + pl.col('description').fill_null('')).alias('text_to_analyze')
    )
    
    sid = SentimentIntensityAnalyzer()
    
    # Polars requires mapping a python function for VADER
    def get_sentiment(text):
        return sid.polarity_scores(text)['compound']

    # Use map_elements (polars > 0.19) or apply
    news_df = news_df.with_columns(
        pl.col('text_to_analyze').map_elements(get_sentiment, return_dtype=pl.Float64).alias('sentiment_score')
    )
    
    # Group by Date and mean
    daily_sentiment = news_df.group_by('date').agg(pl.col('sentiment_score').mean().alias('Avg_Sentiment'))
    daily_sentiment = daily_sentiment.rename({'date': 'Date'})
    
    st.success(f"[{ticker}] Found {daily_sentiment.height} unique days of sentiment data.")

    return daily_sentiment, most_recent_article

def incorporate_sentiment(price_df: pl.DataFrame, sentiment_df: pl.DataFrame) -> pl.DataFrame:
    """
    Merges price and sentiment data, then fills missing sentiment values 
    using FFILL and mean interpolation.
    """

    if sentiment_df.is_empty():
        # Add a default, zero-filled column if no sentiment data exists
        return price_df.with_columns(pl.lit(0.0).alias('Avg_Sentiment'))

    # Ensure Date types match
    price_df = price_df.with_columns(pl.col('Date').cast(pl.Date))
    sentiment_df = sentiment_df.with_columns(pl.col('Date').cast(pl.Date))
    
    # Join DataFrames
    final_df = price_df.join(sentiment_df, on='Date', how='left')

    # Imputation Pipeline
    # Forward Fill then Fill Null with Mean
    final_df = final_df.with_columns(pl.col('Avg_Sentiment').fill_null(strategy='forward'))
    
    mean_sentiment = final_df.select(pl.col('Avg_Sentiment').mean()).item()
    if mean_sentiment is None: mean_sentiment = 0.0
        
    final_df = final_df.with_columns(pl.col('Avg_Sentiment').fill_null(mean_sentiment))
    
    # Scale Sentiment (Need to convert to numpy/pandas for StandardScaler or implement manual scaling)
    # Manual Scaling in Polars is easy: (x - mean) / std
    avg_col = pl.col('Avg_Sentiment')
    final_df = final_df.with_columns(
        ((avg_col - avg_col.mean()) / (avg_col.std() + 1e-9)).alias('Avg_Sentiment')
    )
    
    final_df = final_df.with_columns(pl.col('Avg_Sentiment').fill_null(0.0))
    
    return final_df

def get_significant_lags(series, alpha=0.15, nlags=None):
    # This requires statsmodels which needs numpy array
    series_np = series.to_numpy()
    acf_values, confint_acf = acf(series_np, alpha=alpha, nlags=nlags)
    pacf_values, confint_pacf = pacf(series_np, alpha=alpha, nlags=nlags)
    significant_acf_lags = np.where(np.abs(acf_values) > confint_acf[:, 1] - acf_values)[0]
    significant_pacf_lags = np.where(np.abs(pacf_values) > confint_pacf[:, 1] - pacf_values)[0]
    return significant_acf_lags, significant_pacf_lags

def create_lagged_features(df: pl.DataFrame):
    significant_lags_dict = {}
    features_to_lag = ['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']

    # Since we are modifying DF in loop, keep a list of new expressions
    expressions = []

    for col_name in features_to_lag:
        # Ensure the column exists before processing
        if col_name not in df.columns:
            continue 

        # Extract series for statsmodels
        series = df.select(pl.col(col_name))
        significant_acf, significant_pacf = get_significant_lags(series)
        significant_lags_dict[col_name] = {'acf': significant_acf, 'pacf': significant_pacf}

        for ma_lag in significant_acf:
            if ma_lag > 0:
                # Rolling mean of shifted column
                # pandas: df[col].shift(1).rolling(window=ma_lag).mean()
                expressions.append(
                    pl.col(col_name).shift(1).rolling_mean(window_size=ma_lag).alias(f"{col_name}_ma_lag{ma_lag}")
                )
        for lag in significant_pacf:
            if lag > 0:
                # Autoregressive lags
                # df[f'{col_name}_lag{lag}'] = df[col_name].shift(lag)
                expressions.append(
                    pl.col(col_name).shift(lag).alias(f"{col_name}_lag{lag}")
                )

    if expressions:
        df = df.with_columns(expressions)

    # Drop Nulls created by lags
    df = df.drop_nulls()
    
    return df, significant_lags_dict

def train_test_split(df: pl.DataFrame, train_size=0.80):
    # Convert to Pandas for sklearn compatibility eventually, 
    # but we can slice in Polars first.
    
    split_idx = int(df.height * train_size)
    
    # Split
    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx, df.height - split_idx)
    
    features_to_drop = ['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment', 'Date']
    feature_cols = [c for c in df.columns if c not in features_to_drop]
    
    # We return Pandas DataFrames or Numpy arrays for Scikit-Learn
    x_train = train_df.select(feature_cols).to_pandas()
    y_train = train_df.select('Close').to_pandas().values.ravel()
    
    x_test = test_df.select(feature_cols).to_pandas()
    y_test = test_df.select('Close').to_pandas().values.ravel()
    
    # X_data full
    x_data = df.select(feature_cols).to_pandas()
    y_data = df.select('Close').to_pandas().values.ravel()
    
    # Also return Dates for plotting
    train_dates = train_df.select('Date').to_series().to_list()
    test_dates = test_df.select('Date').to_series().to_list()

    return x_data, y_data, x_train, x_test, y_train, y_test, train_dates, test_dates

def plot_actual_vs_predicted(train_dates, y_train, test_dates, y_test, y_pred, model_name, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_dates, y_train, label="Training Data", color="blue", linewidth=1)
    ax.plot(test_dates, y_test, label="Test Data (Actuals)", color="green", linewidth=1)
    ax.plot(test_dates, y_pred, label="Predicted Test Data", color="red", linewidth=1)
    ax.legend()
    ax.set_title(f"{stock_name} - Historical Actuals vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.grid(True)
    return fig

def save_plot_forecast(original_df, forecast_df, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get last 180 days from Polars DF
    recent_hist = original_df.tail(180)
    hist_dates = recent_hist['Date'].to_list()
    hist_close = recent_hist['Close'].to_list()
    
    ax.plot(hist_dates, hist_close, label="Actual Close", color='blue')
    ax.plot(forecast_df['Date'], forecast_df['Predicted_Close'], label="Rolling Forecast", color='red')
    ax.set_title(f"Predicted Close Prices for {stock_name} (as of {datetime.date.today()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.grid(True)
    ax.legend()
    return fig

def rolling_forecast(df, best_model, n_periods, x_data, significant_lags_dict):
    df = df.to_pandas()

    try:
        # Use only the features that VAR can handle (numeric, non-lagged, non-date features)
        var_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']
        var_features = [f for f in var_features if f in df.columns]

        # --- VAR Collinearity/Singularity Fix: Remove zero-variance features ---
        # Calculate the standard deviation for the VAR features
        std_dev = df[var_features].std()
        # Filter out features where standard deviation is zero (or very close to zero, e.g., < 1e-6)
        stable_var_features = list(std_dev[std_dev > 1e-6].index)

        if len(stable_var_features) < len(var_features):
            removed_features = set(var_features) - set(stable_var_features)
            st.warning(f"VAR Collinearity Fix: Removed zero-variance features from VAR input for stability: {', '.join(removed_features)}")
        
        if not stable_var_features:
            st.error("Cannot train VAR: All features have zero variance after cleaning. Skipping forecast.")
            return [], df # Return empty predictions if all features are constant

        var_features = stable_var_features
        
        # Use the raw features for VAR
        var_model = VAR(df[var_features])
        var_fitted = var_model.fit(ic='aic')
        
        # Check if we have enough historical data for the VAR model
        if len(df) < var_fitted.k_ar:
            st.warning(
                f"Skipping {df.columns[0]}: only {len(df)} data points available, "
                f"but VAR model requires at least {var_fitted.k_ar}.")
            return [], df
        
        rolling_df = df.copy()
        rolling_predictions = []

        progress_bar = st.progress(0, text=f"Generating {n_periods}-day forecast...")

        for i in range(n_periods):
            last_date = rolling_df.index[-1]
            new_date = last_date + pd.offsets.BusinessDay(1)
            
            var_input = rolling_df[var_features].iloc[-var_fitted.k_ar:]
            
            # Catch edge case where even after initial check, slicing fails
            if var_input.shape[0] < var_fitted.k_ar:
                st.warning(f"Insufficient data for step {i+1}. Forecasting halted early.")
                break
            
            var_forecast = var_fitted.forecast(y=var_input.values, steps=1)[0]

            # Map VAR output back to feature names (use a dictionary to store all predictions)
            var_output_map = dict(zip(var_features, var_forecast))
            
            # Use var_output_map to retrieve predictions, defaulting to latest actual if a feature was removed (e.g., zero-sentiment)
            predicted_close_var = var_output_map.get('Close', rolling_df['Close'].iloc[-1])
            predicted_high = var_output_map.get('High', rolling_df['High'].iloc[-1])
            predicted_low = var_output_map.get('Low', rolling_df['Low'].iloc[-1])
            predicted_open = var_output_map.get('Open', rolling_df['Open'].iloc[-1])
            predicted_volume = var_output_map.get('Volume', rolling_df['Volume'].iloc[-1])
            predicted_avg_sentiment = var_output_map.get('Avg_Sentiment', rolling_df['Avg_Sentiment'].iloc[-1])

            next_period_raw = pd.DataFrame({
                'Close': [max(predicted_close_var, 0.01)], 
                'High': [max(predicted_high, 0.01)],
                'Low': [max(predicted_low, 0.01)], 
                'Open': [max(predicted_open, 0.01)],
                'Volume': [max(predicted_volume, 0)],
                'Avg_Sentiment': [predicted_avg_sentiment]
                }, index=[new_date])

            latest_data = pd.concat([rolling_df[['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']], next_period_raw])

            new_row_features = latest_data.copy()

            # Create lagged features for the new row based on the augmented data
            all_lags_created = {}
            for col in ['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']:
                if col not in significant_lags_dict: continue # Skip if no lags found
                
                # Re-calculate MA/Lags for the new row using the latest data (including VAR forecast)
                for lag in significant_lags_dict[col]['pacf']:
                    if lag > 0:
                        all_lags_created[f'{col}_lag{lag}'] = new_row_features[col].shift(lag).iloc[-1]
                for ma_lag in significant_lags_dict[col]['acf']:
                    if ma_lag > 0:
                        all_lags_created[f'{col}_ma_lag{ma_lag}'] = new_row_features[col].shift(1).rolling(window=ma_lag).mean().iloc[-1]
            
            # Convert created lags to a DataFrame row
            new_row_lags = pd.DataFrame([all_lags_created], index=[new_date])
            
            # Create Date Features for the new row
            new_row_lags = new_row_lags.reset_index().rename(columns={'index': 'Date'})
            new_row_lags = create_date_features(new_row_lags)
            new_row_lags = new_row_lags.set_index('Date').asfreq('B').dropna()
            
            # Prepare final input for ElasticNet (must match x_data.columns exactly)
            final_input_row = new_row_lags.reindex(columns=x_data.columns, fill_value=0.0)

            predicted_value = max(best_model.predict(final_input_row)[0], 0.01)
            rolling_predictions.append(predicted_value)

            # 6. Append the final prediction to rolling_df for the next iteration
            final_row_for_next_iteration = next_period_raw.copy()
            final_row_for_next_iteration['Close'] = predicted_value # Use the more accurate ElasticNet prediction for 'Close'
            
            # Append the full raw feature set for the next VAR step
            rolling_df = pd.concat([rolling_df, final_row_for_next_iteration])

            if i % 5 == 0 or i == n_periods -1:
                progress_bar.progress((i + 1) / n_periods, text=f"Day {i+1}/{n_periods} forecasted...")

        progress_bar.empty()
        return rolling_predictions, pl.from_pandas(rolling_df)

    except Exception as e:
        st.error(f"An error occurred during rolling forecast: {e}")
        return [], df

def finalize_forecast_and_metrics(stock_name, rolling_predictions, df: pl.DataFrame, n_periods, rolling_df=None, spy_open_direction=None):
    # Use a default fallback of the last known close price
    last_close = df['Close'][-1] if df.height > 0 else 1.0 
    st.write(f"Last Close: {last_close}")

    # If the rolling_predictions array is empty (length 0), we skip DataFrame creation
    # To avoid the 'ValueError: All arrays must be of the same length'.
    if not isinstance(rolling_predictions, (list, np.ndarray)) or len(rolling_predictions) == 0:
        st.warning(
            f"⚠️ Skipping forecast finalization for {stock_name}. The 'rolling_predictions' array is empty. "
        )
        # Return empty DataFrames with N/A summary stats
        empty_forecast_df = pd.DataFrame(columns=['Date', 'Predicted_Close'])
        
        # Initialize all target variables to safe defaults (0 or N/A) for the summary DataFrame
        nan_summary_data = {
            'ticker_symbol': [stock_name], 
            'short_term_direction': ['N/A'], 
            'short_term_recommendation': ['N/A'],
            'target_buy_price': [np.nan],
            'target_sell_price': [np.nan],
            'stop_loss_price': [np.nan],
            'short_term_predicted_return_%': [np.nan],
            'predicted_open': [np.nan],
            'predicted_high': [np.nan],
            'predicted_low': [np.nan],
            'predicted_sentiment': [np.nan],
            'long_term_direction': ['N/A'],
            'long_term_recommendation': ['N/A'],
            'long_term_sell_price': [np.nan],
            'long_term_predicted_return_%': [np.nan],
            'predicted_high_15_day': [np.nan],
            'predicted_low_15_day': [np.nan], 
            'predicted_second_lowest_15_day': [np.nan],
            'predicted_avg_15_day': [np.nan],
            'predicted_volatility_%': [np.nan]
        }
        empty_summary_df = pd.DataFrame(nan_summary_data)
        return empty_forecast_df, empty_summary_df
    
    # If predictions exist, proceed with main calculations
    # Create Forecast DF
    # Generate future dates
    last_date = df['Date'][-1]
    future_dates = []
    curr = last_date
    while len(future_dates) < n_periods:
        curr += datetime.timedelta(days=1)
        if curr.weekday() < 5:
            future_dates.append(curr)

    rolling_forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': rolling_predictions
    })

    horizon_df = rolling_forecast_df.head(15)
    predicted_avg_3_days = max(round(horizon_df['Predicted_Close'].head(3).mean(), 2), 0.01)
    predicted_high_15_days = max(round(horizon_df['Predicted_Close'].max(), 2), 0.01)
    predicted_low_15_days = max(round(horizon_df['Predicted_Close'].min(), 2), 0.01)
    
    # Compute the second-lowest predicted close within the 15-day horizon to avoid overreacting to a single deep dip
    horizon_vals = horizon_df['Predicted_Close'].dropna().values
    if len(horizon_vals) >= 2:
        sorted_vals = np.sort(horizon_vals)
        predicted_second_lowest_15_days = max(round(float(sorted_vals[1]), 2), 0.01)
    else:
        predicted_second_lowest_15_days = predicted_low_15_days
    
    predicted_avg_15_days = max(round(horizon_df['Predicted_Close'].mean(), 2), 0.01)
    predicted_volatility_15_days = round(horizon_df['Predicted_Close'].std() / predicted_avg_15_days, 3)

    # Reminder: rolling_df (not rolling_forecast_df) is Polars
    # Extract predicted Open/High/Low for next (first forecasted) day if available in rolling_df
    predicted_next_open = predicted_next_high = predicted_next_low = predicted_next_avg_sentiment = None
    predicted_next_open_is_none = True

    if rolling_df is not None and not rolling_df.height > 0:
        try:
            # Filter for dates > last_close date
            future_rows = rolling_df.filter(pl.col('Date') > last_date)
            if future_rows.height > 0:
                next_row = future_rows.row(0, named=True)
                predicted_next_open = next_row.get('Open', last_close)
                predicted_next_high = next_row.get('High', last_close)
                predicted_next_low = next_row.get('Low', last_close)
                predicted_next_avg_sentiment = next_row.get('Avg_Sentiment', 0.0)

                # Set flag if we successfully extracted values (they might still be equal to last_close, but they are not None)
                predicted_next_open_is_none = False 
            else:
                # If future_rows is empty, the values remain None, which is handled below.
                 pass
        except Exception as e:
             logging.error(f"Error extracting next-day features: {e}")
             # If extraction fails, leave values as None and rely on initialization below      

    # Ensure min/max integrity
    if predicted_next_high is not None and predicted_next_open is not None and predicted_next_low is not None:
        predicted_next_low = min(predicted_next_high, predicted_next_open, predicted_next_low)
        predicted_next_high = max(predicted_next_high, predicted_next_open, predicted_next_low)

    # --- Initialization to prevent UnboundLocalError ---
    target_buy_price = last_close
    target_sell_price = last_close
    stop_loss_price = round(last_close * 0.94, 2)
    target_return_price = last_close
    predicted_return = 0.0
    short_term_direction = 'flat'
    long_term_direction = 'flat'
    short_term_recommendation = 'avoid/sell'
    long_term_recommendation = 'avoid/sell'

    # --- Start of Complex Trade Target Calculation ---

    if not predicted_next_open_is_none:

        # Check if the ticker being evaluated is SPY - we'll use this as a proxy for short-term market conditions and whether to weight slightly more bullish or bearish
        if stock_name == 'SPY' and predicted_next_open is not None and last_close is not None:
            if predicted_next_open > last_close:
                spy_open_direction = 'up'
            elif predicted_next_open < last_close:
                spy_open_direction = 'down'
            else:
                spy_open_direction = 'flat'
            # Store SPY predicted open direction in session state
            st.session_state['spy_open_direction'] = spy_open_direction
        else:
            # Retrieve SPY recommendation from session state
            spy_open_direction = st.session_state.get('spy_open_direction', 'avoid/sell')
        
        # Default to last close price if any required prediction is None
        if predicted_next_open is None or predicted_next_low is None or predicted_next_high is None:
            target_buy_price = last_close
            target_sell_price = last_close
            target_return_price = last_close
            predicted_return = 0.0
            stop_loss_price = round(last_close * 0.94, 2)
        else:
            if spy_open_direction == 'up':
                if predicted_next_open != 0.01:
                    target_buy_price = round(0.75 * predicted_next_open + 0.25 * predicted_next_low, 2)
            elif spy_open_direction == 'down':
                target_buy_price = round(0.25 * predicted_next_open + 0.75 * predicted_next_low, 2)
            else:
                target_buy_price = round(np.mean([predicted_next_open, predicted_next_low]), 2)

            target_sell_price = round(np.mean([predicted_next_open, predicted_next_high]), 2)
            target_return_price = round(np.mean([target_sell_price, predicted_avg_3_days]), 2)
            predicted_return = ((target_return_price / target_buy_price) - 1) if target_buy_price > 0 else 0
            stop_loss_price = round(target_buy_price * 0.94, 2)

            # Catch edge case where target_buy_price is lower than predicted_next_low
            if predicted_next_low and target_buy_price < predicted_next_low:
                target_buy_price = predicted_next_low

        short_term_direction = 'flat'
        if predicted_return > 0: 
            short_term_direction = 'up' 
        elif predicted_return < 0: 
            short_term_direction = 'down'

        short_term_recommendation = 'avoid/sell'
        if short_term_direction == 'up' and predicted_return > 0.005:
            short_term_recommendation = 'buy' if predicted_volatility_15_days < 0.10 else 'hold'

        # Adjust recommendation for additional conditions
        if short_term_direction == 'up' and predicted_return > 0.005:
            # If predicted range looks wide relative to avg, prefer hold for safety
            intraday_strength = 0
            if predicted_next_high and predicted_next_low:
                intraday_strength = (predicted_next_high - predicted_next_low) / np.mean([predicted_next_open, predicted_next_low, predicted_next_high])
            short_term_recommendation = 'avoid/sell' if intraday_strength > 0.08 else 'buy'

        # Calculate long-term sell targets, predicted return, and recommendations
        long_term_sell_price = max(round((predicted_avg_15_days * (1 + (0.5 * predicted_volatility_15_days))), 2), 0.01)
        long_term_predicted_return = ((long_term_sell_price / target_buy_price) - 1) if target_buy_price > 0 else 0

        long_term_direction = 'flat'
        if horizon_df['Predicted_Close'].iloc[-1] > target_buy_price: 
            long_term_direction = 'up'
        if predicted_low_15_days < target_buy_price: 
            long_term_direction = 'down'

        # Adjust recommendation for additional conditions
        long_term_recommendation = 'avoid/sell'
        if long_term_direction == 'up' and long_term_predicted_return > 0.03:
            long_term_recommendation = 'buy' if predicted_volatility_15_days < 0.125 else 'hold'

        if long_term_direction == 'up' and predicted_return > 0.03:
            # If predicted range looks wide relative to avg, prefer hold for safety
            long_term_strength = 0
            if predicted_next_high and predicted_next_low and predicted_avg_15_days > 0:
                long_term_strength = (predicted_high_15_days - predicted_low_15_days) / predicted_avg_15_days
            long_term_recommendation = 'avoid/sell' if predicted_volatility_15_days > 0.15 or long_term_strength > 0.10 else 'buy'

        # If a dip within a certain threshold is foreseen in the 15-day horizon, avoid buying
        # Use the second-lowest value to soften responses to a single temporary dip.
        # If the second-lowest is not far below the target buy price, treat the dip as temporary and keep short-term recommendation.
        DIP_TOLERANCE = 0.02  # 2% tolerance; tweak as needed or expose to UI
        if long_term_direction == 'down':
            try:
                # If the second-lowest is significantly below the target buy price (beyond tolerance), cancel short-term buy
                if predicted_second_lowest_15_days < (target_buy_price * (1 - DIP_TOLERANCE)):
                    short_term_recommendation = 'avoid/sell'
                else:
                    # treat as temporary dip: do not override previously determined short_term_recommendation
                    pass
            except Exception:
                short_term_recommendation = 'avoid/sell'

        # If predicted return is very high (greater than 50%), likely too good to be true - avoid
        if predicted_return > 0.50:
            short_term_recommendation = 'avoid/sell'
            long_term_recommendation = 'avoid/sell'

        # If price is extremely low (penny stock), avoid buying
        if target_buy_price < 1.00:
            short_term_recommendation = 'avoid/sell'
            long_term_recommendation = 'avoid/sell'

    # Create summary_df after all calculations
    summary_df = pd.DataFrame({
        'ticker_symbol': [stock_name], 
        'short_term_direction': [short_term_direction], 
        'short_term_recommendation': [short_term_recommendation],
        'target_buy_price': [target_buy_price],
        'target_sell_price': [target_sell_price],
        'stop_loss_price': [stop_loss_price],
        'short_term_predicted_return_%': [predicted_return * 100],
        'predicted_open': [predicted_next_open],
        'predicted_high': [predicted_next_high],
        'predicted_low': [predicted_next_low],
        'predicted_sentiment': [predicted_next_avg_sentiment],
        'long_term_direction': [long_term_direction],
        'long_term_recommendation': [long_term_recommendation],
        'long_term_sell_price': [long_term_sell_price],
        'long_term_predicted_return_%': [long_term_predicted_return * 100],
        'predicted_high_15_day': [predicted_high_15_days],
        'predicted_low_15_day': [predicted_low_15_days], 
        'predicted_second_lowest_15_day': [predicted_second_lowest_15_days],
        'predicted_avg_15_day': [predicted_avg_15_days],
        'predicted_volatility_%': [predicted_volatility_15_days * 100]})

    return rolling_forecast_df, summary_df

def autofit_columns(df, worksheet):
    for i, column in enumerate(df.columns):
        column_width = max(df[column].astype(str).map(len).max(), len(column)) + 3
        worksheet.set_column(i, i, column_width)

# --- Streamlit App UI ---
st.title("📈 Stock Price Forecasting Tool")
st.markdown("""
This application uses machine learning to forecast stock prices. 
Enter stock tickers by pasting them into the text box or by uploading a file.
""")

# Get API key from environment variable
tiingo_api_key = os.getenv("TIINGO_API_KEY")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Ticker Input")
    st.info("App pre-populates the top 200 most active stocks, but feel free to paste your own or upload a file!")
    
    if tiingo_api_key:
        default_tickers = get_top_200_active_tickers(tiingo_api_key)
        default_stocks = ", ".join(default_tickers)
    else:
        default_stocks = "AAPL, MSFT, GOOG, AMZN"

    stock_list_str = st.text_area("Paste Stock Tickers Here", default_stocks, height=150, help="Paste a list of tickers...")
    do_not_buy_list_str = st.text_area("Do Not Buy List (Optional)", "AI, APLS, APPN, AU, AUR, BITF, BL, BTBT, BTCZ, BTDR, BTG, DNN, GDX, GLD, GLDM, GOOG, ICCM, IOVA, JDST, LLC, MARA, MJNA, MSTR, MSTU, MSTX, MSTZ, NGD, PLTD, PSLV, QID, QQQU, QUBT, RDDT, RIOT, SGOL, SLGC, SLV, SOUN, SOXL, SPDN, SPXU, SQQQ, SRM, TQQQ, TSDD, TSLL, TSLQ, TSLS, TSLY, TTD, TZA, ULTY, VIST, VRNS, WULF", height=100, help="Tickers you do not wish to buy...")

    uploaded_file = st.file_uploader(
        "Or Upload a File", 
        type=['txt', 'csv', 'xlsx'],
        help="Upload a .txt, .csv, or .xlsx file with one ticker per line, or in the first column."
    )
    
    st.subheader("Forecasting Parameters")
    n_periods = st.slider("Forecast Horizon (days)", 10, 100, 45)
    
    st.subheader("Model Training Parameters")
    max_trials = st.slider("Max Optimization Trials", 10, 100, 20)
    patience = st.slider("Optimization Patience (early stopping)", 5, 20, 10)
    save_forecasts_to_excel = st.checkbox("Save forecasts per ticker to Excel", value=False)

# --- Main App Logic ---
if st.button("🚀 Run Forecast"):
    stock_list = []
    
    # --- Process inputs ---
    try:
        if uploaded_file is not None:
            st.info(f"Processing uploaded file: {uploaded_file.name}")
            if uploaded_file.name.endswith('.csv'):
                df_upload = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df_upload = pd.read_excel(uploaded_file, engine='openpyxl')
            elif uploaded_file.name.endswith('.txt'):
                # For txt, we decode and let the parser handle it
                stock_list = ['SPY'] # Always include SPY for market context
                stock_list += parse_and_clean_tickers(uploaded_file.getvalue().decode("utf-8"))
                unique_stock_list = []
                unique_temp_set = set()
                for ticker in stock_list:
                    if ticker not in unique_temp_set:
                        unique_temp_set.add(ticker)
                        unique_stock_list.append(ticker)

                del ticker

                do_not_buy_list = parse_and_clean_tickers(do_not_buy_list_str) if do_not_buy_list_str else []
                do_not_buy_list = [ticker.strip().upper() for ticker in do_not_buy_list if ticker.strip()]
                stock_list = []
                stock_list = [ticker for ticker in unique_stock_list if ticker not in do_not_buy_list]
                
                del unique_stock_list, unique_temp_set


            # For csv/xlsx, assume tickers are in the first column
            if 'df_upload' in locals():
                if not df_upload.empty:
                    # Convert first column to list to be parsed
                    raw_tickers = df_upload.iloc[:, 0].tolist()
                    stock_list = ['SPY']  # Always include SPY for market context
                    stock_list += parse_and_clean_tickers(raw_tickers)
                    unique_stock_list = []
                    unique_temp_set = set()
                    for ticker in stock_list:
                        if ticker not in unique_temp_set:
                            unique_temp_set.add(ticker)
                            unique_stock_list.append(ticker)

                    del ticker

                    do_not_buy_list = parse_and_clean_tickers(do_not_buy_list_str) if do_not_buy_list_str else []
                    do_not_buy_list = [ticker.strip().upper() for ticker in do_not_buy_list if ticker.strip()]
                    stock_list = []
                    stock_list = [ticker for ticker in unique_stock_list if ticker not in do_not_buy_list]

                    del unique_stock_list, unique_temp_set
                
                else:
                    st.warning("Uploaded file is empty.")

        elif stock_list_str:
            st.info("Processing tickers from text area.")
            stock_list = ['SPY']  # Always include SPY for market context
            stock_list += parse_and_clean_tickers(stock_list_str)
            unique_stock_list = []
            unique_temp_set = set()
            for ticker in stock_list:
                if ticker not in unique_temp_set:
                    unique_temp_set.add(ticker)
                    unique_stock_list.append(ticker)

            del ticker

            do_not_buy_list = parse_and_clean_tickers(do_not_buy_list_str) if do_not_buy_list_str else []
            do_not_buy_list = [ticker.strip().upper() for ticker in do_not_buy_list if ticker.strip()]
            stock_list = []
            stock_list = [ticker for ticker in unique_stock_list if ticker not in do_not_buy_list]
    
            del unique_stock_list, unique_temp_set

    except Exception as e:
        st.error(f"An error occurred while processing inputs: {e}")

    if not tiingo_api_key:
        st.error("`TIINGO_API_KEY` environment variable not set. Please set it to your Tiingo API key.")
    elif not stock_list:
        st.error("No valid stock tickers found. Please enter tickers in the text box or upload a file.")
    else:
        st.success(f"Found {len(stock_list)} unique tickers to process: {', '.join(stock_list)}")
        st.success(f"The following {len(do_not_buy_list)} tickers were identified as Do Not Buy: {', '.join(do_not_buy_list)}")

        today = datetime.date.today()
        end_date = today + pd.offsets.BusinessDay(1)

        forecast_results = {}
        summary_results = []
        
        # Create an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            
            # --- Ensure SPY is processed first ---
            if 'SPY' in stock_list:
                stock_list.remove('SPY')
                stock_list.insert(0, 'SPY')

            # --- Main loop for processing each stock ---
            for stock_name in stock_list:
                st.header(f"Processing: {stock_name}")
                
                # Initialize news variable outside the try block
                most_recent_article = None

                # Fetch Historical Price Data
                with st.spinner(f"Fetching data for {stock_name}..."):
                    df = get_data(stock_name, end_date, tiingo_api_key)
                
                if df is None:
                    st.error(f"Could not retrieve data for {stock_name}. Skipping.")
                    continue
                
                MIN_HISTORY_REQUIRED = 326
                if df.height < MIN_HISTORY_REQUIRED:
                    st.warning(f"{stock_name} has only {len(df)} historical records. Skipping due to insufficient data.")
                    continue

                # Fetch and Incorporate Sentiment Data
                with st.spinner(f"Sentiment Analysis..."):
                    earliest = df['Date'].min()
                    sentiment_df, article = fetch_and_analyze_sentiment_tiingo(tiingo_api_key, stock_name, earliest, today)
                    df = incorporate_sentiment(df, sentiment_df)

                with st.spinner(f"Feature Engineering..."):
                    df, sig_lags = create_lagged_features(df)
                    if 'Avg_Sentiment' not in df.columns: df = df.with_columns(pl.lit(0.0).alias('Avg_Sentiment'))

                    # Split (returns numpy/pandas for sklearn)
                    x_data, y_data, x_train, x_test, y_train, y_test, train_dates, test_dates = train_test_split(df)

                    # If x_data is empty after dropping raw features/lag creation, something went wrong
                    if x_data.empty:
                        st.error(f"Feature creation failed for {stock_name} after lagging and cleaning. Skipping.")
                        continue

                # --- Model Training ---
                st.subheader(f"Model Training & Optimization for {stock_name}")
                model_scores = {}
                
                def objective(trial):
                    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
                    l1 = trial.suggest_float('l1_ratio', 0, 1)
                    model = ElasticNet(alpha=alpha, l1_ratio=l1)
                    model.fit(x_train, y_train)
                    return mean_absolute_error(y_test, model.predict(x_test))

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=max_trials)
                
                best_model = ElasticNet(**study.best_params)
                best_model.fit(x_train, y_train)
                
                mae = study.best_value
                st.write(f"Best MAE: {mae:.4f}")
                
                fig = plot_actual_vs_predicted(train_dates, y_train, test_dates, y_test, best_model.predict(x_test), "ElasticNet", stock_name)
                st.pyplot(fig)
                
                # --- Forecast ---
                st.subheader(f"Forecast for {stock_name}")
                with st.spinner(f"Forecasting..."):
                    # Retrain on full
                    best_model.fit(x_data, y_data)
                    rolling_preds, rolling_df_raw = rolling_forecast(df, best_model, n_periods, x_data, sig_lags)
                    fc_df, sum_df = finalize_forecast_and_metrics(stock_name, rolling_preds, df, n_periods, rolling_df_raw)
                
                forecast_results[stock_name] = fc_df
                summary_results.append(sum_df)

                st.pyplot(save_plot_forecast(df, fc_df, stock_name))

                # Display Most Recent News Article
                if article:
                    st.subheader(f"🗞️ Most Recent News for {stock_name}")
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.markdown(f"**Published:** {article['date']}")
                    st.markdown(f"**Tickers:** {article['tickers']}")
                    st.caption(article['description'])
                    st.markdown("---")

                st.dataframe(sum_df)

                if save_forecasts_to_excel:
                    sheet = re.sub(r'[\[\]\*:\?/\\ ]', '_', stock_name)[:31]
                    fc_df.to_excel(writer, sheet_name=sheet, index=False)
                
                st.markdown("---")

            # --- Final Summary ---
            if summary_results:
                st.header("📊 Summary")
                combined = pd.concat(summary_results, ignore_index=True)
                st.dataframe(combined)
                combined.to_excel(writer, sheet_name='Summary', index=False)

        # --- Download Button ---
        if summary_results:
            output.seek(0)
            st.download_button(
                label="📥 Download Excel",
                data=output,
                file_name=f"stock_forecasts_{today}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
