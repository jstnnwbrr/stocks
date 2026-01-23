import streamlit as st
import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
import requests
import time
import warnings
import yfinance as yf
import io
import re
import nltk
import torch

from chronos import Chronos2Pipeline
from datetime import timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, pacf
from tiingo import TiingoClient

################### BEGIN CHRONOS 2 SNIPPET #####################
time_col = 'Date'

@st.cache_resource
def load_pipeline():
    """Loads the Chronos-2 model into memory."""
    return Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu" if not torch.cuda.is_available() else "auto",
        dtype=torch.float32
    )

def get_forecast(pipeline, df, target_cols, horizon=45, confidence_level=0.8):
    """
    Generates forecasts using Chronos-2 via the standard predict() API.
    Handles data frequency alignment and missing values (holidays).
    """
    df = df.copy()
    
    # 0. Pre-process to ensure regular frequency (handle holidays/gaps)
    if 'Date' in df.columns:
        df = df.set_index('Date')
    
    # Force Business Day frequency and fill gaps
    # Chronos performs best on continuous series without NaN gaps in the tensor
    df = df.asfreq('B')
    df = df.ffill() 
    df = df.bfill()
    
    df = df.reset_index()  # Ensure 'Date' is a column for processing
    
    alpha = (1 - confidence_level) / 2
    quantiles = [alpha, 0.5, 1 - alpha]
    all_preds = []
    
    # 1. Generate Future Dates
    last_date = df['Date'].max()
    freq_offset = pd.tseries.offsets.BusinessDay(1)
    pred_dates = [last_date + (i + 1) * freq_offset for i in range(horizon)]
    
    # 2. Iterate over specific targets and forecast
    for target in target_cols:
        if target not in df.columns:
            continue
            
        try:
            # Create context tensor: Shape [1, 1, sequence_length]
            # We treat each target as a univariate series
            context = torch.tensor(df[target].values, dtype=torch.float32).reshape(1, 1, -1)
            
            # Predict
            forecast = pipeline.predict(
                context, 
                prediction_length=horizon, 
                limit_prediction_length=False
            )

            # Extract samples and quantiles
            samples = forecast[0].numpy()[0] # Shape [num_samples, horizon]
            q_values = np.quantile(samples, quantiles, axis=0)
            
            all_preds.append(pd.DataFrame({
                "timestamp": pred_dates,
                "target_name": target,
                str(quantiles[0]): q_values[0].flatten(),
                "0.5": q_values[1].flatten(),
                str(quantiles[2]): q_values[2].flatten()
            }))
        except Exception as e:
            st.warning(f"Forecast failed for {target}: {e}")
            continue
    
    preds = pd.concat(all_preds) if all_preds else pd.DataFrame()
    return preds, quantiles

################### END CHRONOS 2 SNIPPET #####################

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

def cleanup_date_features(df):
    df = df.copy()
    # Ensure Date index is clean and timezone-naive before feature creation
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    else:
        # If 'Date' is the index, reset it temporarily.
        df = df.reset_index(names=['Date'])
    
    return df

def create_date_features(df):
    # Ensure Date index is clean and timezone-naive before feature creation
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    else:
        # If 'Date' is the index, reset it temporarily.
        df = df.reset_index(names=['Date'])
    
    temp_df = df.copy() # Use a copy to avoid SettingWithCopyWarning
    
    temp_df['month'] = temp_df['Date'].dt.month
    temp_df['year'] = temp_df['Date'].dt.year
    temp_df['day'] = temp_df['Date'].dt.day
    temp_df['day_of_week'] = temp_df['Date'].dt.day_of_week
    temp_df['is_month_end'] = temp_df['Date'].dt.is_month_end.astype('int64')
    temp_df['is_month_start'] = temp_df['Date'].dt.is_month_start.astype('int64')
    temp_df['is_quarter_end'] = temp_df['Date'].dt.is_quarter_end.astype('int64')
    temp_df['is_quarter_start'] = temp_df['Date'].dt.is_quarter_start.astype('int64')
    
    return temp_df

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

    # Step 1: Clean split of all tokens
    tokens = re.split(r'[\s,;\t\n]+', text_data)

    # Step 2: Keep only short uppercase strings that are likely tickers (1–5 chars, all caps)
    cleaned_tickers = [
        token.strip().upper()
        for token in tokens
        if re.fullmatch(r'[A-Z]{1,5}', token.strip())  # 1-5 uppercase letters only
    ]

    # Step 3: Remove duplicates while preserving order
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

        df = pd.DataFrame(data)
        df = df.sort_values(by="volume", ascending=False)

        top_tickers = ['SPY']  # Always include SPY for market context
        top_tickers += df['ticker'].head(200).tolist()
        top_tickers = parse_and_clean_tickers(top_tickers)
        
        # Ensure that 'SPY' wasn't duplicated
        unique_seen = set()
        unique_top_tickers = []
        for ticker in top_tickers:
            if ticker not in unique_seen:
                unique_seen.add(ticker)
                unique_top_tickers.append(ticker)
        return unique_top_tickers

    except Exception as e:
        st.warning(f"Failed to fetch top active tickers: {e}")
        # Fallback default
        return ["SPY", "AAPL", "MSFT", "GOOG", "AMZN"]

@st.cache_data(ttl=3600)
def get_data(stock_name, end_date, tiingo_api_key):
    # Use Tiingo API as primary data source
    try:
        st.info(f"[{stock_name}] Sourcing data from Tiingo...")
        throttle_request()
        url = f"https://api.tiingo.com/tiingo/daily/{stock_name}/prices"
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
        params = {'startDate': '2023-01-01', 'endDate': end_date.strftime('%Y-%m-%d'), 'resampleFreq': 'daily'}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Tiingo error: {response.status_code} - {response.text}")
        
        data = response.json()
        if not data:
            raise ValueError("Tiingo returned empty data.")
            
        df = pd.DataFrame(data)
        df = df[['date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']].rename(columns={
            'date': 'Date', 'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low', 
            'adjClose': 'Close', 'adjVolume': 'Volume'})
        
        # Ensure the 'Date' column is converted to timezone-naive datetime objects
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        # Create dataframes for chronos and elastic net models
        df_c = cleanup_date_features(df)
        df_c = df_c.set_index('Date').asfreq('B').dropna()

        df_e = create_date_features(df)
        df_e = df_e.set_index('Date').asfreq('B').dropna()

        return df_c, df_e
    
    except Exception as e:
        st.warning(f"Tiingo failed for {stock_name}: {e}. Trying yfinance as backup.")
        try:
            st.info(f"[{stock_name}] Attempting to source data from yfinance...")
            df = yf.download(stock_name, start='2022-01-01', end=end_date, progress=False)
            if not df.empty:
                df = df.reset_index().rename(columns={'index': 'Date'}) # Ensure 'Date' column exists
                
                # Create dataframes for chronos and elastic net models
                df_c = cleanup_date_features(df)
                df_c = df_c.set_index('Date').asfreq('B').dropna()

                df_e = create_date_features(df)
                df_e = df_e.set_index('Date').asfreq('B').dropna()
                return df_c, df_e
            else:
                raise ValueError("yfinance returned empty data.")
        except Exception as yf_e:
            st.error(f"[{stock_name}] Attempt with yfinance failed: {yf_e}")
    
    st.error(f"[{stock_name}] All data sources failed.")
    return None, None

@st.cache_data(ttl=3600) # Cache news for 1 hour
def fetch_and_analyze_sentiment_tiingo(api_key, ticker, start_date, end_date, interval_days=45):
    """
    Fetches news, calculates daily sentiment, and identifies the most recent article.
    
    Returns: 
        (pd.DataFrame, dict): (daily_sentiment_df, most_recent_article)
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
                limit=150
            )
            all_articles.extend(articles)
        
            # Move to the next interval
            current_start = current_end + timedelta(days=1)
            current_end = min(current_start + timedelta(days=interval_days - 1), datetime.datetime.strptime(end_date, '%Y-%m-%d'))

        if not all_articles:
            st.warning(f"[{ticker}] No articles found for sentiment analysis.")
            return pd.DataFrame(), most_recent_article

    except Exception as e:
        st.error(f"[{ticker}] Error fetching Tiingo News for sentiment: {e}")
        return pd.DataFrame(), most_recent_article
    
    # Convert to DataFrame and pre-process
    news_df = pd.DataFrame(all_articles)
    
    # Remove duplicates based on 'title' and 'description'
    news_df = news_df.drop_duplicates(subset=['title', 'description'], keep='first')

    # Process the 'publishedDate' to ensure timezone awareness is handled for sorting
    news_df['publishedDate_dt'] = pd.to_datetime(news_df['publishedDate'], format='ISO8601', errors='coerce').dt.tz_localize(None)
    news_df['date'] = news_df['publishedDate_dt'].dt.date
    
    # Identify the most recent article based on timestamp
    if not news_df.empty:
        most_recent_article_row = news_df.sort_values(by='publishedDate_dt', ascending=False).iloc[0]
        most_recent_article = {
            'title': most_recent_article_row['title'],
            'description': most_recent_article_row['description'],
            'url': most_recent_article_row['url'],
            'date': most_recent_article_row['publishedDate_dt'].strftime('%Y-%m-%d %H:%M:%S'),
            'tickers': most_recent_article_row['tickers']
        }
        
    # Calculate daily sentiment
    news_df['text_to_analyze'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
    
    sid = SentimentIntensityAnalyzer()
    news_df['sentiment_score'] = news_df['text_to_analyze'].apply(lambda text: sid.polarity_scores(text)['compound'])
    
    daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
    daily_sentiment.columns = ['Date', 'Avg_Sentiment']
    
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date']).dt.tz_localize(None)

    st.success(f"[{ticker}] Found {len(daily_sentiment)} unique days of sentiment data.")

    return daily_sentiment, most_recent_article

def incorporate_sentiment(price_df, sentiment_df):
    """
    Merges price and sentiment data, then fills missing sentiment values 
    using FFILL, anchoring to 0.0, and mean interpolation.
    """
    if sentiment_df.empty:
        # Add a default, zero-filled column if no sentiment data exists
        price_df['Avg_Sentiment'] = 0.0
        return price_df

    # Merge the DataFrames (using the Date index from price_df which is trading days)
    # The sentiment scores are aligned by date. Missing trading days will be NaN.
    # Resetting index to use merge on 'Date' column
    df = price_df.reset_index()
    
    final_df = pd.merge(df, sentiment_df, on='Date', how='left')
    
    # Imputation Pipeline
    # Forward Fill (FFILL): Carry the last known sentiment score forward (handles weekends/holidays).
    final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].fillna(method='ffill')
    
    # Mean Interpolation: Fill any remaining gaps with the mean as a baseline sentiment
    avg_sentiment = final_df['Avg_Sentiment'].mean()
    final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].fillna(avg_sentiment)

    try:
        scaler = StandardScaler()
        final_df['Avg_Sentiment'] = scaler.fit_transform(final_df[['Avg_Sentiment']])
    except Exception as e:
        st.warning(f"StandardScaler failed on Avg_Sentiment: {e}. Proceeding without scaling.")

    # Final Fallback: Fill any remaining NaNs (shouldn't happen) with 0.0.
    final_df['Avg_Sentiment'] = final_df['Avg_Sentiment'].fillna(0.0)
    
    # Set Date back as index
    final_df = final_df.set_index('Date').asfreq('B').dropna()
    
    return final_df

def get_significant_lags(series, alpha=0.05, nlags=None):
    acf_values, confint_acf = acf(series, alpha=alpha, nlags=nlags)
    pacf_values, confint_pacf = pacf(series, alpha=alpha, nlags=nlags)
    significant_acf_lags = np.where(np.abs(acf_values) > confint_acf[:, 1] - acf_values)[0]
    significant_pacf_lags = np.where(np.abs(pacf_values) > confint_pacf[:, 1] - pacf_values)[0]
    return significant_acf_lags, significant_pacf_lags

def create_lagged_features(df, interpolate='bfill'):
    significant_lags_dict = {}
    features_to_lag = ['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']

    for col in features_to_lag:
        # Ensure the column exists before processing
        if col not in df.columns:
            continue 

        significant_acf, significant_pacf = get_significant_lags(df[col])
        significant_lags_dict[col] = {'acf': significant_acf, 'pacf': significant_pacf}

        for ma_lag in significant_acf:
            if ma_lag > 0:
                df[f'{col}_ma_lag{ma_lag}'] = df[col].shift(1).rolling(window=ma_lag).mean()
        for lag in significant_pacf:
            if lag > 0:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

    df.dropna(inplace=True)
    if df.isnull().values.any():
        df = df.interpolate(method=interpolate)
        
    return df, significant_lags_dict

def train_test_split(df, train_size=0.80):
    x_data, y_data = df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']), df['Close']
    split_idx = int(len(x_data) * train_size)
    x_train, x_test = x_data.iloc[:split_idx], x_data.iloc[split_idx:]
    y_train, y_test = y_data.iloc[:split_idx], y_data.iloc[split_idx:]
    return x_data, y_data, x_train, x_test, y_train, y_test

def plot_forecast(df, rolling_forecast_df, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-180:], df['Close'][-180:], label="Actual Close", color='blue')
    ax.plot(rolling_forecast_df['Date'], rolling_forecast_df['Predicted_Close'], label="Forecast", color='red')
    ax.set_title(f"Predicted Close Prices for {stock_name} (as of {datetime.date.today()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.grid(True)
    ax.legend()
    return fig

def rolling_forecast(stock_name, df, best_model, n_periods, x_data, significant_lags_dict):
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

        most_recent_sentiment = rolling_df['Avg_Sentiment'].iloc[-1]

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
                'Avg_Sentiment': [most_recent_sentiment]
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
            
            # Add SPY predictions
            if stock_name != 'SPY' and 'Close_SPY' in x_data.columns:
                spy_forecast_df = st.session_state.get('spy_forecast', pd.DataFrame())
                if not spy_forecast_df.empty:
                    spy_next_close = spy_forecast_df[spy_forecast_df['Date'] == new_date]['Predicted_Close']
                    if not spy_next_close.empty:
                        new_row_lags['Close_SPY'] = spy_next_close.values[0]

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
        return rolling_predictions, rolling_df

    except Exception as e:
        st.error(f"An error occurred during rolling forecast: {e}")
        return [], df

def finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods, rolling_df=None, spy_open_direction=None):
    # Use a default fallback of the last known close price
    last_close = df['Close'].iloc[-1] if not df.empty else 1.0 

    # If the rolling_predictions array is empty (length 0), we skip DataFrame creation
    # to avoid the 'ValueError: All arrays must be of the same length'.
    if not isinstance(rolling_predictions, (list, np.ndarray)) or len(rolling_predictions) == 0:
        st.warning(
            f"⚠️ Skipping forecast finalization for {stock_name}. The 'rolling_predictions' array is empty. "
            f"This often occurs when required exogenous data (like news articles for sentiment) "
            f"cannot be found, causing the prediction step to fail gracefully."
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
    rolling_forecast_df = pd.DataFrame({
        'Date': pd.date_range(start=df.index[-1], periods=n_periods + 1, freq='B')[1:],
        'Predicted_Close': rolling_predictions})

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

    # Extract predicted Open/High/Low for next (first forecasted) day if available in rolling_df
    predicted_next_open = predicted_next_high = predicted_next_low = predicted_next_avg_sentiment = None
    predicted_next_open_is_none = True

    if rolling_df is not None and not rolling_df.empty:
        try:
            base_last_date = df.index[-1]
            # Find the first row in rolling_df that is strictly after the last real data date
            future_rows = rolling_df[rolling_df.index > base_last_date]
            if not future_rows.empty:
                next_row = future_rows.iloc[0]

                predicted_next_open = max(round(float(next_row.get('Open', last_close)), 2), 0.01) if pd.notna(next_row.get('Open')) else last_close
                predicted_next_high = max(round(float(next_row.get('High', last_close)), 2), 0.01) if pd.notna(next_row.get('High')) else last_close
                predicted_next_low = max(round(float(next_row.get('Low', last_close)), 2), 0.01) if pd.notna(next_row.get('Low')) else last_close
                predicted_next_avg_sentiment = round(float(next_row.get('Avg_Sentiment', 0.0)), 2) if pd.notna(next_row.get('Avg_Sentiment', 0.0)) else 0.0

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
    stop_loss_price = round(last_close * 0.9, 2)
    target_return_price = last_close
    predicted_return = 0.0
    short_term_direction = 'flat'
    long_term_direction = 'flat'
    short_term_recommendation = 'avoid/sell'
    long_term_recommendation = 'avoid/sell'

    # --- Start of Complex Trade Target Calculation ---

    if not predicted_next_open_is_none:

        # Check if the ticker being evaluated is SPY - we'll use this as a proxy for short-term market conditions and whether to weight slightly more bullish or bearish
        if stock_name == 'SPY' and predicted_next_open is not None and df['Close'].iloc[-1] is not None:
            if predicted_next_open > df['Close'].iloc[-1]:
                spy_open_direction = 'up'
            elif predicted_next_open < df['Close'].iloc[-1]:
                spy_open_direction = 'down'
            else:
                spy_open_direction = 'flat'
            # Store SPY history and predictions in session state
            st.session_state['spy_open_direction'] = spy_open_direction
            st.session_state['spy_history'] = df[['Close']].copy()
            st.session_state['spy_forecast'] = rolling_forecast_df.copy()

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
            if spy_open_direction == 'up' and predicted_next_open != 0.01 and predicted_next_open > last_close:
                target_buy_price = round(0.75 * predicted_next_open + 0.25 * predicted_next_low, 2)
            elif spy_open_direction == 'down' and predicted_next_open != 0.01 and predicted_next_open < last_close:
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
        DIP_TOLERANCE = 0.02  # 2% tolerance; can tweak as needed or expose to UI
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

# --- Main App UI ---
st.title("📈 Stock Price Forecasting Tool")

# Get API key from environment variable
tiingo_api_key = os.getenv("TIINGO_API_KEY")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Ticker Input")
    st.info("App pre-populates the top 200 most active stocks, but feel free to paste your own!")
    
    if tiingo_api_key:
        default_tickers = get_top_200_active_tickers(tiingo_api_key)
        default_stocks = ", ".join(default_tickers)
    else:
        default_stocks = "AAPL, MSFT, GOOG, AMZN"

    stock_list_str = st.text_area("Paste Stock Tickers Here", default_stocks, height=150, help="Paste a list of tickers...")
    do_not_buy_list_str = st.text_area("Do Not Buy List (Optional)", "AST, BIEI, CGC, CGBS, CRON, LDTC, LLC, MJNA, MSOS, NXP, PET, PLC, PLTD, QID, SIX, SLGC, SMCE, SRM, TLRY, WLGS", height=100, help="Tickers you do not wish to buy...")
    
    st.subheader("Forecasting Parameters")
    n_periods = st.slider("Forecast Horizon (days)", 10, 100, 45)
    save_forecasts_to_excel = st.checkbox("Save forecasts per ticker to Excel", value=False)

# --- Main App Logic ---
if st.button("🚀 Run Forecast"):
    stock_list = []
    
    # --- Process inputs ---
    try:
        if stock_list_str:
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
                    df_c, df_e = get_data(stock_name, end_date, tiingo_api_key)
                
                if df_c is None:
                    st.error(f"Could not retrieve data for {stock_name}. Skipping.")
                    continue
                
                MIN_HISTORY_REQUIRED = 326
                if len(df_c) < MIN_HISTORY_REQUIRED:
                    st.warning(f"{stock_name} has only {len(df_c)} historical records. Skipping due to insufficient data.")
                    continue

                # Create features for modeling
                with st.spinner(f"Creating features for {stock_name}..."):
                    # Use the earliest date in the price data as the actual start date for sentiment lookup
                    earliest_price_date = df_c.index.min().date() 
                    sentiment_df, most_recent_article = fetch_and_analyze_sentiment_tiingo(
                        tiingo_api_key, 
                        stock_name, 
                        earliest_price_date.strftime('%Y-%m-%d'), # Start sentiment search from the beginning of price data
                        today.strftime('%Y-%m-%d')
                    )
                    
                    # Objects for chronos modeling process
                    df_c = incorporate_sentiment(df_c, sentiment_df)
                    
                    # Objects for elastic net modeling process
                    df_e = incorporate_sentiment(df_e, sentiment_df)
                    significant_lags_dict = {}
                    df_e, significant_lags_dict = create_lagged_features(df_e, interpolate='bfill')

                    # Ensure the Avg_Sentiment column exists before splitting
                    if 'Avg_Sentiment' not in df_c.columns:
                        df_c['Avg_Sentiment'] = 0.0

                    if 'Avg_Sentiment' not in df_e.columns:
                        df_e['Avg_Sentiment'] = 0.0

                    # Add 'SPY' close price history as a feature for all stocks except 'SPY' itself
                    if stock_name != 'SPY':
                        df_c = df_c.merge(st.session_state['spy_history'], left_index=True, right_index=True, how='left', suffixes=('', '_SPY'))
                        df_c = df_c.interpolate(method='linear').bfill().ffill()

                        df_e = df_e.merge(st.session_state['spy_history'], left_index=True, right_index=True, how='left', suffixes=('', '_SPY'))
                        df_e = df_e.interpolate(method='linear').bfill().ffill()

                    # Perform train test split for elastic net model training
                    x_data, y_data, x_train, x_test, y_train, y_test = train_test_split(df_e)

                    # If df_c is empty after dropping raw features/lag creation, something went wrong
                    if df_c.empty:
                        st.error(f"Feature creation failed for {stock_name}. Skipping.")
                        continue

                    # If x_data is empty after dropping raw features/lag creation, something went wrong
                    if x_data.empty:
                        st.error(f"Feature creation failed for {stock_name}. Skipping.")
                        continue

                # --- Model Training ---
                st.subheader(f"Model Training & Optimization for {stock_name}")
                model_scores = {}

                # Objective functions for Optuna
                def objective_elastic_net(trial):
                    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
                    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                    model.fit(x_train, y_train)
                    return mean_absolute_error(y_test, model.predict(x_test))

                studies = {'ElasticNet': optuna.create_study(direction='minimize')}
                objectives = {'ElasticNet': objective_elastic_net}

                best_model_for_stock = None
                best_mae_for_stock = float('inf')
                best_model_name_for_stock = ""
                
                for model_name, study in studies.items():
                    with st.spinner(f"Optimizing model..."):
                        pruner = optuna.pruners.MedianPruner()
                        study.optimize(objectives[model_name], n_trials=20)
                        
                        best_params = study.best_params
                        model = ElasticNet(**best_params)

                        model.fit(x_train, y_train)
                        mae = study.best_value
                        model_scores[model_name] = (model, mae)

                        if mae < best_mae_for_stock:
                            best_mae_for_stock = mae
                            best_model_for_stock = model
                            best_model_name_for_stock = model_name

                        y_pred = model.predict(x_test)
                
                # --- Forecast ---
                st.subheader(f"Forecast for {stock_name}")
                with st.spinner(f"Running forecast..."):
                    
                    ###### Sub-process for chronos model ######
                    pipeline = load_pipeline()
                    # Explicitly selecting only the core columns and potential SPY merge column
                    target_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'Avg_Sentiment']
                    if 'Close_SPY' in df_c.columns:
                        target_cols.append('Close_SPY')
                    pred_df_c, quantiles = get_forecast(pipeline, df_c, target_cols=target_cols, horizon=n_periods)
                    pred_df_c = pred_df_c.reset_index().rename(columns={'timestamp': 'Date'})[['Date', 'target_name', '0.5']]
                    pred_df_c = pred_df_c.pivot(index='Date', columns='target_name', values='0.5')
                    rolling_predictions_c, rolling_df_c = pred_df_c['Close'].tolist(), pred_df_c
                    rolling_forecast_df_c, summary_df_c = finalize_forecast_and_metrics(stock_name, rolling_predictions_c, df_c, n_periods, rolling_df_c)
                    
                    ###### Sub-process for elastic net model ######
                    X_full, y_full = df_e.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume', 'Avg_Sentiment']), df_e['Close']
                    best_model_for_stock.fit(X_full, y_full)
                    rolling_predictions_e, rolling_df_e = rolling_forecast(stock_name, df_e, best_model_for_stock, n_periods, x_data, significant_lags_dict)
                    rolling_forecast_df_e, summary_df_e = finalize_forecast_and_metrics(stock_name, rolling_predictions_e, df_e, n_periods, rolling_df_e)

                forecast_results[stock_name] = rolling_forecast_df_c
                summary_results.append(summary_df_c)

                st.write("Chronos-2")
                fig_forecast_c = plot_forecast(df_c, rolling_forecast_df_c, stock_name)
                st.pyplot(fig_forecast_c)
                st.write("Elastic Net")
                fig_forecast_e = plot_forecast(df_e, rolling_forecast_df_e, stock_name)
                st.pyplot(fig_forecast_e)

                # Display Most Recent News Article
                if most_recent_article:
                    st.subheader(f"🗞️ Most Recent News for {stock_name}")
                    st.markdown(f"**[{most_recent_article['title']}]({most_recent_article['url']})**")
                    st.markdown(f"**Published:** {most_recent_article['date']}")
                    st.markdown(f"**Tickers:** {most_recent_article['tickers']}")
                    st.caption(most_recent_article['description'])
                    st.markdown("---")

                st.dataframe(summary_df_c, use_container_width=True)
                st.dataframe(summary_df_e, use_container_width=True)

                sheet_name = re.sub(r'[\[\]\*:\?/\\ ]', '_', stock_name)[:31]
                
                if save_forecasts_to_excel:
                    rolling_forecast_df_c.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    autofit_columns(rolling_forecast_df_c, worksheet)
                
                st.markdown("---")
                time.sleep(1) # brief pause to reduce CPU spikes

            # --- Final Summary ---
            if summary_results:
                st.header("📊 Consolidated Summary")
                combined_summary = pd.concat(summary_results, ignore_index=True)
                st.dataframe(combined_summary, use_container_width=True)

                combined_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
                summary_worksheet = writer.sheets["Summary_Stats"]
                autofit_columns(combined_summary, summary_worksheet)
            
        # --- Download Button ---
        if summary_results:
            output.seek(0)
            st.download_button(
                label="📥 Download Forecasts as Excel",
                data=output,
                file_name=f"stock_forecasts_{today}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
