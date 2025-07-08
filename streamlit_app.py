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
import pandas as pd
import random
import requests
import statsmodels.api as sm
import time
import warnings
import yfinance as yf
import io
import re

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, pacf

# --- Initial Setup ---
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
st.set_page_config(layout="wide", page_title="Stock Forecasting App")

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

def create_date_features(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.day_of_week
    df['is_month_end'] = df['Date'].dt.is_month_end.astype('int64')
    df['is_month_start'] = df['Date'].dt.is_month_start.astype('int64')
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype('int64')
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype('int64')
    return df

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

        top_tickers = df['ticker'].head(200).tolist()
        top_tickers = parse_and_clean_tickers(top_tickers)
        return top_tickers

    except Exception as e:
        st.warning(f"Failed to fetch top active tickers: {e}")
        # Fallback default
        return ["AAPL", "MSFT", "GOOG", "AMZN"]

def get_data(stock_name, end_date, tiingo_api_key):
    # Use Tiingo API as primary data source
    try:
        st.info(f"[{stock_name}] Sourcing data from Tiingo...")
        throttle_request()
        url = f"https://api.tiingo.com/tiingo/daily/{stock_name}/prices"
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
        params = {'startDate': '2015-01-01', 'endDate': end_date.strftime('%Y-%m-%d'), 'resampleFreq': 'daily'}
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
        df = create_date_features(df)
        df = df.set_index('Date').asfreq('B').dropna()
        return df
    
    except Exception as e:
        st.warning(f"Tiingo failed for {stock_name}: {e}. Trying yfinance as backup.")
        for attempt in range(3):
            try:
                st.info(f"[{stock_name}] Attempting to source data from yfinance...")
                df = yf.download(stock_name, start='2015-01-01', end=end_date, progress=False)
                if not df.empty:
                    df = df.reset_index()
                    df = create_date_features(df)
                    df = df.set_index('Date').asfreq('B').dropna()
                    return df
                else:
                    raise ValueError("yfinance returned empty data.")
            except Exception as yf_e:
                st.error(f"[{stock_name}] Attempt {attempt + 1} with yfinance failed: {yf_e}")
                time.sleep(2)
    
    st.error(f"[{stock_name}] All data sources failed.")
    return None

def get_significant_lags(series, alpha=0.15, nlags=None):
    acf_values, confint_acf = sm.tsa.stattools.acf(series, alpha=alpha, nlags=nlags)
    pacf_values, confint_pacf = sm.tsa.stattools.pacf(series, alpha=alpha, nlags=nlags)
    significant_acf_lags = np.where(np.abs(acf_values) > confint_acf[:, 1] - acf_values)[0]
    significant_pacf_lags = np.where(np.abs(pacf_values) > confint_pacf[:, 1] - pacf_values)[0]
    return significant_acf_lags, significant_pacf_lags

def create_lagged_features(df, interpolate='bfill'):
    significant_lags_dict = {}
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
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
    x_data, y_data = df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume']), df['Close']
    split_idx = int(len(x_data) * train_size)
    x_train, x_test = x_data.iloc[:split_idx], x_data.iloc[split_idx:]
    y_train, y_test = y_data.iloc[:split_idx], y_data.iloc[split_idx:]
    return x_data, y_data, x_train, x_test, y_train, y_test

def plot_actual_vs_predicted(y_train, y_test, y_pred, model_name, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index, y_train, label="Training Data", color="blue", linewidth=1)
    ax.plot(y_test.index, y_test, label="Test Data (Actuals)", color="green", linewidth=1)
    ax.plot(y_test.index, y_pred, label="Predicted Test Data", color="red", linewidth=1)
    ax.legend()
    ax.set_title(f"{stock_name} - Historical Actuals vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.grid(True)
    return fig

def save_plot_forecast(df, rolling_forecast_df, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-180:], df['Close'][-180:], label="Actual Close", color='blue')
    ax.plot(rolling_forecast_df['Date'], rolling_forecast_df['Predicted_Close'], label="Rolling Forecast", color='red')
    ax.set_title(f"Predicted Close Prices for {stock_name} (as of {datetime.date.today()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.grid(True)
    ax.legend()
    return fig

def rolling_forecast(df, best_model, n_periods, x_data, significant_lags_dict):
    try:
        var_model = VAR(df[['Close', 'High', 'Low', 'Open', 'Volume']])
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
            
            var_input = rolling_df[['Close', 'High', 'Low', 'Open', 'Volume']].iloc[-var_fitted.k_ar:]
            
            # Catch edge case where even after initial check, slicing fails
            if var_input.shape[0] < var_fitted.k_ar:
                st.warning(f"Insufficient data for step {i+1}. Forecasting halted early.")
                break
            
            var_forecast = var_fitted.forecast(y=var_input.values, steps=1)[0]
            predicted_close_var, predicted_high, predicted_low, predicted_open, predicted_volume = var_forecast

            next_period = pd.DataFrame({
                'Close': [max(predicted_close_var, 0.01)], 
                'High': [max(predicted_high, 0.01)],
                'Low': [max(predicted_low, 0.01)], 
                'Open': [max(predicted_open, 0.01)],
                'Volume': [max(predicted_volume, 0)]
                }, index=[new_date])

            latest_data = pd.concat([rolling_df[['Close', 'High', 'Low', 'Open', 'Volume']], next_period])
            new_row = latest_data.copy()

            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                for lag in significant_lags_dict[col]['pacf']:
                    if lag > 0:
                        new_row[f'{col}_lag{lag}'] = new_row[col].shift(lag).iloc[-1]
                for ma_lag in significant_lags_dict[col]['acf']:
                    if ma_lag > 0:
                        new_row[f'{col}_ma_lag{ma_lag}'] = new_row[col].shift(1).rolling(window=ma_lag).mean().iloc[-1]

            feature_cols = new_row.columns.difference(['Close', 'High', 'Low', 'Open', 'Volume'])
            new_row = pd.DataFrame(new_row[feature_cols].values, columns=feature_cols, index=new_row.index).tail(1)
            
            new_row = new_row.reset_index().rename(columns={'index': 'Date'})
            new_row = create_date_features(new_row)
            new_row = new_row.set_index('Date').asfreq('B').dropna()
            new_row = new_row[x_data.columns]

            predicted_value = max(best_model.predict(new_row)[0], 0.01)
            rolling_predictions.append(predicted_value)

            final_row = pd.DataFrame({
                'Close': [predicted_value],
                 'High': [predicted_high],
                'Low': [predicted_low],
                'Open': [predicted_open],
                'Volume': [predicted_volume]
                }, index=[new_date])
            
            rolling_df = pd.concat([rolling_df, final_row])
            if i % 5 == 0 or i == n_periods -1:
                progress_bar.progress((i + 1) / n_periods, text=f"Day {i+1}/{n_periods} forecasted...")
        
        progress_bar.empty()
        return rolling_predictions, rolling_df

    except Exception as e:
        st.error(f"An error occurred during rolling forecast: {e}")
        return [], df

def finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods):
    rolling_forecast_df = pd.DataFrame({
        'Date': pd.date_range(start=df.index[-1], periods=n_periods + 1, freq='B')[1:],
        'Predicted_Close': rolling_predictions})

    horizon_df = rolling_forecast_df.head(15)
    predicted_high_15_days = round(horizon_df['Predicted_Close'].max(), 2)
    predicted_low_15_days = round(horizon_df['Predicted_Close'].min(), 2)
    predicted_avg_15_days = round(horizon_df['Predicted_Close'].mean(), 2)
    predicted_volatility_15_days = round(horizon_df['Predicted_Close'].std() / predicted_avg_15_days, 3) if predicted_avg_15_days > 0 else 0

    direction = 'flat'
    if horizon_df['Predicted_Close'].tail(5).mean() > horizon_df['Predicted_Close'].head(5).mean():
        direction = 'up'
    elif horizon_df['Predicted_Close'].tail(5).mean() < horizon_df['Predicted_Close'].head(5).mean():
        direction = 'down'

    max_buy_price = round((predicted_avg_15_days * (1 - (0.5 * predicted_volatility_15_days))), 2)
    target_sell_price = round((predicted_avg_15_days * (1 + (0.5 * predicted_volatility_15_days))), 2)
    predicted_return = ((target_sell_price / max_buy_price) - 1)

    recommendation = 'avoid/sell'
    if direction == 'up' and predicted_return > 0.03:
        recommendation = 'hold' if predicted_volatility_15_days > 0.10 else 'buy'

    summary_df = pd.DataFrame({
        'Ticker Symbol': [stock_name], 
        'Predicted_High_15_Day': [predicted_high_15_days],
        'Predicted_Low_15_Day': [predicted_low_15_days], 
        'Predicted_Avg_15_Day': [predicted_avg_15_days],
        'Predicted_Volatility_%': [predicted_volatility_15_days * 100], 
        'Max_Buy_Price': [max_buy_price],
        'Target_Sell_Price': [target_sell_price], 
        'Direction': [direction], 
        'Recommendation': [recommendation],
        'Predicted_Return_%': [predicted_return * 100]})
    
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

    stock_list_str = st.text_area(
        "Paste Stock Tickers Here", 
        default_stocks, 
        height=150,
        help="Paste a list of tickers..."
    )

    uploaded_file = st.file_uploader(
        "Or Upload a File", 
        type=['txt', 'csv', 'xlsx'],
        help="Upload a .txt, .csv, or .xlsx file with one ticker per line, or in the first column."
    )
    
    st.subheader("Forecasting Parameters")
    n_periods = st.slider("Forecast Horizon (days)", 10, 100, 50)
    
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
                stock_list = parse_and_clean_tickers(uploaded_file.getvalue().decode("utf-8"))
            
            # For csv/xlsx, assume tickers are in the first column
            if 'df_upload' in locals():
                if not df_upload.empty:
                    # Convert first column to list to be parsed
                    raw_tickers = df_upload.iloc[:, 0].tolist()
                    stock_list = parse_and_clean_tickers(raw_tickers)
                else:
                    st.warning("Uploaded file is empty.")

        elif stock_list_str:
            st.info("Processing tickers from text area.")
            stock_list = parse_and_clean_tickers(stock_list_str)
    
    except Exception as e:
        st.error(f"An error occurred while processing inputs: {e}")

    if not tiingo_api_key:
        st.error("`TIINGO_API_KEY` environment variable not set. Please set it to your Tiingo API key.")
    elif not stock_list:
        st.error("No valid stock tickers found. Please enter tickers in the text box or upload a file.")
    else:
        st.success(f"Found {len(stock_list)} unique tickers to process: {', '.join(stock_list)}")
        today = datetime.date.today()
        end_date = today + pd.offsets.BusinessDay(1)
        
        forecast_results = {}
        summary_results = []
        
        # Create an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            
            # Main loop for processing each stock
            for stock_name in stock_list:
                st.header(f"Processing: {stock_name}")
                
                with st.spinner(f"Fetching data for {stock_name}..."):
                    df = get_data(stock_name, end_date, tiingo_api_key)
                
                if df is None:
                    st.error(f"Could not retrieve data for {stock_name}. Skipping.")
                    continue
                
                MIN_HISTORY_REQUIRED = 500
                if len(df) < MIN_HISTORY_REQUIRED:
                    st.warning(f"{stock_name} has only {len(df)} historical records. Skipping due to insufficient data.")
                    continue

                with st.spinner(f"Creating features for {stock_name}..."):
                    significant_lags_dict = {}
                    df, significant_lags_dict = create_lagged_features(df, interpolate='bfill')
                    x_data, y_data, x_train, x_test, y_train, y_test = train_test_split(df)

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
                        study.optimize(objectives[model_name], n_trials=max_trials)
                        
                        best_params = study.best_params
                        
                        ### THIS WILL NEED TO BE CHANGED IF ADDITIONAL MODELS ARE ADDED IN THE FUTURE ###
                        model = ElasticNet(**best_params)

                        model.fit(x_train, y_train)
                        mae = study.best_value
                        model_scores[model_name] = (model, mae)

                        if mae < best_mae_for_stock:
                            best_mae_for_stock = mae
                            best_model_for_stock = model
                            best_model_name_for_stock = model_name

                        y_pred = model.predict(x_test)
                        fig = plot_actual_vs_predicted(y_train, y_test, y_pred, model_name, stock_name)
                        
                        st.write(f"**Model for {stock_name}** (MAE: {mae:.4f})")
                        st.pyplot(fig)
                
                st.success(f"Best Model for {stock_name}: Mean Absolute Error (MAE): **{best_mae_for_stock:.4f}**")

                # --- Forecasting ---
                st.subheader(f"Forecast for {stock_name}")
                with st.spinner(f"Re-training best model on full data and forecasting..."):
                    X_full, y_full = df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume']), df['Close']
                    best_model_for_stock.fit(X_full, y_full)
                    
                    rolling_predictions, rolling_df = rolling_forecast(df, best_model_for_stock, n_periods, x_data, significant_lags_dict)
                    rolling_forecast_df, summary_df = finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods)
                
                forecast_results[stock_name] = rolling_forecast_df
                summary_results.append(summary_df)

                fig_forecast = save_plot_forecast(df, rolling_forecast_df, stock_name)
                st.pyplot(fig_forecast)

                sheet_name = re.sub(r'[\[\]\*:\?/\\ ]', '_', stock_name)[:31]
                
                if save_forecasts_to_excel:
                    rolling_forecast_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    autofit_columns(rolling_forecast_df, worksheet)
                
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
