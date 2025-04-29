import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import statsmodels.api as sm
import warnings
import yfinance as yf

from sklearn.linear_model import ElasticNet, TweedieRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")

########################
# Define functions

def get_data(stock_name, end_date):
    df = yf.download(stock_name, start='2008-01-01', end=end_date, multi_level_index=False)
    df = df.reset_index()
    df = create_date_features(df)
    df = df.set_index('Date').asfreq('B').dropna()
    df['Close'].plot(title=f"{stock_name}")
    plt.show()
    
    return df

def create_date_features(df):
    df['Date'] = pd.to_datetime(df['Date'], unit='B')
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day'] = df['Date'].dt.daysinmonth
    df['day_of_week'] = df['Date'].dt.day_of_week
    df['is_month_end'] = df['Date'].dt.is_month_end.astype('int64')
    df['is_month_start'] = df['Date'].dt.is_month_start.astype('int64')
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype('int64')
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype('int64')
    
    return df

# Typically, we would use an alpha of 0.05 (statistically significant p-value, but for stock market predictions need to expand tolerance)
def get_significant_lags(series, alpha=0.20, nlags=None):
    """Calculate significant lags using ACF and PACF."""
    
    acf_values = acf(series, nlags=nlags)
    pacf_values = pacf(series, nlags=nlags)
    
    confint_acf = sm.tsa.stattools.acf(series, alpha=alpha, nlags=nlags)[1]
    confint_pacf = sm.tsa.stattools.pacf(series, alpha=alpha, nlags=nlags)[1]
    
    significant_acf_lags = np.where(np.abs(acf_values) > confint_acf[:, 1])[0]
    significant_pacf_lags = np.where(np.abs(pacf_values) > confint_pacf[:, 1])[0]
    
    return significant_acf_lags, significant_pacf_lags

def create_lagged_features(df, interpolate):
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        significant_acf, significant_pacf = get_significant_lags(df[col])
        
        significant_lags_dict[col] = {'acf' : significant_acf,
                                      'pacf' : significant_pacf}
        
        for ma_lag in significant_acf:
            df[f'{col}_ma_lag{ma_lag}'] = df[col].shift(1).rolling(window=ma_lag).mean()
            
        for lag in significant_pacf:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            
    df.dropna(inplace=True)
    df = df.interpolate(interpolate)
    
    return df, significant_lags_dict

def train_test_split(df, train_size=0.80):
    # Separate target from features
    x_data, y_data = df.drop(columns=['Close','High','Low','Open','Volume']), df['Close']
    
    # Split data into training and test sets
    train_size = int(len(x_data) * train_size)
    x_train = x_data.iloc[:train_size]
    x_test = x_data.iloc[train_size:]
    y_train = y_data.iloc[:train_size]
    y_test = y_data.iloc[train_size:]

    return x_data, y_data, x_train, x_test, y_train, y_test
 
def objective_elastic_net(trial):
    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
    fit_intercept = trial.suggest_categorical('fit_intercept', [True, False])
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept)
    pipeline = Pipeline([('regressor', model)])
    
    try:
        pipeline.fit(x_train, y_train)
    except ValueError as e:
        print(f"Error during fitting: {e}")
        return float('inf')
    
    y_pred = pipeline.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    trial.set_user_attr('model', pipeline)
    
    return mae

def objective_tweedie_regressor(trial):
    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    power = trial.suggest_float('power', 1.1, 1.9)
    model = TweedieRegressor(alpha=alpha, power=power)
    pipeline = Pipeline([('regressor', model)])
    
    try:
        pipeline.fit(x_train, y_train)
    except ValueError as e:
        print(f"Error during fitting: {e}")
        return float('inf')
    
    y_pred = pipeline.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    trial.set_user_attr('model', pipeline)
    
    return mae

def objective_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_layer_sizes = tuple(trial.suggest_int(f"n_units_l{i}", 5, 250) for i in range(n_layers))
    solver = trial.suggest_categorical('solver', ['adam', 'lbfgs'])  
    alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-6, 1e-1, log=True)
    
    model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, 
                         activation='relu', 
                         solver=solver, 
                         alpha=alpha, 
                         learning_rate_init=learning_rate_init, 
                         max_iter=100, 
                         early_stopping=True,
                         random_state=101)
    
    pipeline = Pipeline([('regressor', model)])
    
    try:
        pipeline.fit(x_train, y_train)
    except ValueError as e:
        print(f"Error during fitting: {e}")
        return float('inf')
    
    y_pred = pipeline.predict(x_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    trial.set_user_attr('model', pipeline)
    
    return mae

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f"{model_name} - Actual vs. Predicted")
    plt.show()
    
def plot_actual_vs_predicted2(y_train, y_test, y_pred, model_name, stock_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label="Training Data", color="blue", linewidth=1)
    plt.plot(y_test.index, y_test, label="Test Data (Actuals)", color="green", linewidth=1)
    plt.plot(y_test.index, y_pred, label="Predicted Test Data", color="red", linewidth=1)
    plt.legend()
    plt.title(f"{model_name} - Actual vs Predicted Values on Test Set ({stock_name})")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.grid(True)  # Add a grid for better readability
    plt.show()

def save_plot_forecast(stock_name, df, rolling_forecast_df, plot_filename):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index[-180:], df['Close'][-180:], label="Actual Close", color='blue')
    plt.plot(rolling_forecast_df['Date'], rolling_forecast_df['Predicted_Close'], label="Rolling Forecast", color='red')
    plt.title(f"Predicted Close Prices for {stock_name} (as of {datetime.date.today()})")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.legend()

    # Save plot as PNG
    plt.savefig(plot_filename)
    plt.close()
    
def train_models(x_train, y_train, x_test, y_test, stock_name):
    print("\nStart Model Training\n", "_____________________\n", "Model Features: \n\n", x_train.columns)

    studies = {'ElasticNetRegression': optuna.create_study(direction='minimize'),
               'TweedieRegressor': optuna.create_study(direction='minimize'),
               'MLPRegressor': optuna.create_study(direction='minimize'),}


    for model_name, objective in zip(studies.keys(), [objective_elastic_net,
                                                      objective_tweedie_regressor,
                                                      objective_mlp]):
        print("\n", f"Optimizing {model_name}...")
        
        best_score = float('inf')
        trials_without_improvement = 0
        best_model_for_type = None
        
        for trial_number in range(max_trials):
            study = studies[model_name]
            
            try:
                study.optimize(objective, n_trials=1)
                current_score = study.best_value
                
                if current_score < best_score:
                    best_score = current_score
                    trials_without_improvement = 0
                    best_model_for_type = study.best_trial.user_attrs['model']
                    print(f"New Best score (MAE): {best_score} (Trial #: {trial_number})")
                else:
                    trials_without_improvement +=1
                    
                if trials_without_improvement >= patience:
                    print(f"Early stopping after {trial_number + 1} trials for {model_name}.")
                    break
                
            except Exception as e:
                print(f"Error during optimization for {model_name}: {e}")
                continue
            
        if study.best_trial:
            print(f"Best parameters for {model_name}: {study.best_trial.params}")
            print(f"Best score (MAE) for {model_name}: {best_score:.3f}")
        
        if best_model_for_type is not None:
            model_scores[model_name] = (best_model_for_type, best_score)
                
            y_pred = best_model_for_type.predict(x_test)
                
            plot_actual_vs_predicted(y_test, y_pred, model_name)
            plot_actual_vs_predicted2(y_train, y_test, y_pred, model_name, stock_name)
            print(pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred}))
     
    return model_scores

def rolling_forecast(df):
    # Fit VAR model before loop (using original df)
    var_model = VAR(df[['Close', 'High', 'Low', 'Open', 'Volume']])
    var_fitted = var_model.fit(ic='aic')  # adjust lags or use ic='aic'
    
    # Initialize rolling forecast dataframe with original data
    rolling_df = df.copy()
    
    # Store rolling predictions
    rolling_predictions = []
    
    # Rolling Forecast Loop
    for i in range(n_periods):
        # Get the last date in the DataFrame
        last_date = rolling_df.index[-1]
        
        # Create a new date (one period out)
        new_date = last_date + pd.offsets.BusinessDay(1)
        
        # VAR Forecast for supplemental variables
        var_input = rolling_df[['Close', 'High', 'Low', 'Open', 'Volume']].iloc[-var_fitted.k_ar:]
        var_forecast = var_fitted.forecast(y=var_input.values, steps=1)[0]
        predicted_close_var, predicted_high, predicted_low, predicted_open, predicted_volume = var_forecast
        
        predicted_close_var = max(predicted_close_var, 0.01)
        predicted_high = max(predicted_high, 0.01)
        predicted_low = max(predicted_low, 0.01)
        predicted_open = max(predicted_open, 0.01)
        predicted_volume = max(predicted_volume, 0)  # Allow 0 but not negative

        # Create a new row with placeholder values (will be updated)
        next_period = pd.DataFrame({
            'Close': [predicted_close_var],  # placeholder, will be overwritten by ML model later
            'High': [predicted_high],
            'Low': [predicted_low],
            'Open': [predicted_open],
            'Volume': [predicted_volume]},
            index=[new_date])
        
        # Append to get the latest data for lag/MA feature calculation
        latest_data = pd.concat([rolling_df[['Close', 'High', 'Low', 'Open', 'Volume']], next_period])
        
        new_row = latest_data.copy()
        
        for col in ['Close','High','Low','Open','Volume']:
            # Update rolling features (lags and moving averages)
            for lag in significant_lags_dict[col]['pacf']:
                new_row[f'{col}_lag{lag}'] = new_row[col].shift(1).iloc[-lag]
                
            for ma_lag in significant_lags_dict[col]['acf']:
                new_row[f'{col}_ma_lag{ma_lag}'] = new_row[col].shift(1).rolling(window=ma_lag).mean().iloc[-1]
        
        # Keep only latest row and drop raw columns
        feature_cols = new_row.columns.difference(['Close','High','Low','Open','Volume'])
        new_row = pd.DataFrame(new_row[feature_cols].values, columns=feature_cols, index=new_row.index).tail(1)

        # Backup approach to above - deactivate for now
        #new_row = pd.DataFrame(new_row[new_row.columns].drop(columns=['Close','High','Low','Open','Volume']).values, 
                               #columns=new_row[new_row.columns].drop(columns=['Close','High','Low','Open','Volume']).columns, 
                               #index=new_row.index).tail(1)
        
        new_row = new_row.reset_index().rename(columns={'index' : 'Date'})
        new_row = create_date_features(new_row)
        new_row = new_row.set_index('Date').asfreq('B').dropna()
        
        new_row = new_row[x_data.columns]
        
        # Make predictions
        predicted_value = best_model.predict(new_row)[0]
        predicted_value = max(predicted_value, 0.01)
        rolling_predictions.append(predicted_value)
        
        # Construct final row to append
        final_row = pd.DataFrame({'Close':[predicted_value],
                                  'High':[predicted_high],
                                  'Low':[predicted_low],
                                  'Open':[predicted_open],
                                  'Volume':[predicted_volume]}, index=[new_date])
        
        rolling_df = pd.concat([rolling_df, final_row])
        
    return rolling_predictions, rolling_df

def autofit_columns(df, worksheet):
    for i, column in enumerate(df.columns):
        try:
            column_width = max(df[column].astype(str).map(len).max(), len(column)) + 2
        except:
            column_width = len(column) + 2  # fallback in case of error
        worksheet.set_column(i, i, column_width)

def finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods, writer):  
    # Convert rolling predictions into DataFrame
    rolling_forecast_df = pd.DataFrame({
        'Date': pd.date_range(start=df.index[-1], periods=n_periods + 1, freq='B')[1:], 
        'Predicted_Close': rolling_predictions})

    # 30 calendar days ≈ 23 trading days
    horizon_df = rolling_forecast_df.head(23)

    predicted_high_30_days = round(horizon_df['Predicted_Close'].max(), 2)
    predicted_low_30_days = round(horizon_df['Predicted_Close'].min(), 2)
    predicted_avg_30_days = round(horizon_df['Predicted_Close'].mean(), 2)
    predicted_volatility_30_days = round(horizon_df['Predicted_Close'].std() / predicted_avg_30_days, 3)
    
    if horizon_df['Predicted_Close'].tail(5).mean() > horizon_df['Predicted_Close'].head(5).mean():
        direction = 'up'
    elif horizon_df['Predicted_Close'].tail(5).mean() < horizon_df['Predicted_Close'].head(5).mean():
        direction = 'down'
    else:
        direction = 'flat'
    
    max_buy_price = round((predicted_avg_30_days * (1 - predicted_volatility_30_days)), 2)
    target_sell_price = round((predicted_avg_30_days * (1 + predicted_volatility_30_days)), 2)
    
    if direction == 'up':
        if ((target_sell_price / max_buy_price) - 1) > 0.05 :
            recommendation = 'buy' 
        else:
            recommendation = 'hold'      
    else:
        recommendation = 'avoid/sell'

    print(f"\n\n{stock_name} - Predicted High Next 30 Days: {predicted_high_30_days:.2f}",
          f"\nPredicted Low Next 30 Days: {predicted_low_30_days:.2f}",
          f"\nPredicted Avg Price Next 30 Days: {predicted_avg_30_days:.2f}",
          f"\nPredicted Volatility Next 30 Days (Coefficient of Variation): {predicted_volatility_30_days:.2f}",
          f"\nMax Buy Price: {max_buy_price:.2f}",
          f"\nTarget Sell Price: {target_sell_price:.2f}",
          f"\nDirection: {direction}",
          f"\nRecommendation: {recommendation}")

    # Save metrics and forecast to the Excel sheet
    sheet_name = stock_name[:31]  # Use stock name as the sheet name, limit to 31 characters
    rolling_forecast_df.to_excel(writer, sheet_name=sheet_name, startrow=1, index=False)

    # Access xlsxwriter worksheet from the writer
    worksheet = writer.sheets[sheet_name]
    autofit_columns(rolling_forecast_df, worksheet)

    # Create a 1-row DataFrame for summary
    summary_df = pd.DataFrame({
        'Ticker Symbol': [stock_name],
        'Predicted_High_30_Days': [predicted_high_30_days],
        'Predicted_Low_30_Days': [predicted_low_30_days],
        'Predicted_Avg_30_Days': [predicted_avg_30_days],
        'Predicted_Volatility_%': [predicted_volatility_30_days],
        'Max_Buy_Price': [max_buy_price],
        'Target_Sell_Price': [target_sell_price],
        'Direction': [direction],
        'Recommendation': [recommendation]})

    plot_filename = f"{stock_name}_forecast.png"
    
    save_plot_forecast(stock_name, df, rolling_forecast_df, plot_filename)

    # Insert the plot image into the corresponding sheet
    worksheet.insert_image('D1', plot_filename, {'x_scale': 0.8, 'y_scale': 0.8})

    return rolling_forecast_df, summary_df

########################
# RUN APP
########################
# Specify stock names
stock_list = ['DHR',
              'CNC',
              'DLTR',
              'EMN',
              'CPB',
              'BTG',
              'AAPL',
              'CAH',
              'CAT',
              'CCI',
              'CCL',
              'CDW',
              'CF',
              'COF',
              'COO',
              'COST',
              'CPAY',
              'CRM',
              'CRWD',
              'CSGP',
              'CZR',
              'DAL',
              'DD',
              'DECK',
              'DIS',
              'DVN',
              'DXCM',
              'EFX',
              'EG',
              'ENPH',
              'NVDA',
              'TSLA',
              'INTC',
              'PLTR',
              'F',
              'SOFI',
              'LCID',
              'AAL',
              'GOOGL',
              'ABEV',
              'AVTR',
              'NU',
              'PONY',
              'NIO',
              'SMCI',
              'GRAB',
              'T',
              'AGNC',
              'RGTI',
              'AMZN',
              'PFE',
              'GOOG',
              'HOOD']

########################
# Specify high-level settings
today = datetime.date.today()
end_date = today + pd.offsets.BusinessDay(1)
max_trials = 17 # Maximum number of trials per modeling type
patience = 7 # If performance doesn't improve within this many tests, move on

# Forecast horizon
n_periods = 60

# Dictionary to hold your final DataFrames and summary stats
forecast_results = {}
summary_results = []

# Instantiate Excel writer
writer = pd.ExcelWriter(f"C:\\Users\\sar81\\Desktop\\stock_forecasts_{today}.xlsx", engine='xlsxwriter')

# Run process for each stock
for stock_name in stock_list:
    print(f"\nProcessing {stock_name}...")
    # Get data
    try: 
        df = get_data(stock_name, end_date)
      
        # Create lagged features for 'Close','High','Low','Open','Volume' 
        significant_lags_dict = {}
        df, significant_lags_dict = create_lagged_features(df, interpolate='bfill')
        
        # Separate target from features
        x_data, y_data, x_train, x_test, y_train, y_test = train_test_split(df, train_size=0.80)
        
        # Train and optimize models 
        model_scores = {}
        model_scores = train_models(x_train, y_train, x_test, y_test, stock_name) 
         
        best_model_name = min(model_scores, key=lambda k: model_scores[k][1])
        best_model, best_mae_score = model_scores[best_model_name]
        
        # Identify best model
        print("\nBest Model:\n\n", best_model, "\n\n") 
        y_pred = best_model.predict(x_test)
        y_pred_df = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})
        print("\n\n", y_pred_df, "\n\n")
        
        # Re-train best model using most recent data    
        X = df.drop(columns=['Close','High','Low','Open','Volume'])
        y = df['Close']
        
        best_model.fit(X=X, y=y)
        
        # Perform rolling forecast and generate key metrics
        rolling_predictions, rolling_forecast_and_inputs = rolling_forecast(df)
        rolling_forecast_df, summary_df = finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods, writer)
        
        # Prepare key forecast metrics for export
        forecast_results[stock_name] = rolling_forecast_df
        summary_results.append(summary_df)
    
        # Combine summaries into one DataFrame
        combined_summary = pd.concat(summary_results, ignore_index=True)

    except ValueError as e:
        print(f"Error: {e}")
        continue

combined_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
summary_worksheet = writer.sheets["Summary_Stats"]
autofit_columns(summary_df, summary_worksheet)
writer.close()
