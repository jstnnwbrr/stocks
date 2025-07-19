import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import logging
import optuna
import os
import numpy as np
import warnings

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
from datetime import datetime, timedelta
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)
logging.getLogger("optuna").setLevel(logging.CRITICAL)
st.set_page_config(page_title="Stock Forecasting", layout="wide", initial_sidebar_state="expanded")

class FortuneTeller:
    def __init__(self, params):
        self.params = params
        self.scaler = None
        self.covariate_scaler = None
        self.model = None
        self.target_columns = self.params['target_variables']
        self.covariate_columns = []

    def prepare_data(self):
        df = self.params['data'].copy()
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_resampled = df.set_index('Date')[numeric_cols].resample(self.params['freq']).mean().reset_index()

        target_series = TimeSeries.from_dataframe(df_resampled, 'Date', self.target_columns, freq=self.params['freq'])
        self.covariate_columns = [col for col in df_resampled.columns if col not in ['Date'] + self.target_columns]
        st.session_state.covariate_columns = self.covariate_columns

        if self.covariate_columns:
            covariate_series = TimeSeries.from_dataframe(df_resampled, 'Date', self.covariate_columns, freq=self.params['freq'])
            self.covariate_scaler = Scaler()
            self.covariate_scaler.fit(covariate_series)
        
        self.scaler = Scaler()
        series_scaled = self.scaler.fit_transform(target_series)
        return series_scaled, df_resampled
    
    def train_model(self, series_scaled, df_resampled):
        st.info("Starting hyperparameter optimization...This may take a while.")

        val_len = max(self.params['horizon'], 12)
        if val_len >= len(series_scaled):
            st.error(f"Not enough historical data ({len(series_scaled)} periods) for a validation split of {val_len} periods. Please reduce forecast horizon or provide more historical data.")
            self.model = None
            return
        
        train_series, val_series = series_scaled[:-val_len], series_scaled[-val_len:]

        effective_train_len = len(train_series)
        min_input_chunk_length = 12

        if effective_train_len < min_input_chunk_length * 2:
            st.warning(f"Warning: Available training data length ({effective_train_len} periods) is very short relative to the minimum input chunk length ({min_input_chunk_length}). This lead to underfitting and poor predictive performance.")

        train_df_resampled = df_resampled[df_resampled['Date'] <= train_series.time_index.max()]
        val_df_resampled = df_resampled[df_resampled['Date'] > train_series.time_index.max()]

        def objective(trial):
            max_input_chunk_length_bound = min(effective_train_len // 2, 48)
            input_chunk_length = trial.suggest_int('input_chunk_length', min_input_chunk_length, max_input_chunk_length_bound)
            output_chunk_length = trial.suggest_int('output_chunk_length', 1, min(input_chunk_length, self.params['horizon'], 24))
            num_stacks = trial.suggest_int('num_stacks', 2, 50)
            num_blocks = trial.suggest_int('num_blocks', 1, 3)
            num_layers = trial.suggest_int('num_layers', 2, 10)
            layer_widths = trial.suggest_int('layer_widths', 128, 512)
            dropout = trial.suggest_float('dropout', 0.0, 0.4)

            past_covariates_scaled_train = None
            if self.covariate_columns and self.covariate_scaler:
                past_covariate_series_train = TimeSeries.from_dataframe(train_df_resampled, 'Date', self.covariate_columns, freq=self.params['freq'])
                past_covariates_scaled_train = self.covariate_scaler.transform(past_covariate_series_train)
            
            past_covariates_scaled_full_for_val_pred = None
            if self.covariate_columns and self.covariate_scaler:
                full_covariate_df_for_val_pred = pd.concat([train_df_resampled, val_df_resampled]).sort_values('Date').drop_duplicates(subset=['Date'])
                full_covariate_series_for_val_pred = TimeSeries.from_dataframe(full_covariate_df_for_val_pred, 'Date', self.covariate_columns, freq=self.params['freq'])
                past_covariates_scaled_full_for_val_pred = self.covariate_scaler.transform(full_covariate_series_for_val_pred)

            model = NBEATSModel(input_chunk_length=input_chunk_length,
                                output_chunk_length=output_chunk_length,
                                generic_architecture=False,
                                num_stacks=num_stacks,
                                num_blocks=num_blocks,
                                num_layers=num_layers,
                                layer_widths=layer_widths,
                                dropout=dropout,
                                random_state=101,
                                n_epochs=self.params['max_epochs'],
                                pl_trainer_kwargs={"enable_progress_bar": True, "logger": False})
            
            model.fit(train_series, past_covariates=past_covariates_scaled_train, verbose=False)

            val_forecast_scaled = model.predict(n=len(val_series), series=train_series, past_covariates=past_covariates_scaled_full_for_val_pred)
            val_forecast = self.scaler.inverse_transform(val_forecast_scaled)
            mae_score = mae(val_series, val_forecast)

            trial.set_user_attr('model', model)
            return mae_score
        
        sampler = optuna.samplers.TPESampler(seed=101)
        study = optuna.create_study(direction='minimize', sampler=sampler)

        best_mae = float('inf')
        trials_without_improvement = 0
        best_trial_found = None

        st.write(f"Beginning trials. Metric used is mean absolute error (MAE).")

        for i in range(self.params['n_trials']):
            try:
                study.optimize(objective, n_trials=1, timeout=3600)
                current_mae = study.best_value

                if current_mae < best_mae:
                    previous_best_mae = best_mae
                    best_mae = current_mae
                    trials_without_improvement = 0
                    best_trial_found = study.best_trial

                    if previous_best_mae == float('inf'):
                        st.write(f"Trial {i+1}/{self.params['n_trials']}: New Best MAE {best_mae:,.3f}")
                    else:
                        st.write(f"Trial {i+1}/{self.params['n_trials']}: New Best MAE  {best_mae:,.3f}. Percentage Improvement: {((previous_best_mae - best_mae)/previous_best_mae)*100:.2f}%")
                
                else:
                    trials_without_improvement += 1
                    st.write(f"Trial {i+1}/{self.params['n_trials']}: MAE: {current_mae:,.3f}. Trials without improvement: {trials_without_improvement}")
                
                if trials_without_improvement >= self.params['patience']:
                    st.info(f"Early stopping triggered after {trials_without_improvement} trials without improvement.")
                    break

            except Exception as e:
                st.warning(f"Optimization trial {i+1} failed: {e}. Skipping this trial.")

        if best_trial_found:
            best_params = best_trial_found.params
            st.success(f"Optimization finished. Best MAE: {study.best_value:,.3f}")
            st.write(best_params)

            st.info("Retraining model on full historical data with optimized hyperparameters...")

            past_covariates_scaled_full_train = None
            if self.covariate_columns and self.covariate_scaler:
                full_covariate_series_for_full_train = TimeSeries.from_dataframe(df_resampled, 'Date', self.covariate_columns, freq=self.params['freq'])
                past_covariates_scaled_full_train = self.covariate_scaler.transform(full_covariate_series_for_full_train)
            
            retrained_model = NBEATSModel(
                input_chunk_length=best_params['input_chunk_length'],
                output_chunk_length=best_params['output_chunk_length'],
                generic_architecture=False,
                num_stacks=best_params['num_stacks'],
                num_blocks=best_params['num_blocks'],
                num_layers=best_params['num_layers'],
                layer_widths=best_params['layer_widths'],
                dropout=best_params['dropout'],
                random_state=101,
                n_epochs=50,
                pl_trainer_kwargs={"enable_progress_bar": True, "logger": False})
            
            retrained_model.fit(series_scaled, past_covariates=past_covariates_scaled_full_train, verbose=False)
            self.model = retrained_model
            st.success("Model successfully retrained on full data.")
        else:
            st.error("Optimization completed, but no successful trials found or no improvement. No model trained.")
            self.model = None

    def generate_forecast(self, series_scaled):
        past_covariates_scaled = None
        
        if self.covariate_columns:
            historical_covariate_df = self.params['data'].copy()
            historical_covariate_df['Date'] = pd.to_datetime(historical_covariate_df['Date'])
            historical_covariate_df = historical_covariate_df.set_index('Date')[self.covariate_columns].resample(self.params['freq']).mean().reset_index()

            all_dates_needed = pd.date_range(start=historical_covariate_df['Date'].min(),
                                             periods=len(historical_covariate_df) + self.params['horizon'],
                                             freq=self.params['freq'])
            
            full_covariate_df_template = pd.DataFrame({'Date': all_dates_needed})
            full_covariate_df = pd.merge(full_covariate_df_template, historical_covariate_df, on='Date', how='left')

            scenario_df = st.session_state.get('scenario_data', pd.DataFrame())
            if not scenario_df.empty:
                scenario_df['Date'] = pd.to_datetime(scenario_df['Date'])
                scenario_df = scenario_df[['Date'] + self.covariate_columns].set_index('Date').resample(self.params['freq']).mean().reset_index()
                full_covariate_df = pd.merge(full_covariate_df, scenario_df, on='Date', how='left', suffixes=('_existing', '_scenario'))

                for col in self.covariate_columns:
                    full_covariate_df[col] = full_covariate_df[f"{col}_scenario"].fillna(full_covariate_df[f"{col}_existing"])

                full_covariate_df = full_covariate_df.filter(items=['Date'] + self.covariate_columns)
                full_covariate_df = full_covariate_df.sort_values('Date').reset_index(drop=True)

            for col in self.covariate_columns:
                full_covariate_df[col] = full_covariate_df[col].ffill()
                full_covariate_df[col] = full_covariate_df[col].bfill()
                full_covariate_df[col] = full_covariate_df[col].fillna(0)

            past_covariates_series = TimeSeries.from_dataframe(full_covariate_df, 'Date', self.covariate_columns, freq=self.params['freq'])
            past_covariates_scaled = self.covariate_scaler.transform(past_covariates_series)
        else:
            past_covariates_scaled = None

        forecast_scaled = self.model.predict(
            n=self.params['horizon'],
            series=series_scaled,
            past_covariates=past_covariates_scaled
        )

        forecast_ts = self.scaler.inverse_transform(forecast_scaled)
        return forecast_ts.to_dataframe()
    
    def run_forecast(self):
        series_scaled, df_resampled = self.prepare_data()
        self.train_model(series_scaled, df_resampled)

        if self.model is None:
            st.error("Model training failed or was skipped due to insufficient data. Cannot generate forecast.")
            return pd.DataFrame()
        else:
            forecast_df = self.generate_forecast(series_scaled)
            return forecast_df
        
## UI & Helper Functions ##
def detect_frequency(data):
    if 'Date' not in data.columns or len(data) < 2: return 0
    date_diffs = pd.to_datetime(data['Date']).diff().dropna()
    avg_diff = date_diffs.mean()
    if avg_diff.days < 2: return 2
    elif 6 <= avg_diff.days <= 8: return 1
    return 0

def add_date_column(data, periodicity):
    st.info(f"No 'Date' column found. Generating one with {periodicity} frequency.")
    today = datetime.today()
    num_rows = len(data)
    if periodicity == 'Daily':
        offset = pd.offsets.BusinessDay(1)
        freq_str = 'B'
    elif periodicity == 'Weekly':
        offset = pd.DateOffset(weeks=1)
        freq_str = 'W'
    else:
        offset = pd.DateOffset(months=1)
        freq_str = 'M'
    
    end_date = today - timedelta(days=1)
    start_date = end_date - offset * (num_rows - 1)
    date_range = pd.date_range(start=start_date, periods=num_rows, freq=freq_str)
    data['Date'] = date_range
    cols = ['Date'] + [col for col in data.columns if col != 'Date']
    return data[cols]

def create_correlation_heatmap(df, target_variable, title):
    if target_variable not in df.columns:
        return None
    
    corr_data = df.corr(numeric_only=True)[[target_variable]].sort_values(by=target_variable, ascending=False)
    corr_data = corr_data.drop(target_variable)
    num_vars = len(corr_data)
    height = max(300, num_vars * 20)
    fig = px.imshow(corr_data, text_auto='.2f', aspect="auto",
                    color_continuous_scale='viridis_r',
                    labels=dict(color="Correlation"))
    fig.update_layout(height=height, title=title, title_x=0.0)
    return fig

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if df is not None:
            df.to_excel(writer, sheet_name="Forecast", index=True)
    return output.getvalue()

def initialize_scenario_data(data, horizon, freq_str, offset):
    covariate_columns = st.session_state.get('covariate_columns', [])
    if not covariate_columns:
        st.session_state.scenario_data = pd.DataFrame()
        return
    
    last_hist_date = data['Date'].max()
    future_dates = pd.date_range(start=last_hist_date + offset, periods=horizon, freq=freq_str)

    scenario_df = pd.DataFrame(index=future_dates)
    for col in covariate_columns:
        scenario_df[col] = np.nan
    
    st.session_state.scenario_data = scenario_df.reset_index().rename(columns={'index': 'Date'})

def main():
    st.title("Stock Forecasting")
    st.markdown("Upload your data, select your targets, and generate accurate, interpretable forecasts.")
    st.divider()

    if 'data' not in st.session_state: st.session_state.data = pd.DataFrame()
    if 'scenario_data' not in st.session_state: st.session_state.scenario_data = pd.DataFrame()
    if 'forecast' not in st.session_state: st.session_state.forecast = None
    if 'params' not in st.session_state: st.session_state.params = None
    if 'covariate_columns' not in st.session_state: st.session_state.covariate_columns = []

    with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"], key="main_uploader")

        if uploaded_file:
            if st.session_state.get('last_uploaded_file_id') != uploaded_file.file_id:
                for key in ['data', 'scenario_data', 'forecast', 'params', 'covariate_columns']:
                    st.session_state[key] = pd.DataFrame() if 'data' in key else ([] if 'columns' in key else None)

                st.session_state.last_uploaded_file_id = uploaded_file.file_id
                try:
                    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
                    df.columns = df.columns.str.strip()
                    st.session_state.data = df
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    st.session_state.data = pd.DataFrame()
                st.rerun()

            if not st.session_state.data.empty:
                if 'date' not in st.session_state.data.columns.str.lower():
                    temp_periodicity = st.selectbox("Data Frequency for (Date generation)", ["Monthly", "Weekly", "Daily"], index=0, key="temp_freq")
                    if st.button("Generate Date Column"):
                        st.date_input("Data End Date")
                        st.session_state.data = add_date_column(st.session_state.data, temp_periodicity)
                        st.rerun()

                else:
                    st.session_state.data['Date'] = pd.to_datetime(st.session_state.data['Date'])
                    all_cols = [c for c in st.session_state.data.columns if c != 'Date' and pd.api.types.is_numeric_dtype(st.session_state.data[c])]

                    st.subheader("Forecasting Parameters")
                    target_variables = st.multiselect("Select Target Variables", all_cols)
                    horizon = st.number_input("Forecasting Horizon", min_value=1, value=36)
                    periodicity = st.selectbox("Data Frequency", ["Monthly", "Weekly", "Daily"], index=detect_frequency(st.session_state.data))

                    st.subheader("Model Settings")
                    max_epochs = st.number_input("Max Epochs", min_value=10, max_value=100, value=50)
                    n_trials = st.number_input("Max Trials", min_value=5, max_value=50, value=20)
                    patience = st.number_input("Max Patience", min_value=1, max_value=50, value=15)

                    if st.button("Generate Forecast", type="primary", use_container_width=True):
                        if not target_variables:
                            st.error("Please select at least one target variable.")
                        else:
                            with st.spinner("Generating forecast...this may take a moment"):
                                freq_map = {"Monthly": "MS", "Weekly": "W-MON", "Daily": "B"}
                                offset_map = {"Monthly": pd.DateOffset(months=1), "Weekly": pd.DateOffset(weeks=1), "Daily": pd.offsets.BusinessDay(1)}

                                st.session_state.params = {
                                    'data': st.session_state.data,
                                    'target_variables': target_variables,
                                    'horizon': horizon,
                                    'freq': freq_map[periodicity],
                                    'offset': offset_map[periodicity],
                                    'max_epochs': max_epochs,
                                    'n_trials': n_trials,
                                    'patience': patience
                                }

                                model = FortuneTeller(st.session_state.params)
                                forecast_results = model.run_forecast()
                                st.session_state.forecast = forecast_results
                                st.rerun()

    ## Main Area Display ##
    if st.session_state.data.empty:
        st.info("Welcome to the Stock Forecasting App!")
    else:
        if st.session_state.params:
            if st.session_state.scenario_data.empty:
                initialize_scenario_data(st.session_state.data, st.session_state.params['horizon'], st.session_state.params['freq'], st.session_state.params['offset'])

        st.subheader("Historical Data Explorer")
        numeric_cols = st.session_state.data.select_dtypes(include=np.number)
        default_var = st.session_state.params['target_variables'][0] if st.session_state.params else numeric_cols.std().idxmax()

        if default_var:
            col1, col2 = st.columns([3, 2])
            with col1:
                plot_var = st.selectbox("Select variable to explore:", numeric_cols.columns, index=list(numeric_cols.columns).index(default_var))
                fig_hist = px.line(st.session_state.data, x='Date', y=plot_var, title=f"Historical Trend for {plot_var}", height=max(275, ((len(st.session_state.data.columns) - 1) *15)))
                st.plotly_chart(fig_hist, use_container_width=True)
            with col2:
                fig_corr = create_correlation_heatmap(numeric_cols, plot_var, f"Correlations with {plot_var}")
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No numeric columns available for historical data exploration or target selection.")

        if st.session_state.forecast is not None:
            st.divider()
            st.subheader("Forecast Results")

            display_target = st.selectbox("View Forecast for:", st.session_state.params['target_variables'])

            if display_target:
                fig_fc = go.Figure()
                actual_data = st.session_state.data.set_index('Date')[display_target]
                fig_fc.add_trace(go.Scatter(x=actual_data.index, y=actual_data, mode='lines', name="Historical Actual"))
                fig_fc.add_trace(go.Scatter(x=st.session_state.forecast.index, y=st.session_state.forecast[display_target], mode='lines', name="Forecast", line=dict(color='orange', dash='dash')))
                fig_fc.update_layout(title=f"Forecast vs. Actuals for {display_target}", title_x=0.5, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_fc, use_container_width=True)
                excel_export_data = convert_df_to_excel(st.session_state.forecast)
                st.download_button("Export Forecast Data (Excel)", excel_export_data, f"forecast_N-BEATS.xlsx", use_container_width=True)

        st.divider()
        st.subheader("Data & Scenarios")

        tab1, tab2, tab3 = st.tabs(["Historical Data", "Input Scenario Data", "Upload Scenario Data"])

        with tab1:
            st.dataframe(st.session_state.data, use_container_width=True)
        with tab2:
            if not st.session_state.covariate_columns:
                st.info("No covariate columns were identified, which could be a result of empty data or all variables being selected as target variables. Scenario input is not applicable.")
            elif st.session_state.scenario_data.empty:
                st.warning("Click 'Generate Forecast' at least once to initialize the scenario input table.")
            else:
                st.info("Edit the values for future covariates below. The forecast will automatically use this data on the next run.")
                gb = GridOptionsBuilder.from_dataframe(st.session_state.scenario_data)
                gb.configure_column('Date', editable=False)
                gb.configure_default_column(editable=True)
                grid_response = AgGrid(
                    st.session_state.scenario_data,
                    gridOptions=gb.build(),
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    fit_columns_on_grid_load=True,
                    theme='streamlit',
                    height=600,
                    allow_unsafe_jscode=True,
                )
                st.session_state.scenario_data = grid_response['data']

        with tab3:
            if not st.session_state.covariate_columns:
                st.info("No covariate columns were identified, which could be a result of empty data or all variables being selected as target variables. Scenario upload is not applicable.")
                scenario_file = st.file_uploader("Upload Scenario File", type=["csv", "xlsx"], key="scenario_uploader")
                if scenario_file:
                    try:
                        new_scenario_df = pd.read_excel(scenario_file) if scenario_file.name.endswith('.xlsx') else pd.read_csv(scenario_file)
                        new_scenario_df.columns = new_scenario_df.columns.str.strip()

                        if 'Date' in new_scenario_df.columns and all(col in new_scenario_df.columns for col in st.session_state.covariate_columns):
                            st.session_state.scenario_data = new_scenario_df
                            st.success("Scenario data uploaded successfully! It will be used on the next forecast run.")
                            st.rerun()
                        else:
                            st.error(f"Upload failed. File must contain 'Date' and all covariate columns: {st.session_state.covariate_columns}")
                    except Exception as e:
                        st.error(f"Error processing scenario file: {e}")

if __name__ == "__main__":
    main()