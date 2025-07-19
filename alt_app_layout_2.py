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
from darts.models import NBEATSModel, NHiTSModel, ExponentialSmoothing
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mae
from datetime import datetime, timedelta
from io import BytesIO
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# --- Basic Setup ---
warnings.filterwarnings("ignore")
logging.getLogger("darts").setLevel(logging.CRITICAL)
logging.getLogger("pytorch_lightning").setLevel(logging.CRITICAL)
logging.getLogger("lightning.pytorch").setLevel(logging.CRITICAL)
logging.getLogger("numexpr").setLevel(logging.CRITICAL)
logging.getLogger("optuna").setLevel(logging.CRITICAL)
st.set_page_config(page_title="Fortune Teller Stock Forecasting App", layout="wide", initial_sidebar_state="expanded")

class FortuneTeller:
    """
    Contains the core data processing, forecasting, and optimization logic for the app.
    Supports N-BEATS and N-HiTS forecasting techniques with default configurations and integrated constraint handling.
    """
    def __init__(self, params):
        self.params = params
        self.scaler = None
        self.covariate_scaler = None
        self.model = None
        self.target_columns = self.params.get('target_variables', [])
        self.covariate_columns = []
        self.historical_data = self.params.get('data', pd.DataFrame())

    def _prepare_data(self):
        """Prepares the data for Darts, including resampling and scaling, and selects covariates based on correlation."""
        df = self.historical_data.copy()
        
        # Ensure 'Date' column exists and is datetime type before proceeding
        if 'Date' not in df.columns:
            st.error("The historical data DataFrame is missing a 'Date' column.")
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        numeric_cols = df.select_dtypes(include=np.number).columns
        df_resampled = df.set_index('Date')[numeric_cols].resample(self.params['freq']).mean().reset_index()
        self.historical_data = df_resampled # Update with resampled data for consistency

        target_series = TimeSeries.from_dataframe(df_resampled, 'Date', self.target_columns, freq=self.params['freq'])
        
        # Check if target_series is empty after creation
        if target_series.n_timesteps == 0:
            st.error("Selected target variables resulted in an empty time series after data preparation. Please check your data and selections.")
            return None # Indicate failure to prepare data

        potential_covariates = [col for col in df_resampled.columns if col not in ['Date'] + self.target_columns]
        
        selected_covariates = []
        if potential_covariates and self.target_columns:
            primary_target = self.target_columns[0]
            if primary_target in df_resampled.columns:
                correlation_threshold = self.params.get('correlation_threshold', 0.5)
                correlations = df_resampled[potential_covariates].corrwith(df_resampled[primary_target]).abs()
                
                selected_covariates = correlations[correlations >= correlation_threshold].index.tolist()
                
                if selected_covariates:
                    st.sidebar.info(f"Selected {len(selected_covariates)} covariates with correlation >= {correlation_threshold}.")
        
        self.covariate_columns = selected_covariates
        st.session_state.covariate_columns = self.covariate_columns

        if self.covariate_columns:
            covariate_series = TimeSeries.from_dataframe(df_resampled, 'Date', self.covariate_columns, freq=self.params['freq'])
            self.covariate_scaler = Scaler()
            self.covariate_scaler.fit(covariate_series)
        
        self.scaler = Scaler()
        series_scaled = self.scaler.fit_transform(target_series)
        return series_scaled

    def _get_model_instance(self, params=None):
        """Instantiates a model with given parameters."""
        model_name = self.params['model_name']
        model_params = params.copy() if params else {}
        
        common_params = {
            'random_state': 42,
            'n_epochs': self.params.get('max_epochs', 50),
            'pl_trainer_kwargs': {"enable_progress_bar": False, "logger": False}
        }
        model_params.update(common_params)

        if model_name == "N-HiTS":
            return NHiTSModel(**model_params)
        else: # N-BEATS
            model_params['generic_architecture'] = False
            model_params.setdefault('num_stacks', 2)
            return NBEATSModel(**model_params)

    def _get_default_params(self, series_length):
        """Returns paper-recommended default parameters, adjusted for data length."""
        horizon = self.params['horizon']
        input_chunk_length = max(horizon * 2, 24)
        
        if input_chunk_length + horizon >= series_length:
            st.warning(f"Dataset is short for the requested forecast horizon. Adjusting model parameters.")
            input_chunk_length = max(1, series_length - horizon)
        
        if input_chunk_length < 1:
            st.error(f"Historical data is too short to train a model.")
            return None

        if self.params['model_name'] == "N-HiTS":
            return {'input_chunk_length': input_chunk_length, 'output_chunk_length': horizon, 'num_stacks': 2, 'num_blocks': 1, 'num_layers': 2, 'layer_widths': 512}
        else: # N-BEATS
            return {'input_chunk_length': input_chunk_length, 'output_chunk_length': horizon, 'num_stacks': 2, 'num_blocks': 3, 'num_layers': 4, 'layer_widths': 256}

    def _train_model(self, series_scaled):
        """Trains the model."""

        params = self._get_default_params(len(series_scaled))
        if params is None: return
        self.model = self._get_model_instance(params)

        past_covs = None
        if self.covariate_columns:
            cov_series = TimeSeries.from_dataframe(self.historical_data, 'Date', self.covariate_columns, freq=self.params['freq'])
            past_covs = self.covariate_scaler.transform(cov_series)
        
        self.model.fit(series_scaled, past_covariates=past_covs, verbose=False)
        st.success("Model successfully trained.")

    def _get_hybrid_future_covariates(self, user_scenario_df):
        """
        Creates a complete future covariate series by combining user-provided scenarios
        with forecasts for any unspecified variables.
        """
        if not self.covariate_columns:
            return None, None

        last_hist_date = pd.to_datetime(self.historical_data['Date']).max()
        future_dates = pd.date_range(start=last_hist_date + self.params['offset'], periods=self.params['horizon'], freq=self.params['freq'])
        
        # Start with an empty DataFrame for the future
        hybrid_cov_df = pd.DataFrame(index=future_dates)

        for col in self.covariate_columns:
            # If user provided data for this column, use it
            if col in user_scenario_df.columns and not user_scenario_df[col].isnull().all():
                # Reindex to ensure it matches the future dates, then interpolate
                user_col_data = user_scenario_df.set_index('Date')[col]
                hybrid_cov_df[col] = user_col_data.reindex(future_dates).interpolate(method='linear').bfill().ffill()
            else:
                # If no user data, forecast the covariate
                hist_ts = TimeSeries.from_dataframe(self.historical_data, 'Date', col, freq=self.params['freq'])
                # Use a simple, fast model for covariate forecasting
                cov_model = ExponentialSmoothing()
                cov_model.fit(hist_ts)
                forecasted_ts = cov_model.predict(n=self.params['horizon'])
                hybrid_cov_df[col] = forecasted_ts.to_series()
        
        # Combine with historical data for scaling
        hist_cov_df = self.historical_data.set_index('Date')[self.covariate_columns]
        full_cov_df = pd.concat([hist_cov_df, hybrid_cov_df])
        full_cov_df.index.name = 'Date' # Explicitly name the index before resetting it

        # Ensure full_cov_df is not empty before creating TimeSeries
        if full_cov_df.empty:
            st.warning("Full covariate DataFrame is empty. Cannot create TimeSeries for scaling.")
            return None, None

        full_cov_series = TimeSeries.from_dataframe(full_cov_df.reset_index(), 'Date', self.covariate_columns, freq=self.params['freq'])
        
        # Ensure covariate_scaler is fitted before transforming
        if self.covariate_scaler is None:
            st.error("Covariate scaler not fitted. Cannot transform covariate series.")
            return None, None
        
        scaled_cov_series = self.covariate_scaler.transform(full_cov_series)
        
        return scaled_cov_series, hybrid_cov_df

    def _generate_forecast(self, series_scaled, future_covariates_scaled=None):
        """Generates a forecast using the trained model."""
        # Check if series_scaled is valid before predicting
        if series_scaled is None or series_scaled.n_timesteps == 0:
            st.error("Cannot generate forecast: input series is empty or invalid.")
            return None
        
        # Prepare past_covariates for the predict method
        # If future_covariates_scaled is an empty TimeSeries, set it to None
        # to avoid Darts' internal indexing error (KeyError: 0)
        covariates_to_pass = future_covariates_scaled
        if covariates_to_pass is not None and covariates_to_pass.n_timesteps == 0:
            st.warning("Future covariates series is empty. Proceeding without past_covariates.")
            covariates_to_pass = None

        forecast_scaled = self.model.predict(n=self.params['horizon'], series=series_scaled, past_covariates=covariates_to_pass)
        return self.scaler.inverse_transform(forecast_scaled)

    def _run_constrained_optimization(self, series_scaled, base_covariates_df):
        """
        Runs optimization to find a future covariate scenario that respects constraints
        and results in a model that best fits historical data.
        """
        st.info("Starting constrained forecast optimization...")
        constraints = self.params.get('optimization_constraints', {})
        horizon = self.params['horizon']
        epsilon = 1e-6 # Small value to prevent division by zero

        def objective(trial):
            # 1. Suggest future values for each covariate
            suggested_covariates = {}
            for col in self.covariate_columns:
                base_series = base_covariates_df[col]
                suggested_values = []
                for i in range(horizon):
                    val = base_series.iloc[i]
                    # Suggest values in a +/- 20% range around the base forecast
                    low_bound = min(val * 0.8, val * 1.2)
                    high_bound = max(val * 0.8, val * 1.2)
                    if low_bound >= high_bound: low_bound, high_bound = high_bound, low_bound
                    if low_bound == high_bound: high_bound += epsilon
                    suggested_values.append(trial.suggest_float(f"{col}_{i}", low_bound, high_bound))
                suggested_covariates[col] = suggested_values
            
            future_cov_df = pd.DataFrame(suggested_covariates, index=base_covariates_df.index)

            # 2. ENFORCE HARD CONSTRAINTS (value, range, integer)
            enforced_cov_df = future_cov_df.copy()
            for key, cons in constraints.items():
                var = cons.get('variable')
                if var and var in enforced_cov_df.columns:
                    if cons['type'] == 'minimum_value': enforced_cov_df[var] = enforced_cov_df[var].clip(lower=cons['value'])
                    elif cons['type'] == 'maximum_value': enforced_cov_df[var] = enforced_cov_df[var].clip(upper=cons['value'])
                    elif cons['type'] == 'range': enforced_cov_df[var] = enforced_cov_df[var].clip(lower=cons['min'], upper=cons['max'])
            for key, cons in constraints.items():
                var = cons.get('variable')
                if var and var in enforced_cov_df.columns:
                    if cons['type'] == 'integer': enforced_cov_df[var] = enforced_cov_df[var].round().astype(int)

            # 3. Create full covariate series and re-train a model with them
            full_scaled_cov_series, _ = self._get_hybrid_future_covariates(enforced_cov_df.reset_index().rename(columns={'index':'Date'}))
            if self.covariate_columns and (full_scaled_cov_series is None or full_scaled_cov_series.n_timesteps == 0): return float('inf')

            model_params = self._get_default_params(len(series_scaled))
            if model_params is None: return float('inf')
            trial_model = self._get_model_instance(model_params)
            
            past_covs_for_fit = full_scaled_cov_series if self.covariate_columns else None
            
            try:
                trial_model.fit(series_scaled, past_covariates=past_covs_for_fit, verbose=False)
            except Exception: return float('inf') # Skip trial if model fitting fails

            # 4. Calculate historical forecast error (the new objective)
            # Use historical_forecasts to get an in-sample forecast to measure fit.
            try:
                start_point = series_scaled.time_index[model_params['input_chunk_length']]
                historical_forecast = trial_model.historical_forecasts(
                    series_scaled, past_covariates=past_covs_for_fit,
                    start=start_point, retrain=False, verbose=False
                )
                historical_error = mae(series_scaled, historical_forecast)
            except Exception: return float('inf')

            # 5. Calculate penalty for SOFT constraints (ratios)
            trial_forecast_ts_scaled = trial_model.predict(n=horizon, series=series_scaled, past_covariates=past_covs_for_fit)
            trial_forecast_df = self.scaler.inverse_transform(trial_forecast_ts_scaled).to_dataframe()
            
            trial_full_df = pd.concat([trial_forecast_df, enforced_cov_df], axis=1)
            penalty = 0.0
            for key, cons in constraints.items():
                if cons['type'] in ['maintain_ratio', 'maximum_ratio', 'minimum_ratio']:
                    num_var, den_var = cons.get('numerator'), cons.get('denominator')
                    if num_var not in trial_full_df.columns or den_var not in trial_full_df.columns:
                        penalty += 1e9; continue
                    num_ts, den_ts = trial_full_df[num_var], trial_full_df[den_var]
                    if (np.abs(den_ts) < epsilon).any(): penalty += 1e9
                    else:
                        ratio = num_ts / den_ts
                        if cons['type'] == 'maintain_ratio': penalty += np.sum(np.abs(ratio - cons['value'])) * 1e6
                        elif cons['type'] == 'maximum_ratio': penalty += np.sum(np.maximum(0, ratio - cons['value'])) * 1e6
                        elif cons['type'] == 'minimum_ratio': penalty += np.sum(np.maximum(0, cons['value'] - ratio)) * 1e6

            return historical_error + penalty

        study = optuna.create_study(direction='minimize')
        # Note: Retraining the model in each trial is slow. A lower n_trials is recommended for responsiveness.
        study.optimize(objective, n_trials=self.params.get('n_trials', 25), timeout=600)
        st.success("Optimization finished.")
        
        # Reconstruct the optimal covariates from the best trial's raw parameters
        best_params = study.best_params
        optimal_cov_df = pd.DataFrame({col: [best_params[f"{col}_{i}"] for i in range(horizon)] for col in self.covariate_columns}, index=base_covariates_df.index)

        # Re-apply hard constraints to the final best parameters to ensure they are met
        final_enforced_df = optimal_cov_df.copy()
        for key, cons in constraints.items():
            var = cons.get('variable')
            if var and var in final_enforced_df.columns:
                if cons['type'] == 'minimum_value': final_enforced_df[var] = final_enforced_df[var].clip(lower=cons['value'])
                elif cons['type'] == 'maximum_value': final_enforced_df[var] = final_enforced_df[var].clip(upper=cons['value'])
                elif cons['type'] == 'range': final_enforced_df[var] = final_enforced_df[var].clip(lower=cons['min'], upper=cons['max'])
        for key, cons in constraints.items():
            var = cons.get('variable')
            if var and var in final_enforced_df.columns:
                if cons['type'] == 'integer': final_enforced_df[var] = final_enforced_df[var].round().astype(int)
        
        # Retrain a final model using the optimal covariates and generate the final forecast
        st.info("Retraining final model with optimal constrained covariates...")
        scaled_optimal_cov_series, _ = self._get_hybrid_future_covariates(final_enforced_df.reset_index().rename(columns={'index':'Date'}))
        if self.covariate_columns and (scaled_optimal_cov_series is None or scaled_optimal_cov_series.n_timesteps == 0):
            st.error("Optimal covariates series is invalid. Cannot generate constrained forecast.")
            return None, None

        model_params = self._get_default_params(len(series_scaled))
        if model_params is None: return None, None
        final_constrained_model = self._get_model_instance(model_params)

        past_covs_for_fit = scaled_optimal_cov_series if self.covariate_columns else None
        final_constrained_model.fit(series_scaled, past_covariates=past_covs_for_fit, verbose=False)

        constrained_forecast_scaled = final_constrained_model.predict(
            n=horizon, series=series_scaled, past_covariates=past_covs_for_fit
        )
        constrained_forecast_ts = self.scaler.inverse_transform(constrained_forecast_scaled)

        if constrained_forecast_ts is None: return None, None

        return constrained_forecast_ts.to_dataframe(), final_enforced_df


    def run_pipeline(self):
        """Runs the full forecasting and optional optimization pipeline."""
        series_scaled = self._prepare_data()
        if series_scaled is None: return {}

        self._train_model(series_scaled)
        if self.model is None: return {}

        st.info("Generating base forecast using hybrid scenario...")
        base_scaled_cov_series, base_covariates_df = self._get_hybrid_future_covariates(st.session_state.get('scenario_data', pd.DataFrame()))
        
        if self.covariate_columns and (base_scaled_cov_series is None or base_scaled_cov_series.n_timesteps == 0):
            st.error("Failed to prepare base covariates for forecasting. Check data and covariate selection.")
            return {}
        
        base_forecast_ts = self._generate_forecast(series_scaled, base_scaled_cov_series)
        if base_forecast_ts is None: return {}

        results = {'base': base_forecast_ts.to_dataframe(), 'base_covariates': base_covariates_df}

        if self.params.get('optimization_constraints'):
            constrained_forecast_df, optimal_covariates_df = self._run_constrained_optimization(series_scaled, results['base_covariates'])
            
            if constrained_forecast_df is not None:
                results['constrained'] = constrained_forecast_df
                results['optimal_covariates'] = optimal_covariates_df
            else:
                st.warning("Constrained optimization failed or returned no valid forecast.")
        
        return results

# --- UI and Helper Functions ---
def detect_frequency(data):
    if 'Date' not in data.columns or len(data) < 2: return 0
    date_diffs = pd.to_datetime(data['Date']).diff().dropna()
    avg_diff = date_diffs.mean()
    if avg_diff.days <= 2: return 2 # Daily
    if 6 <= avg_diff.days <= 8: return 1 # Weekly
    return 0 # Monthly

def add_date_column(data, periodicity):
    st.info(f"No 'Date' column found. Generating one with {periodicity} frequency.")
    end_date = datetime.today() - timedelta(days=1)
    if periodicity == 'Daily': start_date = end_date - timedelta(days=len(data)-1); freq_str = 'D'
    elif periodicity == 'Weekly': start_date = end_date - timedelta(weeks=len(data)-1); freq_str = 'W'
    else: start_date = end_date - pd.DateOffset(months=len(data)-1); freq_str = 'MS'
    data['Date'] = pd.date_range(start=start_date, periods=len(data), freq=freq_str)
    return data[['Date'] + [col for col in data.columns if col != 'Date']]

def create_correlation_heatmap(df, target_variable, title):
    if target_variable not in df.columns: return None
    numeric_df = df.select_dtypes(include=np.number)
    if target_variable not in numeric_df.columns: return None # Should not happen if target_variable is from all_numeric_cols
    corr_data = numeric_df.corrwith(numeric_df[target_variable]).to_frame(name="Correlation with " + target_variable).drop(target_variable, errors='ignore')
    corr_data = corr_data.sort_values(by="Correlation with " + target_variable, ascending=False)
    height = max(300, len(corr_data) * 20) # Adjust height dynamically
    fig = px.imshow(corr_data, text_auto='.2f', aspect="auto", color_continuous_scale='viridis_r', labels=dict(color="Correlation"))
    fig.update_layout(height=height, title=title, title_x=0.0)
    return fig

@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Forecast", index=True)
    return output.getvalue()

def initialize_scenario_data(data, horizon, freq_str, offset):
    covariate_columns = st.session_state.get('covariate_columns', [])
    if not covariate_columns:
        st.session_state.scenario_data = pd.DataFrame()
        return
    last_hist_date = pd.to_datetime(data['Date']).max()
    future_dates = pd.date_range(start=last_hist_date + offset, periods=horizon, freq=freq_str)
    
    scenario_df = pd.DataFrame(index=future_dates, columns=covariate_columns).fillna(np.nan)
    st.session_state.scenario_data = scenario_df.reset_index().rename(columns={'index': 'Date'})

def main():
    st.title("🔮 Fortune Teller Stock Forecasting App")
    st.divider()

    # --- Initialize Session State ---
    # Initialize all session state variables at the very beginning to prevent AttributeError
    if 'data' not in st.session_state: st.session_state.data = pd.DataFrame()
    if 'scenario_data' not in st.session_state: st.session_state.scenario_data = pd.DataFrame()
    if 'results' not in st.session_state: st.session_state.results = {}
    if 'params' not in st.session_state:
        st.session_state.params = {
            'target_variables': [],
            'horizon': 36,
            'freq': "MS",
            'offset': pd.DateOffset(months=1), # Default to monthly initially
            'model_name': "N-HiTS",
            'correlation_threshold': 0.5, # Default correlation threshold
            'max_epochs': 100,
            'n_trials': 25,
            'optimization_constraints': {}, # Initialize optimization constraints
            'optimization_objective': {} # Initialize optimization objective
        }
    if 'covariate_columns' not in st.session_state: st.session_state.covariate_columns = []
    if 'last_uploaded_file_id' not in st.session_state: st.session_state.last_uploaded_file_id = None
    if 'current_historical_plot_var' not in st.session_state: st.session_state.current_historical_plot_var = None

    with st.sidebar:
        st.header("⚙️ Configuration")
        uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx"])
        
        if uploaded_file:
            current_uploaded_df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith('.xlsx') else pd.read_csv(uploaded_file)
            current_uploaded_df.columns = current_uploaded_df.columns.str.strip()

            if st.session_state.last_uploaded_file_id != uploaded_file.file_id:
                has_valid_date_column = False
                if 'Date' in current_uploaded_df.columns:
                    try:
                        current_uploaded_df['Date'] = pd.to_datetime(current_uploaded_df['Date']); has_valid_date_column = True
                    except Exception: pass # Date column exists but is not parsable, will fall through to prompt

                if not has_valid_date_column:
                    st.warning("No valid 'Date' column found. Please specify the data frequency to generate one.")
                    periodicity_options = ["Monthly", "Weekly", "Daily"]
                    selected_periodicity = st.selectbox("Select the data frequency:", periodicity_options, key="date_freq_selector")
                    if st.button("Generate Date Column", key="generate_date_btn"):
                        st.session_state.data = add_date_column(current_uploaded_df.copy(), selected_periodicity)
                        st.session_state.data['Date'] = pd.to_datetime(st.session_state.data['Date']) # Ensure it's datetime
                        st.session_state.last_uploaded_file_id = uploaded_file.file_id # Mark this file as processed
                        st.rerun() # Rerun to apply the new data state
                    st.stop() # Stop further execution until user generates date column
                
                st.session_state.data = current_uploaded_df.copy()
                st.session_state.last_uploaded_file_id = uploaded_file.file_id # Mark this file as processed
                st.session_state.results = {}
                st.session_state.scenario_data = pd.DataFrame()
                st.session_state.params = {
                    'target_variables': [], 'horizon': 36, 'freq': "MS", 
                    'offset': pd.DateOffset(months=1), 'model_name': "N-HiTS", 'correlation_threshold': 0.5,
                    'max_epochs': 100, 'n_trials': 25, 'optimization_constraints': {},
                    'optimization_objective': {}
                }
                all_cols = [c for c in st.session_state.data.columns if c != 'Date' and pd.api.types.is_numeric_dtype(st.session_state.data[c])]
                default_target = all_cols[0] if all_cols else None
                st.session_state.params['target_variables'] = [default_target] if default_target else []
                st.session_state.current_historical_plot_var = default_target
                temp_freq_map = {"Monthly": "MS", "Weekly": "W-MON", "Daily": "D"}
                temp_offset_map = {"Monthly": pd.DateOffset(months=1), "Weekly": pd.DateOffset(weeks=1), "Daily": pd.DateOffset(days=1)}
                detected_freq_idx = detect_frequency(st.session_state.data)
                detected_freq_str = ["Monthly", "Weekly", "Daily"][detected_freq_idx]
                st.session_state.params['freq'] = temp_freq_map[detected_freq_str]
                st.session_state.params['offset'] = temp_offset_map[detected_freq_str]
                temp_foresight_instance = FortuneTeller(st.session_state.params)
                temp_foresight_instance.historical_data = st.session_state.data.copy() 
                temp_foresight_instance._prepare_data()
                initialize_scenario_data(st.session_state.data, st.session_state.params['horizon'], st.session_state.params['freq'], st.session_state.params['offset'])
                st.rerun() # Rerun to ensure all session state is stable and UI updates

        if not st.session_state.data.empty:
            df = st.session_state.data
            all_numeric_cols = [c for c in df.columns if c != 'Date' and pd.api.types.is_numeric_dtype(df[c])]
            
            st.subheader("Forecasting Parameters")
            target_variables = st.multiselect("Select Target Variables", all_numeric_cols, default=st.session_state.params.get('target_variables', []))
            horizon = st.number_input("Forecasting Horizon", min_value=1, value=st.session_state.params.get('horizon', 36))
            
            st.subheader("Model Settings")
            model_name = st.selectbox("Select Model", ["N-HiTS", "N-BEATS"], index=["N-HiTS", "N-BEATS"].index(st.session_state.params.get('model_name', "N-HiTS")))

            st.subheader("Covariate Selection")
            correlation_threshold = st.slider("Min Correlation for Covariates", 0.0, 1.0, st.session_state.params.get('correlation_threshold', 0.5), 0.01)

            st.subheader("Constraint Modeling")
            with st.expander("Add New Constraint", expanded=False):
                # Ensure constrainable_vars are updated based on current selections
                constrainable_vars = target_variables + [col for col in st.session_state.data.select_dtypes(include=np.number).columns if col not in target_variables]
                constrainable_vars = sorted(list(set(constrainable_vars))) # Get unique sorted list
                
                if constrainable_vars:
                    constraint_target_variable = st.selectbox("Select Variable to Constrain:", ['Select a variable'] + constrainable_vars, key="new_constraint_var_select")
                    if constraint_target_variable != 'Select a variable':
                        constraint_type_new = st.selectbox(
                            f"Constraint Type for {constraint_target_variable}:",
                            ["Min Value", "Max Value", "Range", "Must be Integer", "Maintain Ratio", "Maximum Ratio", "Minimum Ratio"],
                            key=f"new_constraint_type_{constraint_target_variable}"
                        )

                        if constraint_type_new == "Min Value":
                            min_val = st.number_input(f"Minimum value for {constraint_target_variable}:", value=0.0, key=f"new_opt_min_{constraint_target_variable}")
                            if st.button(f"Add Min Value Constraint", key=f"add_min_btn_{constraint_target_variable}"):
                                st.session_state.params['optimization_constraints'][f"min_{constraint_target_variable}"] = {'type': 'minimum_value', 'variable': constraint_target_variable, 'value': min_val}; st.rerun()
                        elif constraint_type_new == "Max Value":
                            max_val = st.number_input(f"Maximum value for {constraint_target_variable}:", value=100.0, key=f"new_opt_max_{constraint_target_variable}")
                            if st.button(f"Add Max Value Constraint", key=f"add_max_btn_{constraint_target_variable}"):
                                st.session_state.params['optimization_constraints'][f"max_{constraint_target_variable}"] = {'type': 'maximum_value', 'variable': constraint_target_variable, 'value': max_val}; st.rerun()
                        elif constraint_type_new == "Range":
                            col_min, col_max = st.columns(2)
                            min_val = col_min.number_input(f"Min value:", value=0.0, key=f"new_opt_range_min_{constraint_target_variable}")
                            max_val = col_max.number_input(f"Max value:", value=100.0, key=f"new_opt_range_max_{constraint_target_variable}")
                            if st.button(f"Add Range Constraint", key=f"add_range_btn_{constraint_target_variable}"):
                                st.session_state.params['optimization_constraints'][f"range_{constraint_target_variable}"] = {'type': 'range', 'variable': constraint_target_variable, 'min': min_val, 'max': max_val}; st.rerun()
                        elif constraint_type_new == "Must be Integer":
                            if st.button(f"Add Integer Constraint", key=f"add_int_btn_{constraint_target_variable}"):
                                st.session_state.params['optimization_constraints'][f"integer_{constraint_target_variable}"] = {'type': 'integer', 'variable': constraint_target_variable}; st.rerun()
                        elif constraint_type_new in ["Maintain Ratio", "Maximum Ratio", "Minimum Ratio"]:
                            denominator_options = [col for col in constrainable_vars if col != constraint_target_variable]
                            if denominator_options:
                                ratio_denominator_var = st.selectbox(f"Denominator:", denominator_options, key=f"new_opt_ratio_denom_{constraint_target_variable}")
                                ratio_value = st.number_input(f"Ratio Value:", value=1.0, key=f"new_opt_ratio_val_{constraint_target_variable}")
                                if st.button(f"Add Ratio Constraint", key=f"add_ratio_btn_{constraint_target_variable}"):
                                    ratio_key = f"{constraint_type_new.replace(' ', '_').lower()}_{constraint_target_variable}_vs_{ratio_denominator_var}"
                                    st.session_state.params['optimization_constraints'][ratio_key] = {'type': constraint_type_new.replace(" ", "_").lower(), 'numerator': constraint_target_variable, 'denominator': ratio_denominator_var, 'value': ratio_value, 'variable': f"{constraint_target_variable}/{ratio_denominator_var}"}; st.rerun()
                            else: st.warning("Need at least two variables for a ratio.")
                else: st.info("Select target variables or ensure covariates are identified to enable constraints.")

            if st.session_state.params['optimization_constraints']:
                st.markdown("**Current Constraints:**")
                for key, const in st.session_state.params['optimization_constraints'].items():
                    col1, col2 = st.columns([4, 1])
                    msg = ""
                    if const['type'] == 'minimum_value': msg = f"**{const['variable']}** >= `{const['value']}`"
                    elif const['type'] == 'maximum_value': msg = f"**{const['variable']}** <= `{const['value']}`"
                    elif const['type'] == 'range': msg = f"**{const['variable']}** in [`{const['min']}`, `{const['max']}`]"
                    elif const['type'] == 'integer': msg = f"**{const['variable']}** must be an Integer"
                    elif const['type'] in ['maintain_ratio', 'maximum_ratio', 'minimum_ratio']:
                        op = "==" if const['type'] == 'maintain_ratio' else "<=" if const['type'] == 'maximum_ratio' else ">="
                        msg = f"**{const['numerator']}** / **{const['denominator']}** {op} `{const['value']}`"
                    col1.write(f"- {msg}")
                    if col2.button("X", key=f"del_{key}"): del st.session_state.params['optimization_constraints'][key]; st.rerun()
            else: st.info("No optimization constraints defined yet.")

            if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
                if not target_variables: st.error("Please select at least one target variable.")
                else:
                    with st.spinner("Running pipeline... This may be a while, especially with constraints."):
                        st.session_state.params.update({'data': df, 'target_variables': target_variables, 'horizon': horizon, 'model_name': model_name, 'correlation_threshold': correlation_threshold})
                        model = FortuneTeller(st.session_state.params)                          
                        st.session_state.results = model.run_pipeline()
                        st.rerun()

    # --- Main Area Display ---
    if st.session_state.data.empty:
        st.info("👋 Welcome! Please upload a data file to begin.")
    else:
        if st.session_state.scenario_data.empty and st.session_state.params.get('target_variables'):
            initialize_scenario_data(st.session_state.data, st.session_state.params['horizon'], st.session_state.params['freq'], st.session_state.params['offset'])

        st.subheader("Historical Data Explorer")
        col1, col2 = st.columns([2, 1])
        with col1:
            all_numeric_cols_for_plot = [c for c in st.session_state.data.columns if c != 'Date' and pd.api.types.is_numeric_dtype(st.session_state.data[c])]
            if st.session_state.current_historical_plot_var is None or st.session_state.current_historical_plot_var not in all_numeric_cols_for_plot:
                st.session_state.current_historical_plot_var = all_numeric_cols_for_plot[0] if all_numeric_cols_for_plot else None
            if st.session_state.current_historical_plot_var:
                try: default_index = all_numeric_cols_for_plot.index(st.session_state.current_historical_plot_var)
                except ValueError: default_index = 0
                plot_var_hist = st.selectbox("Select variable to explore:", all_numeric_cols_for_plot, index=default_index, key="historical_plot_selector")
                st.session_state.current_historical_plot_var = plot_var_hist
                fig_hist = px.line(st.session_state.data, x='Date', y=plot_var_hist, title=f"Historical Trend for {plot_var_hist}")
                st.plotly_chart(fig_hist, use_container_width=True)
            else: st.warning("No numeric columns available to display.")

        with col2:
            if st.session_state.current_historical_plot_var: # Only create correlation heatmap if a variable is selected
                fig_corr = create_correlation_heatmap(st.session_state.data, st.session_state.current_historical_plot_var, f"Correlations with {st.session_state.current_historical_plot_var}")
                if fig_corr: st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Select a variable in the left chart to see its correlations.")
        
        if st.session_state.results:
            st.divider(); st.subheader("📊 Results")
            base_forecast = st.session_state.results.get('base')
            constrained_forecast = st.session_state.results.get('constrained')
            tabs = ["Base Forecast"]
            if constrained_forecast is not None: tabs.append("Constrained Forecast")
            display_tabs = st.tabs(tabs)
            with display_tabs[0]:
                st.markdown("#### Base Forecast")
                st.write("Forecast based on historical trends and your hybrid scenario inputs.")
                if base_forecast is not None:
                    plot_var_forecast_base = st.selectbox("View Forecast for:", base_forecast.columns, key="base_select")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=st.session_state.data['Date'], y=st.session_state.data[plot_var_forecast_base], name='Historical'))
                    fig.add_trace(go.Scatter(x=base_forecast.index, y=base_forecast[plot_var_forecast_base], name='Base Forecast', line=dict(dash='dash')))
                    st.plotly_chart(fig, use_container_width=True)
            if len(display_tabs) > 1 and constrained_forecast is not None:
                with display_tabs[1]:
                    st.markdown("#### Constrained Forecast")
                    plot_var_opt = st.selectbox("View Forecast for:", constrained_forecast.columns, key="opt_select")
                    fig_opt = go.Figure()
                    fig_opt.add_trace(go.Scatter(x=st.session_state.data['Date'], y=st.session_state.data[plot_var_opt], name='Historical'))
                    fig_opt.add_trace(go.Scatter(x=base_forecast.index, y=base_forecast[plot_var_opt], name='Base Forecast', line=dict(color='grey', dash='dash')))
                    fig_opt.add_trace(go.Scatter(x=constrained_forecast.index, y=constrained_forecast[plot_var_opt], name='Constrained Forecast', line=dict(color='green')))
                    st.plotly_chart(fig_opt, use_container_width=True)
                    st.markdown("##### Optimal Future Inputs (Covariates)")
                    st.dataframe(st.session_state.results.get('optimal_covariates'), use_container_width=True)
        
        st.divider(); st.subheader("💾 Data & Scenarios")
        tab1, tab2 = st.tabs(["🧪 Input Scenario Data", "📤 Upload Scenario Data"])
        with tab1:
            st.info("Optional: Edit future values for covariates. You can copy/paste from Excel. Unspecified values will be forecasted.")
            if not st.session_state.scenario_data.empty:
                gb = GridOptionsBuilder.from_dataframe(st.session_state.scenario_data)
                gb.configure_column("Date", editable=False); gb.configure_default_column(editable=True, type=["numericColumn"])
                gb.configure_grid_options(enableRangeSelection=True) # Enables Excel-like copy/paste
                grid_response = AgGrid(st.session_state.scenario_data.fillna(''), gridOptions=gb.build(), update_mode=GridUpdateMode.MODEL_CHANGED, fit_columns_on_grid_load=True, height=400, allow_unsafe_jscode=True)
                st.session_state.scenario_data = grid_response['data'].replace('', np.nan)
            else:
                st.warning("Select Target Variables to initialize the scenario grid.")
        with tab2:
            st.info(f"Upload a CSV/Excel file with a 'Date' column and any of the following optional columns: {', '.join(st.session_state.get('covariate_columns',[]))}")
            scenario_file = st.file_uploader("Upload Scenario File", type=["csv", "xlsx"], key="scenario_uploader")
            if scenario_file:
                try:
                    new_scenario_df = pd.read_excel(scenario_file) if scenario_file.name.endswith('.xlsx') else pd.read_csv(scenario_file)
                    new_scenario_df['Date'] = pd.to_datetime(new_scenario_df['Date'])
                    current_scenario = st.session_state.scenario_data.set_index('Date')
                    uploaded_data = new_scenario_df.set_index('Date')
                    current_scenario.update(uploaded_data)
                    st.session_state.scenario_data = current_scenario.reset_index()
                    st.success("Scenario data uploaded and merged successfully!")
                    st.rerun()
                except Exception as e: st.error(f"Error processing scenario file: {e}")

if __name__ == "__main__":
    main()
