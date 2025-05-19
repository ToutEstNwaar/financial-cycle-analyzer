# streamlit_app.py

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import tempfile
import math

# --- Python Path Modification ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Python Path Modification ---

from indicator_logic import settings as ind_settings
from indicator_logic.data_loader import load_ohlc_from_csv
from indicator_logic.main_calculator import calculate_core_cycle_components, sum_composite_waves
from utils.plotting import plot_indicator_lines, create_cycle_table

st.set_page_config(layout="wide", page_title="Cycle Indicator Analysis")
st.title("Financial Cycle Indicator Analysis :chart_with_upwards_trend:")
st.markdown("Upload OHLCV data (CSV). Analysis updates automatically as settings change.")

# --- Helper function to get default settings ---
def _get_default_settings():
    # DEFAULT_SETTINGS_DICT in settings.py now uses *_base keys for window sizes
    defaults = ind_settings.DEFAULT_SETTINGS_DICT.copy()
    defaults["source_price_type"] = "Close"
    return defaults

# --- Initialize Session State ---
def initialize_settings_in_session_state():
    default_settings = _get_default_settings()
    for key, value in default_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_settings_in_session_state() 

# --- Sidebar ---
st.sidebar.header("Indicator Settings")

if st.sidebar.button("Reset to Defaults", key="reset_settings_button"):
    default_settings = _get_default_settings()
    for key, value in default_settings.items():
        st.session_state[key] = value 
    st.rerun() 

uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"], key="file_uploader_widget")

max_data_len_default = 10000 
min_practical_window_size = 20 # Min for Past and Future Window Size sliders
min_allowable_max_per = 2 # Min for MaxPer slider

if 'ohlc_data_length_for_sliders' not in st.session_state:
    st.session_state.ohlc_data_length_for_sliders = max_data_len_default

if uploaded_file is not None and 'ohlc_data_for_length_check_processed' not in st.session_state:
    try:
        # Light check first
        temp_df = pd.read_csv(uploaded_file, nrows=1) 
        uploaded_file.seek(0) 
        if not temp_df.empty:
            df_len_check = pd.read_csv(uploaded_file) # Full load for length
            st.session_state.ohlc_data_length_for_sliders = len(df_len_check)
            uploaded_file.seek(0) 
            st.session_state.ohlc_data_for_length_check_processed = True 
    except Exception:
        st.session_state.ohlc_data_length_for_sliders = max_data_len_default

st.sidebar.subheader("Input Data Source")
source_options = ["Close", "Open", "High", "Low", "(H+L)/2", "(H+L+C)/3", "(O+H+L+C)/4", "(H+L+C+C)/4"]
st.sidebar.selectbox("Price Source", options=source_options, key="source_price_type")

st.sidebar.subheader("General Settings")

# --- Past Window Size ---
current_max_past_window = st.session_state.ohlc_data_length_for_sliders if uploaded_file else max_data_len_default
# Ensure session state value respects current dynamic max and practical min
if st.session_state.WindowSizePast_base > current_max_past_window:
    st.session_state.WindowSizePast_base = current_max_past_window
if st.session_state.WindowSizePast_base < min_practical_window_size:
    st.session_state.WindowSizePast_base = min_practical_window_size

st.sidebar.slider(
    "Past Window Size (Analysis Lookback)", # Label updated
    min_value=min_practical_window_size,
    max_value=max(min_practical_window_size, current_max_past_window), 
    step=10, 
    key="WindowSizePast_base", # Key remains _base as it's used by main_calculator
    help=(
        "Defines the number of historical bars (SampleSize) for the core cycle analysis. "
        "Backend validation will check compatibility with 'Max Period' and 'Bar to Calculate'. "
        "Also serves as the base length for the displayed past wave."
        )
)

# --- MaxPer ---
# Ensure session state value respects fixed min/max for slider
if st.session_state.MaxPer > 500:
    st.session_state.MaxPer = 500
if st.session_state.MaxPer < min_allowable_max_per:
    st.session_state.MaxPer = min_allowable_max_per

st.sidebar.slider(
    "Max Period (MaxPer)",
    min_value=min_allowable_max_per, 
    max_value=500, # Fixed max as per user request
    step=1,
    key="MaxPer",
    help=(
        "Max cycle period to search for. Backend validation will ensure compatibility with "
        "'Past Window Size' and other settings. Error will be shown if incompatible."
    )
)

# --- Future Window Size ---
if st.session_state.WindowSizeFuture_base > 1000:
    st.session_state.WindowSizeFuture_base = 1000
if st.session_state.WindowSizeFuture_base < min_practical_window_size:
    st.session_state.WindowSizeFuture_base = min_practical_window_size

st.sidebar.slider(
    "Future Window Size", # Label updated
    min_value=min_practical_window_size,
    max_value=1000, # Fixed max as per user request
    step=10,
    key="WindowSizeFuture_base", # Key remains _base
    help="Defines the base number of bars for the future wave projection. Actual projection length will be at least 2 * MaxPer."
)

# --- Bar to Calculate (Moved after Future Window Size) ---
# Ensure session state value respects fixed min/max for slider
if st.session_state.BarToCalculate > 100:
    st.session_state.BarToCalculate = 100
if st.session_state.BarToCalculate < 1:
    st.session_state.BarToCalculate = 1

st.sidebar.number_input(
    "Bar to Calculate (Offset)",
    min_value=1,
    max_value=100, # Fixed practical UI limit
    step=1,
    key="BarToCalculate",
    help="Offset from the most recent end of the 'Past Window Size' for Goertzel analysis. Backend will validate against 'Past Window Size'."
)

# --- Other General Settings ---
current_max_per_val_for_cycle_selection = st.session_state.MaxPer # MaxPer can now be up to 500

st.sidebar.number_input(
    "Start At Cycle (Rank)",
    min_value=1,
    max_value=max(1, current_max_per_val_for_cycle_selection), 
    step=1,
    key="StartAtCycle"
)
st.sidebar.number_input(
    "Use Top Cycles (Count)",
    min_value=1,
    max_value=max(1, current_max_per_val_for_cycle_selection),
    step=1,
    key="UseTopCycles"
)

st.sidebar.subheader("Source Price Processing")
detrend_mode_options = [
    ind_settings.NONE_SMTH_DT, ind_settings.ZLAGSMTH, ind_settings.HPSMTH,
    ind_settings.ZLAGSMTHDT, ind_settings.HPSMTHDT, ind_settings.LOG_ZLAG_REGRESSION_DT
]
st.sidebar.selectbox("Detrending/Smoothing Mode",options=detrend_mode_options,key="detrendornot")

if st.session_state.detrendornot == ind_settings.ZLAGSMTH:
    st.sidebar.slider("ZLMA Smooth Period", 1, 100, step=1, key="ZLMAsmoothPer")
elif st.session_state.detrendornot == ind_settings.HPSMTH:
    st.sidebar.slider("HPF Smoothing Period", 1, 100, step=1, key="HPsmoothPer")
elif st.session_state.detrendornot == ind_settings.ZLAGSMTHDT:
    st.sidebar.slider("ZLMA Detrend Fast Period", 1, 100, step=1, key="DT_ZLper1")
    st.sidebar.slider("ZLMA Detrend Slow Period", 1, 200, step=1, key="DT_ZLper2")
elif st.session_state.detrendornot == ind_settings.HPSMTHDT:
    st.sidebar.slider("HPF Detrend Fast Period", 1, 100, step=1, key="DT_HPper1")
    st.sidebar.slider("HPF Detrend Slow Period", 1, 200, step=1, key="DT_HPper2")
elif st.session_state.detrendornot == ind_settings.LOG_ZLAG_REGRESSION_DT:
    st.sidebar.slider("Log ZLR Smooth Period", 1, 50, step=1, key="DT_RegZLsmoothPer")

st.sidebar.subheader("Bartels Cycle Significance")
st.sidebar.checkbox("Filter with Bartels Test", key="FilterBartels")
if st.session_state.FilterBartels:
    st.sidebar.slider("Bartels: N Cycles", 1, 20, step=1, key="BartNoCycles")
    st.sidebar.slider("Bartels: Smooth Per", 1, 20, step=1, key="BartSmoothPer")
    st.sidebar.slider("Bartels: Sig Limit (%)", 0.0, 100.0, step=0.1, key="BartSigLimit")
    st.sidebar.checkbox("Sort by Bartels Sig", key="SortBartels")

st.sidebar.subheader("Miscellaneous Goertzel Settings")
st.sidebar.checkbox("Squared Amplitude", key="squaredAmp")
st.sidebar.checkbox("Use Addition for Phase", key="useAddition")
st.sidebar.checkbox("Use Cosine for Waves", key="useCosine")
st.sidebar.checkbox("Use Cycle Strength", key="UseCycleStrength")
st.sidebar.checkbox("Subtract Noise Cycles", key="SubtractNoise")
st.sidebar.checkbox("Use Specific Cycle List", key="UseCycleList")
if st.session_state.UseCycleList:
    st.sidebar.number_input("C1 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle1")
    st.sidebar.number_input("C2 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle2")
    st.sidebar.number_input("C3 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle3")
    st.sidebar.number_input("C4 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle4")
    st.sidebar.number_input("C5 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle5")

# --- Main Area: Automatic Analysis and Results Display ---
if uploaded_file is not None:
    @st.cache_data
    def get_data_from_uploaded_file_wrapper(uploaded_file_obj_wrapper):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file_obj_wrapper.getvalue()) 
            temp_file_path = tmp_file.name
        data = load_ohlc_from_csv(temp_file_path)
        try:
            os.remove(temp_file_path)
        except OSError:
            pass 
        return data

    current_uploaded_file_obj = st.session_state.get('file_uploader_widget', uploaded_file)
    ohlc_data = None
    if current_uploaded_file_obj:
        ohlc_data = get_data_from_uploaded_file_wrapper(current_uploaded_file_obj)
        if 'ohlc_data_for_length_check_processed' not in st.session_state and ohlc_data is not None:
            new_max_len = len(ohlc_data)
            if st.session_state.ohlc_data_length_for_sliders != new_max_len: # only rerun if length changed
                 st.session_state.ohlc_data_length_for_sliders = new_max_len
                 st.session_state.ohlc_data_for_length_check_processed = True 
                 st.rerun()

    if ohlc_data is None or ohlc_data.empty:
        display_name = current_uploaded_file_obj.name if current_uploaded_file_obj else "the uploaded file"
        st.error(f"Failed to load/process data from '{display_name}'. Ensure CSV has Date,Open,High,Low,Close.")
        if 'ohlc_data_for_length_check_processed' in st.session_state:
            del st.session_state.ohlc_data_for_length_check_processed
    else:
        st.subheader(f"Data Preview: {current_uploaded_file_obj.name} ({len(ohlc_data)} rows)")
        st.dataframe(ohlc_data.head(3))

        core_calc_arg_keys = [
            "source_price_type", "MaxPer", "WindowSizePast_base", "WindowSizeFuture_base",
            "detrendornot", "DT_ZLper1", "DT_ZLper2", "DT_HPper1", "DT_HPper2",
            "DT_RegZLsmoothPer", "HPsmoothPer", "ZLMAsmoothPer", "BarToCalculate",
            "FilterBartels", "BartNoCycles", "BartSmoothPer", "BartSigLimit", "SortBartels",
            "squaredAmp", "UseCycleStrength", "useAddition", "useCosine"
        ]
        core_calc_args = {}
        settings_ok = True
        for key in core_calc_arg_keys:
            if key in st.session_state:
                core_calc_args[key] = st.session_state[key]
            else:
                default_val = _get_default_settings().get(key)
                if default_val is not None:
                    st.warning(f"Warning: Core setting key '{key}' not found in session state. Using default: {default_val}")
                    core_calc_args[key] = default_val
                    st.session_state[key] = default_val 
                else:
                    st.error(f"CRITICAL ERROR: Core setting key '{key}' is missing and has no default. Cannot proceed.")
                    settings_ok = False
                    break
        
        if settings_ok:
            @st.cache_data
            def cached_core_calculation_wrapper(_ohlc_data_hashable_key, settings_tuple_wrapper):
                kwargs = dict(settings_tuple_wrapper)
                # ohlc_data is accessed from the outer scope here, ensure it's the correct one.
                return calculate_core_cycle_components(ohlc_data.copy(), **kwargs)

            core_calc_args_tuple = tuple(sorted(core_calc_args.items()))
            ohlc_data_hash_for_cache_key = None
            if ohlc_data is not None:
                 ohlc_data_hash_for_cache_key = pd.util.hash_pandas_object(ohlc_data).sum()
            
            core_components = cached_core_calculation_wrapper(ohlc_data_hash_for_cache_key, core_calc_args_tuple)

            if core_components["status"] == "error":
                st.error(f"Core Calculation Error: {core_components['message']}")
            else:
                wave_sum_arg_keys = [
                    "UseCycleList", "Cycle1", "Cycle2", "Cycle3", "Cycle4", "Cycle5",
                    "StartAtCycle", "UseTopCycles", "SubtractNoise"
                ]
                wave_sum_args = {}
                wave_settings_ok = True
                for key in wave_sum_arg_keys:
                    if key in st.session_state:
                        wave_sum_args[key] = st.session_state[key]
                    else:
                        default_val = _get_default_settings().get(key)
                        if default_val is not None:
                             st.warning(f"Warning: Wave setting key '{key}' not found in session state. Using default: {default_val}")
                             wave_sum_args[key] = default_val
                             st.session_state[key] = default_val
                        else:
                             st.error(f"CRITICAL ERROR: Wave setting key '{key}' is missing and has no default.")
                             wave_settings_ok = False
                             break
                
                if wave_settings_ok:
                    wave_results = sum_composite_waves(core_components, **wave_sum_args)
                    full_results = {**core_components, **wave_results}

                    st.subheader("Cycle Analysis Results")
                    cycle_table_df = create_cycle_table(
                        full_results['number_of_cycles'], full_results['cyclebuffer'], full_results['amplitudebuffer'],
                        full_results['phasebuffer'], full_results['cycleBartelsBuffer'],
                        st.session_state.UseCycleStrength, st.session_state.FilterBartels
                    )
                    if not cycle_table_df.empty: st.dataframe(cycle_table_df)
                    else: st.write("No significant cycles found to display in table.")

                    st.subheader(f"Indicator Plot (Last Updated: {datetime.datetime.now().strftime('%H:%M:%S.%f')})")
                    calc_bar_idx = full_results.get('calculation_bar_idx', len(ohlc_data) - 1)
                    plot_ws_past = full_results.get('current_WindowSizePast', st.session_state.WindowSizePast_base)
                    plot_ws_future = full_results.get('current_WindowSizeFuture', st.session_state.WindowSizeFuture_base)
                    analysis_sample_size = full_results.get('calculated_SampleSize', st.session_state.WindowSizePast_base)

                    fig = plot_indicator_lines(
                        ohlc_data, 
                        full_results.get('goertzel_past_wave'), 
                        full_results.get('goertzel_future_wave'),
                        calc_bar_idx, 
                        plot_ws_past, 
                        plot_ws_future, 
                        title=f"{current_uploaded_file_obj.name} - Cycle Overlay (Analysis Window: {analysis_sample_size} bars)"
                    )
                    if fig: st.pyplot(fig)
                    else: st.warning("Plot generation failed or no figure returned.")

                    st.subheader("Download Results")
                    download_settings = {**core_calc_args, **wave_sum_args}
                    download_settings["SampleSize_used_for_analysis"] = analysis_sample_size
                    download_settings["WindowSizePast_for_wave_matrix_calc"] = plot_ws_past 
                    download_settings["WindowSizeFuture_for_wave_matrix_calc"] = plot_ws_future

                    base_fn = os.path.splitext(current_uploaded_file_obj.name)[0]
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_fn = f"{base_fn}_analysis_{ts}.json"

                    past_wave_data_list = []
                    if full_results.get('goertzel_past_wave') is not None and hasattr(full_results['goertzel_past_wave'], 'tolist'):
                        past_wave_data_list = np.round(np.array(full_results['goertzel_past_wave']), 4).tolist()
                    
                    future_wave_data_list = []
                    if full_results.get('goertzel_future_wave') is not None and hasattr(full_results['goertzel_future_wave'], 'tolist'):
                        future_wave_data_list = np.round(np.array(full_results['goertzel_future_wave']), 4).tolist()

                    json_dl_data = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'settings_used': {}, # Will be populated by serializable_settings
                        'cycle_table_data': cycle_table_df.reset_index().to_dict(orient='records') if not cycle_table_df.empty else [],
                        'num_cycles': full_results.get('number_of_cycles', 0),
                        'past_wave_data': past_wave_data_list,
                        'future_wave_data': future_wave_data_list
                    }
                    try:
                        serializable_settings = {}
                        for k, v in download_settings.items():
                            if isinstance(v, np.integer): # Catches np.int32, np.int64 etc.
                                serializable_settings[k] = int(v)
                            elif isinstance(v, np.floating): # Catches np.float32, np.float64 etc.
                                serializable_settings[k] = float(v)
                            elif isinstance(v, np.bool_): # Catches np.bool_
                                serializable_settings[k] = bool(v)
                            elif isinstance(v, np.ndarray):
                                serializable_settings[k] = v.tolist()
                            else:
                                serializable_settings[k] = v
                        json_dl_data['settings_used'] = serializable_settings
                        
                        json_str = json.dumps(json_dl_data, indent=4) 
                        st.download_button("Download Data (JSON)", json_str, json_fn, "application/json", key="dl_json_btn")
                    except TypeError as e:
                        st.error(f"JSON preparation error: {e}. Check for non-standard data types (especially NumPy types).")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during JSON preparation: {e}")
else:
    st.sidebar.info("Awaiting CSV file upload...")
    if 'ohlc_data_for_length_check_processed' in st.session_state:
        del st.session_state.ohlc_data_for_length_check_processed

st.sidebar.markdown("---")
st.sidebar.markdown("Cycle Analysis Tool v2.3 (UI & Defaults Updated)")