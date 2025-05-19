# streamlit_app.py

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import tempfile
import math # Added for calculations

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
    # Start with a copy of the dictionary from settings.py
    defaults_from_file = ind_settings.DEFAULT_SETTINGS_DICT.copy()
    
    # Initialize a new dictionary for session state defaults
    # to ensure correct keys like 'WindowSizePast_base'.
    defaults_for_session = {}

    # Map keys from DEFAULT_SETTINGS_DICT to session state keys if they differ
    # For example, map "WindowSizePast" to "WindowSizePast_base"
    key_map = {
        "WindowSizePast": "WindowSizePast_base",
        "WindowSizeFuture": "WindowSizeFuture_base"
    }

    for key, value in defaults_from_file.items():
        session_key = key_map.get(key, key) # Use mapped key if exists, else original key
        defaults_for_session[session_key] = value
    
    # Add any specific defaults not in DEFAULT_SETTINGS_DICT or override if necessary
    defaults_for_session["source_price_type"] = "Close"
    
    # Ensure base window sizes are present if not mapped from a different key
    if "WindowSizePast_base" not in defaults_for_session:
        defaults_for_session["WindowSizePast_base"] = ind_settings.DEFAULT_WINDOW_SIZE_PAST
    if "WindowSizeFuture_base" not in defaults_for_session:
        defaults_for_session["WindowSizeFuture_base"] = ind_settings.DEFAULT_WINDOW_SIZE_FUTURE
        
    return defaults_for_session

# --- Initialize Session State ---
def initialize_settings_in_session_state():
    default_settings = _get_default_settings()
    for key, value in default_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_settings_in_session_state() 

# --- Sidebar ---
st.sidebar.header("Indicator Settings")

# --- Reset Button ---
if st.sidebar.button("Reset to Defaults", key="reset_settings_button"):
    default_settings = _get_default_settings()
    for key, value in default_settings.items():
        st.session_state[key] = value 
    st.rerun() 

uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"], key="file_uploader_widget")

# --- Determine max available data length for sliders ---
max_data_len = 10000 # A large default if no data is loaded
min_practical_wsp = 10 # A practical minimum for WindowSizePast_base
if 'ohlc_data_length_for_sliders' not in st.session_state:
    st.session_state.ohlc_data_length_for_sliders = max_data_len

if uploaded_file is not None and 'ohlc_data_for_length_check_processed' not in st.session_state:
    # Minimal load just to get length, not cached for full processing here
    # This is a simplified check. For very large files, consider alternatives.
    try:
        temp_df = pd.read_csv(uploaded_file, nrows=1) # Read only one row to check if readable
        uploaded_file.seek(0) # Reset file pointer
        if not temp_df.empty:
            # Full load for length check might be slow for huge files.
            # This example assumes it's acceptable or data is not excessively large.
            # A more robust way might be to get length after the main data load,
            # but that introduces a slight delay in updating slider max.
            df_len_check = pd.read_csv(uploaded_file)
            st.session_state.ohlc_data_length_for_sliders = len(df_len_check)
            uploaded_file.seek(0) # Reset again for the actual data loader
            st.session_state.ohlc_data_for_length_check_processed = True # Mark as processed
    except Exception:
        st.session_state.ohlc_data_length_for_sliders = max_data_len # Fallback on error

# --- Widgets for settings ---
st.sidebar.subheader("Input Data Source")
source_options = ["Close", "Open", "High", "Low", "(H+L)/2", "(H+L+C)/3", "(O+H+L+C)/4", "(H+L+C+C)/4"]
st.sidebar.selectbox(
    "Price Source",
    options=source_options,
    key="source_price_type"
)

st.sidebar.subheader("General Settings")

# --- WindowSizePast_base ---
current_wsp_max = st.session_state.ohlc_data_length_for_sliders if uploaded_file else max_data_len
# Ensure WindowSizePast_base doesn't exceed available data
if st.session_state.WindowSizePast_base > current_wsp_max:
    st.session_state.WindowSizePast_base = current_wsp_max

st.sidebar.slider(
    "Base Past Window Size (Analysis Lookback)",
    min_value=min_practical_wsp,
    max_value=max(min_practical_wsp, current_wsp_max), # Ensure max_value is not less than min_value
    step=10, # Or 1, depending on desired granularity
    key="WindowSizePast_base",
    help=(
        "Defines the number of historical bars (SampleSize) for the core cycle analysis. "
        "'Max Period' and 'Bar to Calculate' will be constrained by this value. "
        "Also serves as the base length for the displayed past wave."
        )
)

# --- BarToCalculate ---
# Max value for BarToCalculate must be less than WindowSizePast_base to allow for analysis window
# Goertzel needs at least 2*MaxPer data points *after* BarToCalculate within SampleSize.
# A simple constraint: BarToCalculate <= WindowSizePast_base - (2 * min_max_per_slider_val)
# For UI, let's use a simpler constraint: BarToCalculate < WindowSizePast_base
min_max_per_slider_val = 2 # Smallest practical MaxPer for UI purposes
max_bar_to_calculate = max(1, st.session_state.WindowSizePast_base - (2*min_max_per_slider_val) ) # Ensure at least a small window for analysis
if st.session_state.BarToCalculate > max_bar_to_calculate:
    st.session_state.BarToCalculate = max_bar_to_calculate

st.sidebar.number_input(
    "Bar to Calculate (Offset from end of SampleSize)",
    min_value=1,
    max_value=max(1, max_bar_to_calculate), # Ensure max_value is at least 1
    step=1,
    key="BarToCalculate",
    help="Offset from the most recent end of the 'Base Past Window Size' for Goertzel analysis. 1 means use the last bar in the window as the most recent point for Goertzel's 2*MaxPer segment."
)


# --- MaxPer (dynamically constrained) ---
min_allowable_max_per_calc = 2  # From main_calculator.py for actual calculation
# Calculate UI constraints for MaxPer slider
# Constraint 1 (Goertzel): MaxPer <= (WindowSizePast_base - BarToCalculate) / 2
ui_max_per_constraint_goertzel = math.floor(
    (st.session_state.WindowSizePast_base - st.session_state.BarToCalculate) / 2
) if (st.session_state.WindowSizePast_base - st.session_state.BarToCalculate) >= (2 * min_allowable_max_per_calc) else min_allowable_max_per_calc -1

# Constraint 2 (Bartels)
ui_max_per_constraint_bartels = float('inf')
if st.session_state.get("FilterBartels", ind_settings.DEFAULT_FILTER_BARTELS) and st.session_state.get("BartNoCycles", ind_settings.DEFAULT_BARTELS_NUM_CYCLES) > 0:
    bart_n_cycles = st.session_state.get("BartNoCycles", ind_settings.DEFAULT_BARTELS_NUM_CYCLES)
    ui_max_per_constraint_bartels = math.floor(
        st.session_state.WindowSizePast_base / bart_n_cycles
    ) if st.session_state.WindowSizePast_base >= (bart_n_cycles * min_allowable_max_per_calc) else min_allowable_max_per_calc -1

effective_max_per_for_slider = max(min_allowable_max_per_calc, min(ui_max_per_constraint_goertzel, ui_max_per_constraint_bartels))

# Clamp current MaxPer if it's out of new valid range
if st.session_state.MaxPer > effective_max_per_for_slider:
    st.session_state.MaxPer = effective_max_per_for_slider
if st.session_state.MaxPer < min_allowable_max_per_calc:
    st.session_state.MaxPer = min_allowable_max_per_calc


st.sidebar.slider(
    "Max Period (MaxPer)",
    min_value=min_allowable_max_per_calc, 
    max_value=max(min_allowable_max_per_calc, effective_max_per_for_slider), # Ensure max >= min
    step=1,
    key="MaxPer",
    help=(
        "Max cycle period to search for. This is constrained by 'Base Past Window Size', "
        "'Bar to Calculate', and 'Bartels: N Cycles' (if active). "
        "If set too high for current settings, it will be automatically adjusted or cause an error."
    )
)


st.sidebar.slider(
    "Base Future Window Size",
    min_value=20,
    max_value=500,
    step=10,
    key="WindowSizeFuture_base",
    help="Defines the base number of bars for the future wave projection. Actual projection length will be at least 2 * MaxPer."
)

# --- Other General Settings (Max value for cycle selection depends on MaxPer) ---
current_max_per_val = st.session_state.MaxPer

st.sidebar.number_input(
    "Start At Cycle (Rank)",
    min_value=1,
    max_value=max(1, current_max_per_val), # MaxPer can be small
    step=1,
    key="StartAtCycle"
)
st.sidebar.number_input(
    "Use Top Cycles (Count)",
    min_value=1,
    max_value=max(1, current_max_per_val),
    step=1,
    key="UseTopCycles"
)


st.sidebar.subheader("Source Price Processing")
detrend_mode_options = [
    ind_settings.NONE_SMTH_DT, ind_settings.ZLAGSMTH, ind_settings.HPSMTH,
    ind_settings.ZLAGSMTHDT, ind_settings.HPSMTHDT, ind_settings.LOG_ZLAG_REGRESSION_DT
]
st.sidebar.selectbox(
    "Detrending/Smoothing Mode",
    options=detrend_mode_options,
    key="detrendornot"
)

# Conditional sliders for detrending/smoothing periods
# Min/Max values for these period sliders should be practical. E.g., >=1.
# Ensure ZLMAsmoothPer and HPsmoothPer (and others) have valid session state before use.
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
    # BartNoCycles influences MaxPer's max constraint, so ensure it's positive.
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
    # Max value for cycle rank selection also depends on MaxPer
    st.sidebar.number_input("C1 Rank", 0, max(0,current_max_per_val), step=1, key="Cycle1")
    st.sidebar.number_input("C2 Rank", 0, max(0,current_max_per_val), step=1, key="Cycle2")
    st.sidebar.number_input("C3 Rank", 0, max(0,current_max_per_val), step=1, key="Cycle3")
    st.sidebar.number_input("C4 Rank", 0, max(0,current_max_per_val), step=1, key="Cycle4")
    st.sidebar.number_input("C5 Rank", 0, max(0,current_max_per_val), step=1, key="Cycle5")

# --- Main Area: Automatic Analysis and Results Display ---
if uploaded_file is not None:
    @st.cache_data
    def get_data_from_uploaded_file_wrapper(uploaded_file_obj_wrapper):
        # uploaded_file_obj_wrapper is the actual UploadedFile object
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file_obj_wrapper.getvalue()) # Use getvalue()
            temp_file_path = tmp_file.name
        data = load_ohlc_from_csv(temp_file_path)
        try:
            os.remove(temp_file_path)
        except OSError:
            pass # Ignore error if file already deleted or other issue
        return data

    # Get the uploaded file object from session state if it exists (due to rerun)
    # or from the direct uploader variable if it's a new upload.
    current_uploaded_file_obj = st.session_state.get('file_uploader_widget', uploaded_file)

    ohlc_data = None
    if current_uploaded_file_obj:
        ohlc_data = get_data_from_uploaded_file_wrapper(current_uploaded_file_obj)
        # Update slider max based on newly loaded data if not done before
        if 'ohlc_data_for_length_check_processed' not in st.session_state and ohlc_data is not None:
            st.session_state.ohlc_data_length_for_sliders = len(ohlc_data)
            st.session_state.ohlc_data_for_length_check_processed = True
            st.rerun() # Rerun to update slider ranges with data length

    if ohlc_data is None or ohlc_data.empty:
        display_name = current_uploaded_file_obj.name if current_uploaded_file_obj else "the uploaded file"
        st.error(f"Failed to load/process data from '{display_name}'. Ensure CSV has Date,Open,High,Low,Close.")
        # Reset the processed flag if data load fails, to allow re-check on next upload
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
                # This fallback should ideally not be hit if initialize_settings is robust
                default_val = _get_default_settings().get(key) # _get_default_settings now uses _base keys
                if default_val is None and key in ["WindowSizePast_base", "WindowSizeFuture_base"]: # explicit check
                     # This case should ideally be covered by _get_default_settings ensuring these keys.
                     if key == "WindowSizePast_base": default_val = ind_settings.DEFAULT_WINDOW_SIZE_PAST
                     if key == "WindowSizeFuture_base": default_val = ind_settings.DEFAULT_WINDOW_SIZE_FUTURE
                
                if default_val is not None:
                    st.warning(f"Warning: Core setting key '{key}' not found in session state. Using default: {default_val}")
                    core_calc_args[key] = default_val
                    st.session_state[key] = default_val # Add to session state
                else:
                    st.error(f"CRITICAL ERROR: Core setting key '{key}' is missing and has no default. Cannot proceed.")
                    settings_ok = False
                    break
        
        if settings_ok:
            @st.cache_data
            def cached_core_calculation_wrapper(data_df_tuple_wrapper, settings_tuple_wrapper):
                # Reconstruct DataFrame from tuple of tuples if necessary, or ensure data_df is hashable
                # For simplicity, assuming data_df can be cached or its hash doesn't change unless content does.
                # If ohlc_data is a pandas DataFrame, it might need special handling for @st.cache_data
                # A common pattern is to pass a hash or a tuple representation.
                # Here, we assume ohlc_data itself is passed and Streamlit handles its caching.
                # data_df = pd.DataFrame(data_df_tuple_wrapper[0], columns=data_df_tuple_wrapper[1]) # Example if passing tuple
                kwargs = dict(settings_tuple_wrapper)
                return calculate_core_cycle_components(ohlc_data.copy(), **kwargs)

            core_calc_args_tuple = tuple(sorted(core_calc_args.items()))
            # Pass ohlc_data directly; Streamlit's caching should handle it.
            # If caching issues arise with DataFrame, convert to a hashable format.
            core_components = cached_core_calculation_wrapper(ohlc_data, core_calc_args_tuple)

            if core_components["status"] == "error":
                st.error(f"Core Calculation Error: {core_components['message']}")
            else:
                wave_sum_arg_keys = [
                    "UseCycleList", "Cycle1", "Cycle2", "Cycle3", "Cycle4", "Cycle5",
                    "StartAtCycle", "UseTopCycles", "SubtractNoise"
                ]
                wave_sum_args = {}
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
                             # Potentially stop further processing here
                             break 


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
                
                # Use the 'current_WindowSizePast/Future' from results for plotting, as these are adjusted ones by main_calculator
                plot_ws_past = full_results.get('current_WindowSizePast', st.session_state.WindowSizePast_base)
                plot_ws_future = full_results.get('current_WindowSizeFuture', st.session_state.WindowSizeFuture_base)
                # The SampleSize used for analysis is now directly st.session_state.WindowSizePast_base
                analysis_sample_size = st.session_state.WindowSizePast_base


                fig = plot_indicator_lines(
                    ohlc_data, 
                    full_results.get('goertzel_past_wave'), 
                    full_results.get('goertzel_future_wave'),
                    calc_bar_idx, 
                    plot_ws_past, # This is the length of the plotted past wave
                    plot_ws_future, # This is the length of the plotted future wave
                    title=f"{current_uploaded_file_obj.name} - Cycle Overlay (Analysis Window: {analysis_sample_size} bars)"
                )
                if fig: st.pyplot(fig)
                else: st.warning("Plot generation failed or no figure returned.")

                st.subheader("Download Results")
                download_settings = {**core_calc_args, **wave_sum_args}
                # Add the actual SampleSize used for analysis
                download_settings["SampleSize_used_for_analysis"] = analysis_sample_size
                # The 'WindowSizePast_base' from UI is already in core_calc_args.
                # 'current_WindowSizePast' from results is the actual length of goeWorkPast matrix.
                download_settings["WindowSizePast_for_wave_matrix_calc"] = plot_ws_past 
                download_settings["WindowSizeFuture_for_wave_matrix_calc"] = plot_ws_future


                base_fn = os.path.splitext(current_uploaded_file_obj.name)[0]
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                json_fn = f"{base_fn}_analysis_{ts}.json"

                # Ensure wave data is serializable
                past_wave_data_list = []
                if full_results.get('goertzel_past_wave') is not None and hasattr(full_results['goertzel_past_wave'], 'tolist'):
                    past_wave_data_list = np.round(np.array(full_results['goertzel_past_wave']), 4).tolist()
                
                future_wave_data_list = []
                if full_results.get('goertzel_future_wave') is not None and hasattr(full_results['goertzel_future_wave'], 'tolist'):
                    future_wave_data_list = np.round(np.array(full_results['goertzel_future_wave']), 4).tolist()

                json_dl_data = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'settings_used': download_settings,
                    'cycle_table_data': cycle_table_df.reset_index().to_dict(orient='records') if not cycle_table_df.empty else [],
                    'num_cycles': full_results.get('number_of_cycles', 0),
                    'past_wave_data': past_wave_data_list,
                    'future_wave_data': future_wave_data_list
                }
                try:
                    # Convert numpy types in settings to native Python types for JSON serialization
                    serializable_settings = {}
                    for k, v in download_settings.items():
                        if isinstance(v, (np.integer, np.int_)):
                            serializable_settings[k] = int(v)
                        elif isinstance(v, (np.floating, np.float_)):
                            serializable_settings[k] = float(v)
                        elif isinstance(v, np.ndarray):
                            serializable_settings[k] = v.tolist()
                        else:
                            serializable_settings[k] = v
                    json_dl_data['settings_used'] = serializable_settings

                    json_str = json.dumps(json_dl_data, indent=4) # allow_nan=False removed for flexibility, handle NaNs if they appear
                    st.download_button("Download Data (JSON)", json_str, json_fn, "application/json", key="dl_json_btn")
                except TypeError as e:
                    st.error(f"JSON preparation error: {e}. Check for non-standard data types.")
                except Exception as e:
                    st.error(f"An unexpected error occurred during JSON preparation: {e}")
else:
    st.sidebar.info("Awaiting CSV file upload...")
    # Reset the flag if no file is uploaded, so it checks next time
    if 'ohlc_data_for_length_check_processed' in st.session_state:
        del st.session_state.ohlc_data_for_length_check_processed


st.sidebar.markdown("---")
st.sidebar.markdown("Cycle Analysis Tool v2.1 (Updated Window Logic)")