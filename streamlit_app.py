# streamlit_app.py

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import tempfile

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
    defaults = ind_settings.DEFAULT_SETTINGS_DICT.copy()
    defaults["source_price_type"] = "Close" # Default defined in the previous version
    defaults["WindowSizePast_base"] = ind_settings.DEFAULT_WINDOW_SIZE_PAST
    defaults["WindowSizeFuture_base"] = ind_settings.DEFAULT_WINDOW_SIZE_FUTURE
    # Ensure all keys that have widgets are present in the defaults from ind_settings
    # or are added here if they are app-specific session state keys.
    return defaults

# --- Initialize Session State ---
def initialize_settings_in_session_state():
    default_settings = _get_default_settings()
    for key, value in default_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_settings_in_session_state() # This ensures keys are in session_state before widgets are created

# --- Sidebar ---
st.sidebar.header("Indicator Settings")

# --- Reset Button ---
if st.sidebar.button("Reset to Defaults", key="reset_settings_button"):
    default_settings = _get_default_settings()
    for key, value in default_settings.items():
        st.session_state[key] = value # Directly overwrite session state with defaults
    st.rerun() # Re-run the app from the top

uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"], key="file_uploader_widget")

# --- Widgets for settings ---
# For widgets with keys corresponding to initialized session state variables,
# Streamlit will use st.session_state[key] for their value.

st.sidebar.subheader("Input Data Source")
source_options = ["Close", "Open", "High", "Low", "(H+L)/2", "(H+L+C)/3", "(O+H+L+C)/4", "(H+L+C+C)/4"]
st.sidebar.selectbox(
    "Price Source",
    options=source_options,
    key="source_price_type"
)

st.sidebar.subheader("General Settings")
st.sidebar.slider(
    "Max Period (MaxPer)",
    min_value=10,
    max_value=300,
    step=1,
    key="MaxPer"
)

# Example for base window sizes if they were to be user-editable
# st.sidebar.slider("Base Past Window", 10, 500, step=10, key="WindowSizePast_base")
# st.sidebar.slider("Base Future Window", 10, 500, step=10, key="WindowSizeFuture_base")

st.sidebar.number_input(
    "Start At Cycle",
    min_value=1,
    max_value=st.session_state.MaxPer,
    step=1,
    key="StartAtCycle"
)
st.sidebar.number_input(
    "Use Top Cycles",
    min_value=1,
    max_value=st.session_state.MaxPer,
    step=1,
    key="UseTopCycles"
)
st.sidebar.number_input(
    "Bar to Calculate (Offset)",
    min_value=1,
    max_value=100,
    step=1,
    key="BarToCalculate"
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
    st.sidebar.number_input("C1 Rank", 0, st.session_state.MaxPer, step=1, key="Cycle1")
    st.sidebar.number_input("C2 Rank", 0, st.session_state.MaxPer, step=1, key="Cycle2")
    st.sidebar.number_input("C3 Rank", 0, st.session_state.MaxPer, step=1, key="Cycle3")
    st.sidebar.number_input("C4 Rank", 0, st.session_state.MaxPer, step=1, key="Cycle4")
    st.sidebar.number_input("C5 Rank", 0, st.session_state.MaxPer, step=1, key="Cycle5")

# --- Main Area: Automatic Analysis and Results Display ---
if uploaded_file is not None:
    @st.cache_data
    def get_data_from_uploaded_file_wrapper(uploaded_file_obj_wrapper):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file_obj_wrapper.getbuffer())
            temp_file_path = tmp_file.name
        data = load_ohlc_from_csv(temp_file_path)
        try:
            os.remove(temp_file_path)
        except OSError:
            pass
        return data

    ohlc_data = get_data_from_uploaded_file_wrapper(uploaded_file)

    if ohlc_data is None or ohlc_data.empty:
        st.error(f"Failed to load/process data from '{uploaded_file.name}'. Ensure CSV has Date,Open,High,Low,Close.")
    else:
        st.subheader(f"Data Preview: {uploaded_file.name} ({len(ohlc_data)} rows)")
        st.dataframe(ohlc_data.head(3))

        core_calc_arg_keys = [
            "source_price_type", "MaxPer", "WindowSizePast_base", "WindowSizeFuture_base",
            "detrendornot", "DT_ZLper1", "DT_ZLper2", "DT_HPper1", "DT_HPper2",
            "DT_RegZLsmoothPer", "HPsmoothPer", "ZLMAsmoothPer", "BarToCalculate",
            "FilterBartels", "BartNoCycles", "BartSmoothPer", "BartSigLimit", "SortBartels",
            "squaredAmp", "UseCycleStrength", "useAddition", "useCosine"
        ]
        core_calc_args = {}
        for key in core_calc_arg_keys:
            if key in st.session_state:
                core_calc_args[key] = st.session_state[key]
            else:
                # Fallback, though initialize_settings_in_session_state should prevent this
                default_val = _get_default_settings().get(key)
                st.warning(f"Warning: Core setting key '{key}' not found in session state. Using default: {default_val}")
                core_calc_args[key] = default_val
                if key not in st.session_state: # If truly missing, add to session state to prevent repeated warnings
                    st.session_state[key] = default_val


        @st.cache_data
        def cached_core_calculation_wrapper(data_df, settings_tuple):
            kwargs = dict(settings_tuple)
            return calculate_core_cycle_components(data_df.copy(), **kwargs)

        core_calc_args_tuple = tuple(sorted(core_calc_args.items()))
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
                    # Fallback for wave sum args
                    default_val = _get_default_settings().get(key)
                    st.warning(f"Warning: Wave setting key '{key}' not found in session state. Using default: {default_val}")
                    wave_sum_args[key] = default_val
                    if key not in st.session_state:
                         st.session_state[key] = default_val


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

            fig = plot_indicator_lines(
                ohlc_data, full_results['goertzel_past_wave'], full_results['goertzel_future_wave'],
                calc_bar_idx, plot_ws_past, plot_ws_future,
                title=f"{uploaded_file.name} - Cycle Overlay"
            )
            if fig: st.pyplot(fig)
            else: st.warning("Plot generation failed or no figure returned.")

            st.subheader("Download Results")
            download_settings = {**core_calc_args, **wave_sum_args}
            download_settings["WindowSizePast_from_UI_if_different"] = st.session_state.WindowSizePast_base
            download_settings["WindowSizeFuture_from_UI_if_different"] = st.session_state.WindowSizeFuture_base

            base_fn = os.path.splitext(uploaded_file.name)[0]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_fn = f"{base_fn}_analysis_{ts}.json"

            past_wave_data = np.round(np.array(full_results['goertzel_past_wave']), 4).tolist() if full_results.get('goertzel_past_wave') is not None else []
            future_wave_data = np.round(np.array(full_results['goertzel_future_wave']), 4).tolist() if full_results.get('goertzel_future_wave') is not None else []

            json_dl_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'settings_used': download_settings,
                'cycle_table_data': cycle_table_df.reset_index().to_dict(orient='records') if not cycle_table_df.empty else [],
                'num_cycles': full_results.get('number_of_cycles', 0),
                'past_wave_data': past_wave_data,
                'future_wave_data': future_wave_data
            }
            try:
                json_str = json.dumps(json_dl_data, indent=4, allow_nan=False)
                st.download_button("Download Data (JSON)", json_str, json_fn, "application/json", key="dl_json_btn")
            except TypeError as e:
                st.error(f"JSON preparation error: {e}. This might be due to non-standard data types (e.g. numpy types).")
            except Exception as e:
                st.error(f"An unexpected error occurred during JSON preparation: {e}")
else:
    st.sidebar.info("Awaiting CSV file upload...")

st.sidebar.markdown("---")
st.sidebar.markdown("Cycle Analysis Tool v2")
