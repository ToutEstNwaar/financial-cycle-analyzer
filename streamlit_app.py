# streamlit_app.py

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(layout="wide", page_title="Cycle Indicator Analysis")

# Suppress deprecation warning from streamlit-cookies-manager dependency
import warnings
warnings.filterwarnings("ignore", message=".*st.cache.*deprecated.*", category=DeprecationWarning)

import sys
import os
import pandas as pd
import numpy as np
import datetime
import pytz
import json
import tempfile
import math



# --- Python Path Modification ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Python Path Modification ---

# Milestone 3: Data Integration imports
from data_integration import CryptoDataFetcher, DatabaseManager, DataExporter

from indicator_logic import settings as ind_settings
from indicator_logic.data_loader import load_ohlc_from_csv
from indicator_logic.main_calculator import calculate_core_cycle_components, sum_composite_waves
from utils.plotting import plot_indicator_lines, create_cycle_table

st.title("Financial Cycle Indicator Analysis :chart_with_upwards_trend:")
st.markdown("Upload OHLCV data (CSV). Analysis updates automatically as settings change.")

# Settings constants
SETTINGS_FILENAME = "financial_cycle_analyzer_settings.json"
AUTO_SAVE_SETTINGS_FILE = "last_session_settings.json"

# --- Helper function to get default settings ---
def _get_default_settings():
    # DEFAULT_SETTINGS_DICT in settings.py now uses *_base keys for window sizes
    defaults = ind_settings.DEFAULT_SETTINGS_DICT.copy()
    defaults["source_price_type"] = "Close"
    return defaults

# --- Settings Persistence Functions ---
def get_settings_to_persist():
    """Get current settings that should be persisted to cookies"""
    settings_to_save = {}
    
    # Core indicator settings
    for key in _get_default_settings().keys():
        if key in st.session_state:
            settings_to_save[key] = st.session_state[key]
    
    # Data source settings
    data_source_keys = [
        'data_source', 'selected_exchange', 'selected_symbol', 'selected_timeframe',
        'custom_symbol_input', 'live_data_is_stale'
    ]
    for key in data_source_keys:
        if key in st.session_state:
            settings_to_save[key] = st.session_state[key]
    
    # Special handling for user intended window size
    if 'user_intended_WindowSizePast_base' in st.session_state:
        settings_to_save['user_intended_WindowSizePast_base'] = st.session_state['user_intended_WindowSizePast_base']
    
    return settings_to_save

def auto_save_settings_to_file():
    """Automatically save current settings to a local file"""
    try:
        settings_to_save = get_settings_to_persist()
        with open(AUTO_SAVE_SETTINGS_FILE, 'w') as f:
            json.dump(settings_to_save, f, indent=4)
        # Don't show success message for auto-save to avoid UI clutter
    except Exception as e:
        # Silently fail for auto-save to avoid disrupting user experience
        pass

def auto_load_settings_from_file():
    """Automatically load settings from local file if it exists"""
    try:
        if os.path.exists(AUTO_SAVE_SETTINGS_FILE):
            with open(AUTO_SAVE_SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
            
            # Define the set of keys that are permissible to load
            valid_keys = set(_get_default_settings().keys()) | {
                'data_source', 'selected_exchange', 'selected_symbol', 'selected_timeframe',
                'custom_symbol_input', 'live_data_is_stale', 
                'user_intended_WindowSizePast_base'
            }

            settings_loaded_count = 0
            for key, value in loaded_settings.items():
                if key == 'user_intended_WindowSizePast_base':
                    st.session_state['user_intended_WindowSizePast_base'] = value
                    st.session_state['WindowSizePast_base'] = value 
                    settings_loaded_count += 1
                elif key == 'custom_symbol_input':
                    st.session_state['custom_symbol_input'] = str(value).strip().upper()
                    settings_loaded_count += 1
                elif key in valid_keys:
                    st.session_state[key] = value
                    settings_loaded_count += 1
            
            if settings_loaded_count > 0:
                st.toast(f"✅ Restored {settings_loaded_count} settings from previous session", icon="🔄")
                
    except Exception as e:
        # Silently fail for auto-load to avoid disrupting app startup
        pass



# --- Initialize Session State ---
def initialize_settings_in_session_state():
    default_settings = _get_default_settings()
    
    # First set defaults for any missing keys
    for key, value in default_settings.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Then try to load previous session settings (will override defaults)
    auto_load_settings_from_file()

# Initialize Milestone 3 components
def initialize_milestone3_components():
    """Initialize data integration components for Milestone 3"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'data_exporter' not in st.session_state:
        st.session_state.data_exporter = DataExporter()

    if 'data_source' not in st.session_state:
        st.session_state.data_source = 'file'  # 'file' or 'live'
    if 'selected_exchange' not in st.session_state:
        st.session_state.selected_exchange = 'binance'
    if 'selected_symbol' not in st.session_state:
        # Ensure the default symbol is in the tradable pairs list
        if hasattr(st.session_state, 'tradable_pairs_list') and st.session_state.tradable_pairs_list:
            st.session_state.selected_symbol = st.session_state.tradable_pairs_list[0]
        else:
            st.session_state.selected_symbol = 'BTC/USDT'
    if 'selected_timeframe' not in st.session_state:
        st.session_state.selected_timeframe = '1d'
    if 'live_data_is_stale' not in st.session_state:
        st.session_state.live_data_is_stale = True  # Initially stale until first fetch
    if 'custom_symbol_input' not in st.session_state:
        st.session_state.custom_symbol_input = ""  # Initialize custom symbol input

# --- Load Tradable Pairs Configuration ---
def load_tradable_pairs(config_path="tradable_pairs.json"):
    """Loads the predefined list of tradable pairs from a JSON config file."""
    try:
        with open(config_path, 'r') as f:
            pairs = json.load(f)
        if not isinstance(pairs, list) or not all(isinstance(pair, str) for pair in pairs):
            st.error(f"Error: '{config_path}' should contain a JSON list of strings.")
            return ['BTC/USDT'] # Fallback
        return pairs
    except FileNotFoundError:
        st.error(f"Error: Tradable pairs configuration file '{config_path}' not found.")
        return ['BTC/USDT'] # Fallback
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from '{config_path}'.")
        return ['BTC/USDT'] # Fallback

# Initialize tradable pairs in session state if not already present
if 'tradable_pairs_list' not in st.session_state:
    st.session_state.tradable_pairs_list = load_tradable_pairs()

# Initialize settings and components
initialize_settings_in_session_state()
initialize_milestone3_components()



# --- Utility Functions ---
def mark_live_data_stale():
    """Mark live data as stale when selection parameters change"""
    st.session_state.live_data_is_stale = True

def on_setting_change():
    """Callback for when any setting changes - auto-save settings"""
    auto_save_settings_to_file()

def mark_live_data_stale_and_save():
    """Mark live data as stale and auto-save settings"""
    mark_live_data_stale()
    auto_save_settings_to_file()
def get_next_candle_time(timeframe, bars_ahead=1):
    """
    Calculate when the next candle(s) will open based on current time and timeframe.
    
    Args:
        timeframe: String like '1m', '5m', '15m', '1h', '4h', '1d', '1w'
        bars_ahead: Number of bars ahead to calculate (default 1)
    
    Returns:
        datetime: UTC timezone-aware datetime of when the target candle opens
    """
    # Use UTC timezone for consistency
    now = datetime.datetime.now(pytz.utc)
    
    timeframe_minutes = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, 
        '2h': 120, '4h': 240, '6h': 360, '8h': 480, '12h': 720,
        '1d': 1440, '1w': 10080
    }
    
    tf_minutes = timeframe_minutes.get(timeframe, 60)
    
    if tf_minutes >= 1440:  # Daily or weekly
        if tf_minutes == 1440:  # Daily
            # Daily candles typically open at 00:00 UTC
            next_candle = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if next_candle <= now:
                next_candle += datetime.timedelta(days=1)
            # Add additional days for bars_ahead > 1
            if bars_ahead > 1:
                next_candle += datetime.timedelta(days=bars_ahead-1)
        else:  # Weekly (1w)
            # Weekly candles typically open on Monday 00:00 UTC
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                # It's Monday, check if we're past 00:00
                if now.hour > 0 or now.minute > 0 or now.second > 0:
                    days_until_monday = 7  # Next Monday
            
            next_candle = now.replace(hour=0, minute=0, second=0, microsecond=0)
            next_candle += datetime.timedelta(days=days_until_monday)
            # Add additional weeks for bars_ahead > 1
            if bars_ahead > 1:
                next_candle += datetime.timedelta(weeks=bars_ahead-1)
    else:
        # For intraday timeframes, calculate next alignment
        # Round current time down to the nearest timeframe boundary
        minutes_since_midnight = now.hour * 60 + now.minute
        candles_since_midnight = minutes_since_midnight // tf_minutes
        
        # Calculate the next candle boundary
        next_candle_minutes = (candles_since_midnight + bars_ahead) * tf_minutes
        
        # Create the next candle time
        next_candle = now.replace(hour=0, minute=0, second=0, microsecond=0)
        next_candle += datetime.timedelta(minutes=next_candle_minutes)
        
        # If we've gone past the current day, adjust to next day(s)
        while next_candle.day != now.day and next_candle < now:
            next_candle += datetime.timedelta(days=1)
    
    return next_candle

# --- Sidebar Rendering Function ---
def render_sidebar():
    """Render the complete sidebar with all settings and data source controls"""
    # --- Indicator Settings Header ---
    st.sidebar.header("Indicator Settings")

    if st.sidebar.button("Reset to Defaults", key="reset_settings_button_sidebar"):
        default_settings = _get_default_settings()
        for key, value in default_settings.items():
            st.session_state[key] = value 
        auto_save_settings_to_file()  # Auto-save after reset
        st.rerun()

    # Download Settings Button
    settings_to_download = get_settings_to_persist()
    settings_json_string = json.dumps(settings_to_download, indent=4)
    
    st.sidebar.download_button(
        label="📥 Download Current Settings",
        data=settings_json_string,
        file_name=SETTINGS_FILENAME,
        mime="application/json",
        key="download_settings_btn",
        help="Download your current settings as a JSON file"
    )

    # Upload Settings File
    uploaded_settings_file = st.sidebar.file_uploader(
        "📤 Upload Settings File (.json)", 
        type=["json"],
        key="upload_settings_widget",
        help="Upload a previously downloaded settings file to restore configuration"
    )

 

    # --- Milestone 3: Data Source Selection ---
    st.sidebar.header("Data Source")

    data_source_option = st.sidebar.radio(
        "Choose Data Source",
        options=["Upload File", "Live Crypto Data"],
        index=0 if st.session_state.data_source == 'file' else 1,
        key="data_source_radio",
        on_change=on_setting_change
    )

    # Update session state based on radio selection
    if data_source_option == "Upload File":
        st.session_state.data_source = 'file'
        uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"], key="file_uploader_widget")
        
    elif data_source_option == "Live Crypto Data":
        st.session_state.data_source = 'live'
        uploaded_file = None  # Clear file upload when using live data
        
        # Exchange selection
        exchange_options = ['binance', 'coinbase', 'kraken', 'bybit']
        st.sidebar.selectbox(
            "Exchange",
            options=exchange_options,
            key="selected_exchange",
            on_change=mark_live_data_stale_and_save
        )
        
        # Symbol selection
        # Use the loaded list from session state
        symbol_options = st.session_state.tradable_pairs_list 
        
        st.sidebar.selectbox(
            "Trading Pair",
            options=symbol_options,
            key="selected_symbol",
            on_change=mark_live_data_stale_and_save
        )
        
        # Custom symbol input field
        st.sidebar.text_input(
            "Or Enter Custom Pair (e.g., ETH/BTC)", 
            key="custom_symbol_input",
            on_change=mark_live_data_stale_and_save # Typing a new custom symbol also makes data stale
        )
        
        # Timeframe selection
        timeframe_options = ['1m', '5m', '15m', '1h', '4h', '1d', '1w']
        st.sidebar.selectbox(
            "Timeframe",
            options=timeframe_options,
            key="selected_timeframe",
            on_change=mark_live_data_stale_and_save
        )
        
        # Days back for historical data
        days_back = st.sidebar.slider(
            "Historical Days",
            min_value=30,
            max_value=365,
            value=90,
            step=10,
            key="days_back_slider",
            help="Number of days of historical data to fetch"
        )
        
        # Fetch data button
        if st.sidebar.button("Fetch Live Data", key="fetch_live_data_btn"):
            # Determine the effective symbol to use
            effective_symbol = st.session_state.selected_symbol 
            custom_input_processed = st.session_state.get("custom_symbol_input", "").strip().upper()
            if custom_input_processed:
                effective_symbol = custom_input_processed
            
            with st.spinner(f"Fetching {effective_symbol} data from {st.session_state.selected_exchange}..."):
                try:
                    fetcher = CryptoDataFetcher(exchange_name=st.session_state.selected_exchange)
                    live_data = fetcher.get_historical_data(
                        symbol=effective_symbol,
                        timeframe=st.session_state.selected_timeframe,
                        days_back=days_back
                    )
                    
                    if not live_data.empty:
                        # Store in database
                        success = st.session_state.db_manager.store_price_data(
                            live_data, 
                            effective_symbol, 
                            st.session_state.selected_timeframe
                        )
                        
                        if success:
                            st.session_state.live_data = live_data
                            st.session_state.ohlc_data_length_for_sliders = len(live_data)
                            st.session_state.live_data_is_stale = False  # Data is now fresh
                            st.sidebar.success(f"✅ Fetched {len(live_data)} candles")
                            st.sidebar.info(f"Price range: ${live_data['Close'].min():.2f} - ${live_data['Close'].max():.2f}")
                        else:
                            st.sidebar.error("Failed to store data in database")
                    else:
                        st.sidebar.error("No data received from exchange")
                        
                except Exception as e:
                    st.sidebar.error(f"Error fetching data: {str(e)}")

    # Legacy file uploader (only shown when file source is selected)
    if st.session_state.data_source == 'file':
        pass  # uploaded_file is already set above
    else:
        uploaded_file = None

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
    st.sidebar.selectbox("Price Source", options=source_options, key="source_price_type", on_change=on_setting_change)

    st.sidebar.subheader("General Settings")

    # --- Past Window Size ---
    # Consider both file uploads and live data for determining max window size
    has_data = (uploaded_file is not None) or (st.session_state.data_source == 'live' and 'live_data' in st.session_state and not st.session_state.live_data.empty)

    current_data_length = st.session_state.ohlc_data_length_for_sliders if has_data else 0

    # Initialize user_intended_WindowSizePast_base if it doesn't exist
    # Default it to the initial value of WindowSizePast_base (from _get_default_settings)
    if 'user_intended_WindowSizePast_base' not in st.session_state:
        st.session_state.user_intended_WindowSizePast_base = st.session_state.WindowSizePast_base

    # Determine the slider's max value based SOLELY on current data length and practical min
    # Ensure max_value is always greater than min_value for Streamlit slider
    slider_max_value = max(min_practical_window_size + 1, current_data_length)

    # Determine the slider's current value
    # It should be the user's last intended value, but clamped by the slider's current possible max and min
    current_slider_value = st.session_state.user_intended_WindowSizePast_base
    if current_slider_value > slider_max_value:
        current_slider_value = slider_max_value
    if current_slider_value < min_practical_window_size:
        current_slider_value = min_practical_window_size

    # Update the main session state variable that the analysis uses
    st.session_state.WindowSizePast_base = current_slider_value

    # Callback for when the user *manually* changes the slider
    def on_window_size_change():
        # When user interacts, their new value becomes the "intended" value
        st.session_state.user_intended_WindowSizePast_base = st.session_state.WindowSizePast_base_slider_widget
        # Also update the main value immediately
        st.session_state.WindowSizePast_base = st.session_state.WindowSizePast_base_slider_widget
        # Auto-save settings when user changes window size
        auto_save_settings_to_file()

    st.sidebar.slider(
        "Past Window Size (Analysis Lookback)", # Label updated
        min_value=min_practical_window_size,
        max_value=slider_max_value, # This is now strictly based on data length
        value=current_slider_value, # Use the calculated current_slider_value
        step=10, 
        key="WindowSizePast_base_slider_widget", # Use a distinct key for the widget itself
        on_change=on_window_size_change,
        help=(
            "Defines the number of historical bars (SampleSize) for the core cycle analysis. "
            "Backend validation will check compatibility with 'Max Period' and 'Bar to Calculate'. "
            "Also serves as the base length for the displayed past wave."
            )
    )

    # Ensure the main analysis key is also updated if the callback hasn't run yet (e.g. on first load after constraint)
    # This might be redundant if the value=current_slider_value already handles it, but can be a safeguard
    if st.session_state.WindowSizePast_base != st.session_state.get("WindowSizePast_base_slider_widget", current_slider_value):
        st.session_state.WindowSizePast_base = st.session_state.get("WindowSizePast_base_slider_widget", current_slider_value)

    # --- MaxPer ---
    st.sidebar.slider(
        "Max Period (MaxPer)",
        min_value=min_allowable_max_per, 
        max_value=500, # Fixed max as per user request
        step=1,
        key="MaxPer",
        on_change=on_setting_change,
        help=(
            "Max cycle period to search for. Backend validation will ensure compatibility with "
            "'Past Window Size' and other settings. Error will be shown if incompatible."
        )
    )

    # --- Future Window Size ---
    st.sidebar.slider(
        "Future Window Size", # Label updated
        min_value=min_practical_window_size,
        max_value=1000, # Fixed max as per user request
        step=10,
        key="WindowSizeFuture_base", # Key remains _base
        on_change=on_setting_change,
        help="Defines the base number of bars for the future wave projection. Actual projection length will be at least 2 * MaxPer."
    )

    # --- Bar to Calculate (Moved after Future Window Size) ---
    st.sidebar.number_input(
        "Bar to Calculate (Offset)",
        min_value=1,
        max_value=100, # Fixed practical UI limit
        step=1,
        key="BarToCalculate",
        on_change=on_setting_change,
        help="Offset from the most recent end of the 'Past Window Size' for Goertzel analysis. Backend will validate against 'Past Window Size'."
    )

    # --- Other General Settings ---
    current_max_per_val_for_cycle_selection = st.session_state.MaxPer # MaxPer can now be up to 500

    st.sidebar.number_input(
        "Start At Cycle (Rank)",
        min_value=1,
        max_value=max(1, current_max_per_val_for_cycle_selection), 
        step=1,
        key="StartAtCycle",
        on_change=on_setting_change
    )
    st.sidebar.number_input(
        "Use Top Cycles (Count)",
        min_value=1,
        max_value=max(1, current_max_per_val_for_cycle_selection),
        step=1,
        key="UseTopCycles",
        on_change=on_setting_change
    )

    st.sidebar.subheader("Source Price Processing")
    detrend_mode_options = [
        ind_settings.NONE_SMTH_DT, ind_settings.ZLAGSMTH, ind_settings.HPSMTH,
        ind_settings.ZLAGSMTHDT, ind_settings.HPSMTHDT, ind_settings.LOG_ZLAG_REGRESSION_DT
    ]
    st.sidebar.selectbox("Detrending/Smoothing Mode",options=detrend_mode_options,key="detrendornot",on_change=on_setting_change)

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

    # --- Sidebar Footer ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("Cycle Analysis Tool v4.0 (Milestone 4: Settings Persistence)")
    st.sidebar.markdown("**Enhanced with Live Data & Auto-Recalculation**")
    
    # Show auto-save status
    if os.path.exists(AUTO_SAVE_SETTINGS_FILE):
        try:
            mod_time = os.path.getmtime(AUTO_SAVE_SETTINGS_FILE)
            last_save = datetime.datetime.fromtimestamp(mod_time).strftime('%H:%M:%S')
            st.sidebar.caption(f"⚡ Settings auto-saved at {last_save}")
        except:
            st.sidebar.caption("⚡ Settings auto-save enabled")
    
    return uploaded_file

# --- Main Area: Data Loading and Preview ---
def load_and_preview_data(uploaded_file_obj):
    """Load data and show preview with stale flag logic for live data"""
    ohlc_data_internal = None
    data_source_name_internal = ""
    show_preview_for_data_internal = None

    # --- Data Source Logic ---
    if st.session_state.data_source == 'file' and uploaded_file_obj is not None:
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

        current_uploaded_file_obj = st.session_state.get('file_uploader_widget', uploaded_file_obj)
        if current_uploaded_file_obj:
            ohlc_data_internal = get_data_from_uploaded_file_wrapper(current_uploaded_file_obj)
            data_source_name_internal = current_uploaded_file_obj.name
            if 'ohlc_data_for_length_check_processed' not in st.session_state and ohlc_data_internal is not None:
                new_max_len = len(ohlc_data_internal)
                if st.session_state.ohlc_data_length_for_sliders != new_max_len: # only rerun if length changed
                     st.session_state.ohlc_data_length_for_sliders = new_max_len
                     st.session_state.ohlc_data_for_length_check_processed = True 
                     st.rerun()
            
            if ohlc_data_internal is not None:
                show_preview_for_data_internal = ohlc_data_internal.reset_index()
    
    elif st.session_state.data_source == 'live':
        # Determine the effective symbol for display
        effective_symbol = st.session_state.selected_symbol 
        custom_input_processed = st.session_state.get("custom_symbol_input", "").strip().upper()
        if custom_input_processed:
            effective_symbol = custom_input_processed
        data_source_name_internal = f"{effective_symbol} ({st.session_state.selected_exchange})"
        if 'live_data' in st.session_state and not st.session_state.get('live_data_is_stale', True):
            # Use the successfully fetched and non-stale live_data for ohlc_data processing
            raw_live_data = st.session_state.live_data
            ohlc_data_internal = raw_live_data.copy()
            
            # Set Date as index and convert column names to lowercase (to match data_loader format)
            if 'Date' in ohlc_data_internal.columns:
                ohlc_data_internal.set_index('Date', inplace=True)
            
            # Convert column names to lowercase
            ohlc_data_internal.columns = ohlc_data_internal.columns.str.lower()
            
            # Ensure we have the required OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in ohlc_data_internal.columns]
            if missing_cols:
                st.error(f"Missing required columns in live data: {missing_cols}")
                ohlc_data_internal = None
            else:
                # Ensure OHLC columns are numeric
                for col in required_cols:
                    ohlc_data_internal[col] = pd.to_numeric(ohlc_data_internal[col], errors='coerce')
                
                # Drop rows with NaN values
                ohlc_data_internal.dropna(subset=required_cols, inplace=True)
                
                # Sort by date index
                ohlc_data_internal.sort_index(inplace=True)
            
            # For preview, use the raw st.session_state.live_data to show original format
            show_preview_for_data_internal = st.session_state.live_data 
        # Else, ohlc_data_internal remains None if live_data is stale or not present

    # --- Preview Display ---
    if show_preview_for_data_internal is not None and not show_preview_for_data_internal.empty:
        st.subheader(f"Data Preview: {data_source_name_internal} ({len(show_preview_for_data_internal)} rows)")
        if len(show_preview_for_data_internal) > 10:
            first_5 = show_preview_for_data_internal.head(5)
            last_5 = show_preview_for_data_internal.tail(5)
            combined_preview = pd.concat([first_5, last_5])
            st.dataframe(combined_preview)
            st.caption(f"Showing first 5 and last 5 rows of {len(show_preview_for_data_internal)} total rows")
        else:
            st.dataframe(show_preview_for_data_internal)
    elif st.session_state.data_source == 'live' and st.session_state.get('live_data_is_stale', True):
        st.info(f"Preview for {data_source_name_internal} will update after fetching data. Click 'Fetch Live Data'.")
    elif st.session_state.data_source == 'live':
        st.info(f"Click 'Fetch Live Data' for {data_source_name_internal} to see preview and run analysis.")
    elif st.session_state.data_source == 'file' and uploaded_file_obj is None:
        st.info("Please upload a CSV file to begin analysis.")
    elif st.session_state.data_source == 'file':
        st.error(f"Failed to load/process data from '{uploaded_file_obj.name}'. Ensure CSV has Date,Open,High,Low,Close.")
        if 'ohlc_data_for_length_check_processed' in st.session_state:
            del st.session_state.ohlc_data_for_length_check_processed

    # Show latest price for live data (only if data is not stale)
    if st.session_state.data_source == 'live' and ohlc_data_internal is not None:
        try:
            # Determine the effective symbol for latest price
            effective_symbol = st.session_state.selected_symbol 
            custom_input_processed = st.session_state.get("custom_symbol_input", "").strip().upper()
            if custom_input_processed:
                effective_symbol = custom_input_processed
            fetcher = CryptoDataFetcher(exchange_name=st.session_state.selected_exchange)
            latest_price = fetcher.fetch_latest_price(effective_symbol)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${latest_price['last_price']:,.2f}")
            with col2:
                st.metric("24h Change", f"{latest_price['change_24h']:.2f}%")
            with col3:
                st.metric("Volume", f"{latest_price['volume']:,.0f}")
        except Exception as e:
            st.warning(f"Could not fetch latest price: {e}")
    
    return ohlc_data_internal, data_source_name_internal

# --- Analysis Results Display Function ---
def display_analysis_results(current_ohlc_data, current_data_source_name):
    """Display analysis results with downloads, database management, and scheduling"""
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
                return calculate_core_cycle_components(current_ohlc_data.copy(), **kwargs)

            core_calc_args_tuple = tuple(sorted(core_calc_args.items()))
            ohlc_data_hash_for_cache_key = None
            if current_ohlc_data is not None:
                 ohlc_data_hash_for_cache_key = pd.util.hash_pandas_object(current_ohlc_data).sum()
            
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
                    calc_bar_idx = full_results.get('calculation_bar_idx', len(current_ohlc_data) - 1)
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
                        title=f"{current_data_source_name} - Cycle Overlay (Analysis Window: {analysis_sample_size} bars)"
                    )
                    if fig: st.pyplot(fig)
                    else: st.warning("Plot generation failed or no figure returned.")

                    # --- Enhanced Download & Database Features (Milestone 3) ---
                    st.subheader("Download Results & Database Management")
                    
                    # Store analysis results in database
                    run_id = f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Prepare cycle results for database storage
                    cycle_results = []
                    if not cycle_table_df.empty:
                        for _, row in cycle_table_df.iterrows():
                            cycle_results.append({
                                'rank': int(row.get('Rank', 0)),
                                'period': float(row.get('Period', 0)),
                                'amplitude': float(row.get('Amplitude', 0)),
                                'phase': float(row.get('Phase', 0)),
                                'bartels_score': float(row.get('Bartels', 0)) if 'Bartels' in row else 0.0,
                                'strength': float(row.get('Strength', 0)) if 'Strength' in row else 0.0
                            })
                    
                    # Prepare wave data
                    past_wave_data_list = []
                    if full_results.get('goertzel_past_wave') is not None and hasattr(full_results['goertzel_past_wave'], 'tolist'):
                        past_wave_data_list = np.round(np.array(full_results['goertzel_past_wave']), 4).tolist()
                    
                    future_wave_data_list = []
                    if full_results.get('goertzel_future_wave') is not None and hasattr(full_results['goertzel_future_wave'], 'tolist'):
                        future_wave_data_list = np.round(np.array(full_results['goertzel_future_wave']), 4).tolist()
                    
                    wave_data = {
                        'past': past_wave_data_list,
                        'future': future_wave_data_list
                    }
                    
                    # Store in database
                    download_settings = {**core_calc_args, **wave_sum_args}
                    download_settings["SampleSize_used_for_analysis"] = analysis_sample_size
                    download_settings["WindowSizePast_for_wave_matrix_calc"] = plot_ws_past 
                    download_settings["WindowSizeFuture_for_wave_matrix_calc"] = plot_ws_future
                    
                    # Determine the effective symbol for database storage
                    if st.session_state.data_source == 'live':
                        symbol_for_db = st.session_state.selected_symbol
                        custom_input_processed = st.session_state.get("custom_symbol_input", "").strip().upper()
                        if custom_input_processed:
                            symbol_for_db = custom_input_processed
                    else:
                        symbol_for_db = current_data_source_name
                    timeframe_for_db = st.session_state.selected_timeframe if st.session_state.data_source == 'live' else '1d'
                    
                    try:
                        db_success = st.session_state.db_manager.store_analysis_results(
                            run_id=run_id,
                            symbol=symbol_for_db,
                            timeframe=timeframe_for_db,
                            parameters=download_settings,
                            cycle_results=cycle_results,
                            wave_data=wave_data
                        )
                        if db_success:
                            st.success(f"✅ Analysis results saved to database (Run ID: {run_id})")
                        else:
                            st.warning("⚠️ Failed to save to database, but downloads are still available")
                    except Exception as e:
                        st.warning(f"⚠️ Database storage failed: {e}")
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON download (original functionality)
                        json_dl_data = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'run_id': run_id,
                            'data_source': st.session_state.data_source,
                            'symbol': symbol_for_db,
                            'timeframe': timeframe_for_db,
                            'settings_used': {},
                            'cycle_table_data': cycle_table_df.reset_index().to_dict(orient='records') if not cycle_table_df.empty else [],
                            'num_cycles': full_results.get('number_of_cycles', 0),
                            'past_wave_data': past_wave_data_list,
                            'future_wave_data': future_wave_data_list
                        }
                        
                        try:
                            serializable_settings = {}
                            for k, v in download_settings.items():
                                if isinstance(v, np.integer):
                                    serializable_settings[k] = int(v)
                                elif isinstance(v, np.floating):
                                    serializable_settings[k] = float(v)
                                elif isinstance(v, np.bool_):
                                    serializable_settings[k] = bool(v)
                                elif isinstance(v, np.ndarray):
                                    serializable_settings[k] = v.tolist()
                                else:
                                    serializable_settings[k] = v
                            json_dl_data['settings_used'] = serializable_settings
                            
                            json_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                                json_dl_data, f"{symbol_for_db}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'json'
                            )
                            st.download_button(
                                "📄 Download JSON",
                                json_bytes,
                                f"{symbol_for_db}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                "application/json",
                                key="dl_json_btn"
                            )
                        except Exception as e:
                            st.error(f"JSON preparation error: {e}")
                    
                    with col2:
                        # CSV download for cycle results
                        if not cycle_table_df.empty:
                            try:
                                csv_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                                    cycle_table_df, f"{symbol_for_db}_cycles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 'csv'
                                )
                                st.download_button(
                                    "📊 Download Cycles CSV",
                                    csv_bytes,
                                    f"{symbol_for_db}_cycles_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    "text/csv",
                                    key="dl_csv_btn"
                                )
                            except Exception as e:
                                st.error(f"CSV preparation error: {e}")
                    
                    # Database management section
                    with st.expander("🗄️ Database Management", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Database Statistics**")
                            try:
                                stats = st.session_state.db_manager.get_database_stats()
                                for key, value in stats.items():
                                    st.write(f"• {key}: {value}")
                            except Exception as e:
                                st.error(f"Error getting database stats: {e}")
                        
                        with col2:
                            st.write("**Recent Analysis Runs**")
                            try:
                                # Get recent runs from database
                                recent_runs = st.session_state.db_manager.get_recent_runs(symbol=symbol_for_db, limit=5)
                                if recent_runs:
                                    for run in recent_runs:
                                        run_time = run.get('timestamp', 'Unknown')
                                        if isinstance(run_time, str):
                                            try:
                                                # Parse timestamp and format nicely
                                                dt = datetime.datetime.fromisoformat(run_time.replace('Z', '+00:00'))
                                                formatted_time = dt.strftime('%m/%d %H:%M')
                                            except:
                                                formatted_time = run_time[:16]  # Fallback
                                        else:
                                            formatted_time = str(run_time)[:16]
                                        
                                        st.write(f"• {formatted_time} - {run.get('symbol', 'Unknown')} ({run.get('timeframe', 'Unknown')})")
                                else:
                                    st.write("• No recent runs found")
                                st.write(f"• Current run: {run_id}")
                                st.write("• Database: financial_cycle_analyzer.db")
                            except Exception as e:
                                st.error(f"Error getting recent runs: {e}")
                                st.write(f"• Current run: {run_id}")
                                st.write("• Database: financial_cycle_analyzer.db")
                    
                    # Scheduling section
                    with st.expander("⏰ Analysis Scheduling", expanded=False):
                        st.write("**Automatic Recalculation**")
                        st.info("💡 Recalculation works by checking for new bars every N timeframe intervals. For example, if timeframe is 5m and N=1, it checks every 5 minutes for new data.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            schedule_enabled = st.checkbox("Enable Auto-Recalculation", key="schedule_enabled")
                            if schedule_enabled:
                                recalc_bars = st.selectbox(
                                    "Recalculate Every N Bars",
                                    options=[1, 2, 3, 5, 10],
                                    format_func=lambda x: f"{x} new bar{'s' if x > 1 else ''}",
                                    key="recalc_bars"
                                )
                                
                                # Use the centralized get_next_candle_time function
                                
                                next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                                time_until_next = next_candle_time - datetime.datetime.now(pytz.utc)
                                
                                if time_until_next.total_seconds() < 60:
                                    interval_text = f"{int(time_until_next.total_seconds())} seconds"
                                elif time_until_next.total_seconds() < 3600:
                                    minutes = int(time_until_next.total_seconds() // 60)
                                    interval_text = f"{minutes} minute{'s' if minutes > 1 else ''}"
                                elif time_until_next.total_seconds() < 86400:
                                    hours = int(time_until_next.total_seconds() // 3600)
                                    interval_text = f"{hours} hour{'s' if hours > 1 else ''}"
                                else:
                                    days = int(time_until_next.total_seconds() // 86400)
                                    interval_text = f"{days} day{'s' if days > 1 else ''}"
                                
                                st.write(f"⏱️ **Next Update:** When {recalc_bars} new {timeframe_for_db} candle{'s' if recalc_bars > 1 else ''} open{'s' if recalc_bars == 1 else ''}")
                                st.write(f"🕐 **Next Candle Opens:** {next_candle_time.strftime('%H:%M:%S')} (in {interval_text})")
                        
                        with col2:
                            # Initialize schedule tracking in session state
                            if 'schedule_info' not in st.session_state:
                                st.session_state.schedule_info = {}
                            
                            schedule_id = f"auto_{symbol_for_db}_{timeframe_for_db}"
                            is_scheduled = schedule_id in st.session_state.schedule_info
                            
                            # Auto-start/stop scheduling based on checkbox
                            if schedule_enabled and not is_scheduled:
                                # Calculate next candle open time using centralized function
                                next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                                
                                # Store schedule info in session state for countdown
                                st.session_state.schedule_info[schedule_id] = {
                                    'symbol': symbol_for_db,
                                    'timeframe': timeframe_for_db,
                                    'last_run': datetime.datetime.now(pytz.utc),
                                    'next_run': next_candle_time,
                                    'settings': download_settings.copy(),
                                    'recalc_bars': recalc_bars
                                }
                                st.success(f"✅ Auto-scheduling enabled: next update when {recalc_bars} new {timeframe_for_db} candle{'s' if recalc_bars > 1 else ''} open{'s' if recalc_bars == 1 else ''}")
                                st.rerun()
                            
                            elif not schedule_enabled and is_scheduled:
                                # Auto-stop when checkbox is unchecked
                                del st.session_state.schedule_info[schedule_id]
                                st.success("✅ Auto-scheduling disabled")
                                st.rerun()
                            
                            elif is_scheduled:
                                schedule_info = st.session_state.schedule_info[schedule_id]
                                
                                # Create countdown fragment that auto-updates
                                @st.fragment(run_every="1s")
                                def countdown_display():
                                    next_run = schedule_info['next_run']
                                    now = datetime.datetime.now(pytz.utc)
                                    
                                    if now >= next_run:
                                        # Time to run analysis
                                        st.warning("🔄 Running scheduled analysis...")
                                        
                                        try:
                                            # Fetch only new data (fetch a small number of recent bars to be safe)
                                            fetcher = CryptoDataFetcher(exchange_name=st.session_state.selected_exchange)
                                            bars_to_fetch = max(5, recalc_bars + 2)  # Fetch at least 5 bars to be safe
                                            new_data = fetcher.fetch_ohlcv(
                                                symbol=symbol_for_db,
                                                timeframe=timeframe_for_db,
                                                limit=bars_to_fetch
                                            )
                                            
                                            if not new_data.empty:
                                                # Merge new data with existing data
                                                if 'live_data' in st.session_state and not st.session_state.live_data.empty:
                                                    existing_data = st.session_state.live_data.copy()
                                                    
                                                    # Concatenate existing and new data
                                                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                                                    
                                                    # Remove duplicates based on Date column, keeping the latest version
                                                    combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
                                                    
                                                    # Sort by Date
                                                    combined_data = combined_data.sort_values('Date').reset_index(drop=True)
                                                    
                                                    # Store merged data in database
                                                    db_success = st.session_state.db_manager.store_price_data(
                                                        combined_data, 
                                                        symbol_for_db, 
                                                        timeframe_for_db
                                                    )
                                                    
                                                    # Update session state with merged data
                                                    st.session_state.live_data = combined_data
                                                    st.session_state.ohlc_data_length_for_sliders = len(combined_data)
                                                else:
                                                    # No existing full dataset - fetch complete historical data instead of using small incremental fetch
                                                    st.info("Auto-recalc: Fetching initial historical dataset...")
                                                    try:
                                                        # Get the Historical Days value from the slider
                                                        days_back_value = st.session_state.get("days_back_slider", 90)
                                                        
                                                        # Fetch full historical data
                                                        full_historical_data = fetcher.get_historical_data(
                                                            symbol=symbol_for_db,
                                                            timeframe=timeframe_for_db,
                                                            days_back=days_back_value
                                                        )
                                                        
                                                        if not full_historical_data.empty:
                                                            # Store in database
                                                            db_success = st.session_state.db_manager.store_price_data(
                                                                full_historical_data, 
                                                                symbol_for_db, 
                                                                timeframe_for_db
                                                            )
                                                            
                                                            # Store the full historical dataset
                                                            st.session_state.live_data = full_historical_data
                                                            st.session_state.ohlc_data_length_for_sliders = len(full_historical_data)
                                                            
                                                            if db_success:
                                                                st.success(f"Auto-recalc: Fetched initial historical dataset ({len(full_historical_data)} bars)")
                                                            else:
                                                                st.warning(f"Auto-recalc: Fetched initial historical dataset ({len(full_historical_data)} bars) but failed to store in database")
                                                        else:
                                                            # Full fetch failed, fall back to small data as last resort
                                                            st.warning("Auto-recalc: Full historical fetch returned no data. Using incremental data as fallback.")
                                                            st.session_state.live_data = new_data
                                                            st.session_state.ohlc_data_length_for_sliders = len(new_data)
                                                    
                                                    except Exception as full_fetch_error:
                                                        # Full historical fetch failed, use small incremental data as last resort
                                                        st.error(f"Auto-recalc: Failed to fetch full historical data ({full_fetch_error}). Analysis may be impacted.")
                                                        st.session_state.live_data = new_data
                                                        st.session_state.ohlc_data_length_for_sliders = len(new_data)
                                                
                                                # Calculate next candle open time for next execution
                                                next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                                                
                                                # Update schedule for next run
                                                schedule_info['last_run'] = now
                                                schedule_info['next_run'] = next_candle_time
                                                
                                                st.success("✅ Scheduled analysis completed! New data fetched.")
                                                st.rerun()  # Trigger full app rerun to update analysis
                                            else:
                                                st.warning("⚠️ No new data available")
                                                # Still update next run time to next candle
                                                next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                                                schedule_info['next_run'] = next_candle_time
                                                
                                        except Exception as e:
                                            st.error(f"Scheduled analysis failed: {e}")
                                            # Update next run time to next candle even if failed
                                            next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                                            schedule_info['next_run'] = next_candle_time
                                    
                                    else:
                                        # Show live countdown
                                        time_remaining = next_run - now
                                        total_seconds = int(time_remaining.total_seconds())
                                        
                                        if total_seconds > 0:
                                            hours = total_seconds // 3600
                                            minutes = (total_seconds % 3600) // 60
                                            seconds = total_seconds % 60
                                            
                                            if hours > 0:
                                                countdown_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                                            else:
                                                countdown_text = f"{minutes:02d}:{seconds:02d}"
                                            
                                            st.write(f"⏰ **Next run in:** {countdown_text}")
                                        else:
                                            st.write("⏰ **Next run:** Any moment now...")
                                
                                # Call the auto-updating countdown fragment
                                countdown_display()
                            
                            # Show active schedules
                            if st.session_state.schedule_info:
                                st.write(f"**Active Schedules:** {len(st.session_state.schedule_info)}")
                                for sched_id, info in st.session_state.schedule_info.items():
                                    st.write(f"• {info['symbol']} ({info['timeframe']}) - Every {info['recalc_bars']} bar(s)")

# --- Main App Flow ---
# Process uploaded settings file BEFORE rendering sidebar to avoid widget conflicts
uploaded_settings_file = st.session_state.get('upload_settings_widget')
if uploaded_settings_file is not None:
    try:
        loaded_settings_from_upload = json.load(uploaded_settings_file)
        
        # Define the set of keys that are permissible to load
        # This should be consistent with get_settings_to_persist()
        valid_keys = set(_get_default_settings().keys()) | {
            'data_source', 'selected_exchange', 'selected_symbol', 'selected_timeframe',
            'custom_symbol_input', 'live_data_is_stale', 
            'user_intended_WindowSizePast_base'  # If you persist this
        }

        settings_applied_count = 0
        
        for key, value in loaded_settings_from_upload.items():
            if key == 'user_intended_WindowSizePast_base':
                st.session_state['user_intended_WindowSizePast_base'] = value
                # Also set the actual analysis value; it will be clamped by the slider logic later if needed
                st.session_state['WindowSizePast_base'] = value 
                settings_applied_count += 1
            elif key == 'custom_symbol_input':
                st.session_state['custom_symbol_input'] = str(value).strip().upper() # Ensure it's processed
                settings_applied_count += 1
            elif key in valid_keys:
                st.session_state[key] = value
                settings_applied_count += 1
            else:
                st.sidebar.warning(f"Ignoring unknown setting '{key}' from uploaded file.")
        
        if settings_applied_count > 0:
            st.sidebar.success(f"✅ Successfully loaded {settings_applied_count} settings from file!")
            # Auto-save the newly loaded settings as the new "last session"
            auto_save_settings_to_file()
            st.rerun()  # Rerun to apply settings throughout the UI
        else:
            st.sidebar.warning("No valid settings found in the uploaded file.")

    except json.JSONDecodeError:
        st.sidebar.error("❌ Error: Invalid JSON file. Could not load settings.")
    except Exception as e:
        st.sidebar.error(f"❌ Error processing settings file: {e}")

# Render sidebar and get uploaded file
uploaded_file = render_sidebar()

# Call the data loading and preview function
ohlc_data, data_source_name = load_and_preview_data(uploaded_file)

# Continue with analysis if data is available
if ohlc_data is not None and not ohlc_data.empty:
    display_analysis_results(ohlc_data, data_source_name)

# Clean up session state when no data is available
if st.session_state.data_source == 'file' and uploaded_file is None:
    if 'ohlc_data_for_length_check_processed' in st.session_state:
        del st.session_state.ohlc_data_for_length_check_processed
    if 'live_data' in st.session_state:
        del st.session_state.live_data