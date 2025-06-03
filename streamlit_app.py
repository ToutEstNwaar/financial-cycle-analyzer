# streamlit_app.py

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import datetime
import pytz
import json
import tempfile
import math

# MUST be first Streamlit command
st.set_page_config(layout="wide", page_title="Cycle Indicator Analysis")

# Suppress deprecation warning from streamlit-cookies-manager dependency
import warnings
warnings.filterwarnings("ignore", message=".*st.cache.*deprecated.*", category=DeprecationWarning)

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

# Add debug tracing AFTER all imports are complete
print(f"🔍 DEBUG: App started/rerun at {datetime.datetime.now()}")
print(f"🔍 DEBUG: Session state keys: {list(st.session_state.keys())}")
print(f"🔍 DEBUG: run_calculation = {st.session_state.get('run_calculation', 'Not set')}")
print(f"🔍 DEBUG: calculation_needed = {st.session_state.get('calculation_needed', 'Not set')}")

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
        'custom_symbol_input'
    ]
    for key in data_source_keys:
        if key in st.session_state:
            settings_to_save[key] = st.session_state[key]
    
    # Auto-recalculation settings
    schedule_keys = ['schedule_enabled', 'recalc_bars']
    for key in schedule_keys:
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
                'custom_symbol_input', 
                'user_intended_WindowSizePast_base', 'schedule_enabled', 'recalc_bars'
            }

            settings_loaded_count = 0
            significant_changes = 0  # Track only significant setting changes
            
            for key, value in loaded_settings.items():
                # Check if this is actually a new/changed value
                current_value = st.session_state.get(key)
                
                if key == 'user_intended_WindowSizePast_base':
                    if current_value != value:
                        st.session_state['user_intended_WindowSizePast_base'] = value
                        st.session_state['WindowSizePast_base'] = value 
                        settings_loaded_count += 1
                        significant_changes += 1
                elif key == 'custom_symbol_input':
                    processed_value = str(value).strip().upper()
                    if current_value != processed_value:
                        st.session_state['custom_symbol_input'] = processed_value
                        settings_loaded_count += 1
                        if processed_value:  # Only count as significant if not empty
                            significant_changes += 1
                elif key in valid_keys:
                    if current_value != value:
                        st.session_state[key] = value
                        settings_loaded_count += 1
                        # Only count certain keys as significant changes worth notifying about
                        if key not in ['schedule_enabled', 'recalc_bars']:
                            significant_changes += 1
            
            # Only show toast for significant changes (more than just schedule settings)
            if significant_changes > 0:
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
    
    # Ensure live data freshness is maintained when data exists
    if (st.session_state.data_source == 'live' and 
        'live_data' in st.session_state and 
        not st.session_state.live_data.empty and
        not st.session_state.get('live_data_is_stale', True)):
        # Data exists and is fresh - ensure it stays fresh
        st.session_state.live_data_is_stale = False
    
    # Initialize calculation control state
    if 'calculation_needed' not in st.session_state:
        st.session_state.calculation_needed = True  # Initially need calculation
    if 'run_calculation' not in st.session_state:
        st.session_state.run_calculation = False  # Controls when to actually run calculation
    
    # Initialize auto-recalculation settings
    if 'schedule_enabled' not in st.session_state:
        st.session_state.schedule_enabled = False  # Auto-recalculation disabled by default
    if 'recalc_bars' not in st.session_state:
        st.session_state.recalc_bars = 1  # Default to recalculate every 1 new bar
    
    # Initialize Group 2 wave summation control state
    if 'rerun_wave_summation' not in st.session_state:
        st.session_state.rerun_wave_summation = False  # Controls when to re-run wave summation
    if 'cached_core_components_output' not in st.session_state:
        st.session_state.cached_core_components_output = None  # Stores core components for re-use

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

# Load tradable pairs
st.session_state.tradable_pairs_list = load_tradable_pairs()



# --- Utility Functions ---
def mark_live_data_stale():
    """Mark live data as stale when selection parameters change"""
    st.session_state.live_data_is_stale = True

def on_setting_change():
    """Callback for when any setting changes - auto-save settings"""
    auto_save_settings_to_file()
    # Mark settings download data as needing refresh
    st.session_state.settings_data_needs_refresh = True

def mark_live_data_stale_and_save():
    """Mark live data as stale and auto-save settings"""
    mark_live_data_stale()
    auto_save_settings_to_file()
    # Mark settings download data as needing refresh
    st.session_state.settings_data_needs_refresh = True

# Add new function to mark calculation as needed
def mark_calculation_needed():
    """Mark that calculation needs to be run when parameters change"""
    st.session_state.calculation_needed = True

def on_setting_change_with_calc_flag():
    """Callback for when any setting changes - auto-save settings and mark calculation needed"""
    auto_save_settings_to_file()
    mark_calculation_needed()
    # Mark settings download data as needing refresh
    st.session_state.settings_data_needs_refresh = True
    # Clear all cached items related to a specific analysis result
    keys_to_clear = [
        'cached_fig_for_display', 'cached_table_for_display', 'cached_plot_timestamp',
        'cached_run_id_for_downloads', 'cached_symbol_for_db_for_downloads', 
        'cached_timeframe_for_db_for_downloads', 'cached_cycle_results_for_downloads',
        'cached_wave_data_for_downloads', 'cached_download_settings_for_downloads',
        'cached_download_timestamp', 'cached_json_filename', 'cached_json_download_bytes',
        'cached_cycles_csv_filename', 'cached_cycles_csv_download_bytes',
        'cached_combined_csv_filename', 'cached_combined_csv_download_bytes',
        'cached_db_stats_display', 'cached_recent_runs_display',
        'cached_core_components_output'  # Clear core components when Group 1 settings change
    ]
    for key_to_clear in keys_to_clear:
        if key_to_clear in st.session_state:
            del st.session_state[key_to_clear]

def on_group2_setting_change():
    """Callback for when Group 2 settings change - auto-save settings and trigger wave summation re-run"""
    auto_save_settings_to_file()
    # Set flag to trigger wave summation re-run
    st.session_state.rerun_wave_summation = True
    # Mark settings download data as needing refresh
    st.session_state.settings_data_needs_refresh = True
    # Clear cached items related to display as these will need to be regenerated
    keys_to_clear = [
        'cached_fig_for_display', 'cached_table_for_display', 'cached_plot_timestamp',
        'cached_run_id_for_downloads', 'cached_symbol_for_db_for_downloads', 
        'cached_timeframe_for_db_for_downloads', 'cached_cycle_results_for_downloads',
        'cached_wave_data_for_downloads', 'cached_download_settings_for_downloads',
        'cached_download_timestamp', 'cached_json_filename', 'cached_json_download_bytes',
        'cached_cycles_csv_filename', 'cached_cycles_csv_download_bytes',
        'cached_combined_csv_filename', 'cached_combined_csv_download_bytes',
        'cached_db_stats_display', 'cached_recent_runs_display'
    ]
    for key_to_clear in keys_to_clear:
        if key_to_clear in st.session_state:
            del st.session_state[key_to_clear]

def check_and_run_auto_recalculation():
    """
    Global function to initialize auto-recalculation schedule if needed.
    The actual triggering is now handled by the countdown fragment.
    """
    # Only run if auto-recalculation is enabled and we have live data
    if not st.session_state.get('schedule_enabled', False):
        return
    
    if st.session_state.data_source != 'live':
        return
    
    # Initialize schedule tracking
    if 'schedule_info' not in st.session_state:
        st.session_state.schedule_info = {}
    
    # Determine symbol and timeframe
    symbol_for_db = st.session_state.selected_symbol
    custom_input_processed = st.session_state.get("custom_symbol_input", "").strip().upper()
    if custom_input_processed:
        symbol_for_db = custom_input_processed
    timeframe_for_db = st.session_state.selected_timeframe
    
    schedule_id = f"auto_{symbol_for_db}_{timeframe_for_db}"
    is_scheduled = schedule_id in st.session_state.schedule_info
    recalc_bars = st.session_state.get('recalc_bars', 1)
    
    # Auto-start scheduling when enabled
    if not is_scheduled:
        # Calculate next candle open time using centralized function
        next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
        
        # Store schedule info in session state for countdown
        st.session_state.schedule_info[schedule_id] = {
            'symbol': symbol_for_db,
            'timeframe': timeframe_for_db,
            'last_run': datetime.datetime.now(pytz.utc),
            'next_run': next_candle_time,
            'recalc_bars': recalc_bars
        }
        st.success(f"✅ Auto-recalculation scheduled for {symbol_for_db} every {recalc_bars} {timeframe_for_db} bar(s)")
    
    # The actual triggering logic is now handled by the countdown fragment

def display_auto_recalc_status():
    """Display auto-recalculation status and countdown if active"""
    if not st.session_state.get('schedule_enabled', False):
        return
    
    if st.session_state.data_source != 'live':
        st.info("💡 Auto-recalculation only works with live data")
        return
    
    # Get schedule info
    symbol_for_db = st.session_state.selected_symbol
    custom_input_processed = st.session_state.get("custom_symbol_input", "").strip().upper()
    if custom_input_processed:
        symbol_for_db = custom_input_processed
    timeframe_for_db = st.session_state.selected_timeframe
    
    schedule_id = f"auto_{symbol_for_db}_{timeframe_for_db}"
    
    if schedule_id in st.session_state.schedule_info:
        schedule_info = st.session_state.schedule_info[schedule_id]
        next_run = schedule_info['next_run']
        now = datetime.datetime.now(pytz.utc)
        recalc_bars = schedule_info['recalc_bars']
        
        # Create countdown that auto-updates
        @st.fragment(run_every="1s")
        def countdown_fragment():
            current_time = datetime.datetime.now(pytz.utc)
            time_remaining = next_run - current_time
            total_seconds = int(time_remaining.total_seconds())
            
            if total_seconds > 0:
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                
                if hours > 0:
                    countdown_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    countdown_text = f"{minutes:02d}:{seconds:02d}"
                
                st.info(f"⏰ **Next auto-analysis in:** {countdown_text}")
                st.write(f"**Target:** When {recalc_bars} new {timeframe_for_db} candle{'s' if recalc_bars > 1 else ''} open{'s' if recalc_bars == 1 else ''}")
            else:
                st.info("⏰ **Auto-analysis:** Triggering any moment now...")
                
                # Time to run analysis! Trigger it here in the fragment
                schedule_info = st.session_state.schedule_info[schedule_id]
                current_time_trigger = datetime.datetime.now(pytz.utc)
                
                # Only trigger if we haven't already triggered recently (prevent multiple triggers)
                last_trigger_time = schedule_info.get('last_trigger_time', None)
                if last_trigger_time is None or (current_time_trigger - last_trigger_time).total_seconds() > 30:
                    
                    st.warning("🔄 Auto-recalculation triggered! Fetching new data...")
                    
                    try:
                        # Fetch new data
                        fetcher = CryptoDataFetcher(exchange_name=st.session_state.selected_exchange)
                        bars_to_fetch = max(5, recalc_bars + 2)
                        new_data = fetcher.fetch_ohlcv(
                            symbol=symbol_for_db,
                            timeframe=timeframe_for_db,
                            limit=bars_to_fetch
                        )
                        
                        if not new_data.empty:
                            # Merge with existing data if available
                            if 'live_data' in st.session_state and not st.session_state.live_data.empty:
                                existing_data = st.session_state.live_data.copy()
                                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                                combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
                                combined_data = combined_data.sort_values('Date').reset_index(drop=True)
                            else:
                                # No existing data, use new data
                                combined_data = new_data
                            
                            # Store in database
                            db_success = st.session_state.db_manager.store_price_data(
                                combined_data, 
                                symbol_for_db, 
                                timeframe_for_db
                            )
                            
                            # Update session state
                            st.session_state.live_data = combined_data
                            st.session_state.ohlc_data_length_for_sliders = len(combined_data)
                            st.session_state.live_data_is_stale = False
                            
                            # Calculate next run time
                            next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                            schedule_info['last_run'] = current_time_trigger
                            schedule_info['next_run'] = next_candle_time
                            schedule_info['last_trigger_time'] = current_time_trigger  # Track when we triggered
                            
                            # Trigger cycle engine
                            st.session_state.run_calculation = True
                            st.session_state.calculation_needed = False
                            auto_save_settings_to_file()
                            
                            st.success("✅ Auto-recalculation completed! New data fetched and analysis triggered.")
                            st.rerun()
                        else:
                            st.warning("⚠️ No new data available for auto-recalculation")
                            # Still update next run time
                            next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                            schedule_info['next_run'] = next_candle_time
                            schedule_info['last_trigger_time'] = current_time_trigger
                            
                    except Exception as e:
                        st.error(f"Auto-recalculation failed: {e}")
                        # Update next run time even if failed
                        next_candle_time = get_next_candle_time(timeframe_for_db, recalc_bars)
                        schedule_info['next_run'] = next_candle_time
                        schedule_info['last_trigger_time'] = current_time_trigger

        countdown_fragment()
    else:
        st.info("✅ Auto-recalculation enabled - waiting for initialization...")

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
    
    print(f"🔍 DEBUG: Preparing settings download button")
    print(f"🔍 DEBUG: settings_data_needs_refresh = {st.session_state.get('settings_data_needs_refresh', 'Not set')}")
    
    # Prepare settings download data once and store in session state
    if 'settings_download_data' not in st.session_state or st.session_state.get('settings_data_needs_refresh', True):
        print(f"🔍 DEBUG: Generating new settings download data")
        settings_json_string = json.dumps(settings_to_download, indent=4)
        st.session_state.settings_download_data = settings_json_string
        st.session_state.settings_data_needs_refresh = False
    else:
        print(f"🔍 DEBUG: Using cached settings download data")

    def settings_download_fragment():
        print(f"🔍 DEBUG: Creating settings download button widget")
        
        # Store analysis state before download button (in case it causes rerun)
        if st.session_state.get('run_calculation', False):
            st.session_state['analysis_was_running'] = True
        
        downloaded = st.sidebar.download_button(
            label="📥 Download Current Settings",
            data=st.session_state.settings_download_data,
            file_name=SETTINGS_FILENAME,
            mime="application/json",
            key="download_settings_btn",
            help="Download your current settings as a JSON file",
            use_container_width=True
        )
        
        # No need to set preserve_analysis_results flag anymore
        if downloaded:
            print(f"🔍 DEBUG: Settings download triggered")
        
        print(f"🔍 DEBUG: Settings download button widget created")

    settings_download_fragment()

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
        on_change=on_setting_change_with_calc_flag
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
    st.sidebar.selectbox("Price Source", options=source_options, key="source_price_type", on_change=on_setting_change_with_calc_flag)

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
        # Mark settings download data as needing refresh
        st.session_state.settings_data_needs_refresh = True

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
        on_change=on_setting_change_with_calc_flag,
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
        on_change=on_group2_setting_change,
        help="Defines the base number of bars for the future wave projection. Actual projection length will be at least 2 * MaxPer."
    )

    # --- Bar to Calculate (Moved after Future Window Size) ---
    st.sidebar.number_input(
        "Bar to Calculate (Offset)",
        min_value=1,
        max_value=100, # Fixed practical UI limit
        step=1,
        key="BarToCalculate",
        on_change=on_setting_change_with_calc_flag,
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
        on_change=on_group2_setting_change
    )
    st.sidebar.number_input(
        "Use Top Cycles (Count)",
        min_value=1,
        max_value=max(1, current_max_per_val_for_cycle_selection),
        step=1,
        key="UseTopCycles",
        on_change=on_group2_setting_change
    )

    st.sidebar.subheader("Source Price Processing")
    detrend_mode_options = [
        ind_settings.NONE_SMTH_DT, ind_settings.ZLAGSMTH, ind_settings.HPSMTH,
        ind_settings.ZLAGSMTHDT, ind_settings.HPSMTHDT, ind_settings.LOG_ZLAG_REGRESSION_DT
    ]
    st.sidebar.selectbox("Detrending/Smoothing Mode",options=detrend_mode_options,key="detrendornot",on_change=on_setting_change_with_calc_flag)

    if st.session_state.detrendornot == ind_settings.ZLAGSMTH:
        st.sidebar.slider("ZLMA Smooth Period", 1, 100, step=1, key="ZLMAsmoothPer", on_change=on_setting_change_with_calc_flag)
    elif st.session_state.detrendornot == ind_settings.HPSMTH:
        st.sidebar.slider("HPF Smoothing Period", 1, 100, step=1, key="HPsmoothPer", on_change=on_setting_change_with_calc_flag)
    elif st.session_state.detrendornot == ind_settings.ZLAGSMTHDT:
        st.sidebar.slider("ZLMA Detrend Fast Period", 1, 100, step=1, key="DT_ZLper1", on_change=on_setting_change_with_calc_flag)
        st.sidebar.slider("ZLMA Detrend Slow Period", 1, 200, step=1, key="DT_ZLper2", on_change=on_setting_change_with_calc_flag)
    elif st.session_state.detrendornot == ind_settings.HPSMTHDT:
        st.sidebar.slider("HPF Detrend Fast Period", 1, 100, step=1, key="DT_HPper1", on_change=on_setting_change_with_calc_flag)
        st.sidebar.slider("HPF Detrend Slow Period", 1, 200, step=1, key="DT_HPper2", on_change=on_setting_change_with_calc_flag)
    elif st.session_state.detrendornot == ind_settings.LOG_ZLAG_REGRESSION_DT:
        st.sidebar.slider("Log ZLR Smooth Period", 1, 50, step=1, key="DT_RegZLsmoothPer", on_change=on_setting_change_with_calc_flag)

    st.sidebar.subheader("Bartels Cycle Significance")
    st.sidebar.checkbox("Filter with Bartels Test", key="FilterBartels", on_change=on_setting_change_with_calc_flag)
    if st.session_state.FilterBartels:
        st.sidebar.slider("Bartels: N Cycles", 1, 20, step=1, key="BartNoCycles", on_change=on_setting_change_with_calc_flag)
        st.sidebar.slider("Bartels: Smooth Per", 1, 20, step=1, key="BartSmoothPer", on_change=on_setting_change_with_calc_flag)
        st.sidebar.slider("Bartels: Sig Limit (%)", 0.0, 100.0, step=0.1, key="BartSigLimit", on_change=on_setting_change_with_calc_flag)
        st.sidebar.checkbox("Sort by Bartels Sig", key="SortBartels", on_change=on_setting_change_with_calc_flag)

    st.sidebar.subheader("Miscellaneous Goertzel Settings")
    st.sidebar.checkbox("Squared Amplitude", key="squaredAmp", on_change=on_setting_change_with_calc_flag)
    st.sidebar.checkbox("Use Addition for Phase", key="useAddition", on_change=on_setting_change_with_calc_flag)
    st.sidebar.checkbox("Use Cosine for Waves", key="useCosine", on_change=on_setting_change_with_calc_flag)
    st.sidebar.checkbox("Use Cycle Strength", key="UseCycleStrength", on_change=on_setting_change_with_calc_flag)
    st.sidebar.checkbox("Subtract Noise Cycles", key="SubtractNoise", on_change=on_group2_setting_change)
    st.sidebar.checkbox("Use Specific Cycle List", key="UseCycleList", on_change=on_group2_setting_change)
    if st.session_state.UseCycleList:
        st.sidebar.number_input("C1 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle1", on_change=on_group2_setting_change)
        st.sidebar.number_input("C2 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle2", on_change=on_group2_setting_change)
        st.sidebar.number_input("C3 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle3", on_change=on_group2_setting_change)
        st.sidebar.number_input("C4 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle4", on_change=on_group2_setting_change)
        st.sidebar.number_input("C5 Rank", 0, max(0, current_max_per_val_for_cycle_selection), step=1, key="Cycle5", on_change=on_group2_setting_change)

    # --- Auto-Recalculation Settings ---
    st.sidebar.subheader("⏰ Auto-Recalculation")
    schedule_enabled = st.sidebar.checkbox("Enable Auto-Recalculation", key="schedule_enabled", on_change=on_setting_change, help="Automatically run analysis when new candles are available")
    if schedule_enabled:
        recalc_bars = st.sidebar.selectbox(
            "Recalculate Every N Bars",
            options=[1, 2, 3, 5, 10],
            format_func=lambda x: f"{x} new bar{'s' if x > 1 else ''}",
            key="recalc_bars",
            on_change=on_setting_change,
            help="How many new bars to wait before triggering recalculation"
        )
        st.sidebar.info("💡 Only works with live data. Engine runs when new candles open.")

    # --- Run Cycle Engine Button ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Run Analysis")
    
    # Show calculation status
    has_data = (uploaded_file is not None) or (st.session_state.data_source == 'live' and 'live_data' in st.session_state and not st.session_state.live_data.empty)
    
    if st.session_state.get('calculation_needed', True):
        status_message = "⚠️ Parameters changed - calculation needed"
        status_color = "orange"
    elif st.session_state.get('rerun_wave_summation', False):
        status_message = "🔄 Wave parameters changed - auto-updating"
        status_color = "blue"
    else:
        status_message = "✅ Results up to date"
        status_color = "green"
    
    st.sidebar.markdown(f"**Status:** :{status_color}[{status_message}]")
    
    # Run button
    run_button_disabled = not has_data
    run_button_help = "Load data first to enable analysis" if run_button_disabled else "Run cycle analysis with current parameters"
    
    if st.sidebar.button(
        "🔄 Run Cycle Engine", 
        disabled=run_button_disabled,
        help=run_button_help,
        key="run_cycle_engine_btn",
        type="primary"
    ):
        st.session_state.run_calculation = True
        st.session_state.calculation_needed = False

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

# --- New Helper Function for Downloads and DB Management ---
def _render_downloads_and_db_management(current_ohlc_data, current_data_source_name, is_full_computation_run):
    st.subheader("Download Results & Database Management")

    run_id_for_display = st.session_state.get('cached_run_id_for_downloads', f"run_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    symbol_for_display = st.session_state.get('cached_symbol_for_db_for_downloads', current_data_source_name)
    timeframe_for_display = st.session_state.get('cached_timeframe_for_db_for_downloads', '1d')
    # For DB write: settings, cycle_results (list of dicts), wave_data (dict of lists)
    settings_for_db = st.session_state.get('cached_download_settings_for_downloads', {})
    cycle_results_for_db = st.session_state.get('cached_cycle_results_for_downloads', [])
    wave_data_for_db = st.session_state.get('cached_wave_data_for_downloads', {})
    
    if not is_full_computation_run:
        st.info(f"💾 Database write skipped for cached display. Last saved Run ID for these results: {run_id_for_display}")
    else: # Full computation run, perform DB write using the cached ID and data from this run
        try:
            db_success = st.session_state.db_manager.store_analysis_results(
                run_id=run_id_for_display, # This ID was generated and cached during this full run
                symbol=symbol_for_display,
                timeframe=timeframe_for_display,
                parameters=settings_for_db,
                cycle_results=cycle_results_for_db,
                wave_data=wave_data_for_db
            )
            if db_success: st.success(f"✅ Analysis results saved to database (Run ID: {run_id_for_display})")
            else: st.warning("⚠️ Failed to save to database (full run), but downloads are available")
        except Exception as e: st.warning(f"⚠️ Database storage failed (full run): {e}")

    with st.expander("📥 Download Analysis Results", expanded=False):
        col1, col2, col3 = st.columns(3)
        download_key_suffix_render = run_id_for_display # Use the consistent run_id

        with col1: # JSON
            if st.session_state.get('cached_json_download_bytes'):
                if st.download_button("📄 Download JSON", 
                                   data=st.session_state.cached_json_download_bytes,
                                   file_name=st.session_state.cached_json_filename,
                                   mime="application/json", 
                                   key=f"dl_json_btn_{download_key_suffix_render}", use_container_width=True):
                    pass  # Download completed, no additional flags needed
            else:
                st.button("📄 Download JSON", disabled=True, use_container_width=True, help="JSON data not available for download.")
        
        with col2: # Cycles CSV
            if st.session_state.get('cached_cycles_csv_download_bytes'):
                if st.download_button("📊 Download Cycles CSV", 
                                   data=st.session_state.cached_cycles_csv_download_bytes,
                                   file_name=st.session_state.cached_cycles_csv_filename,
                                   mime="text/csv", 
                                   key=f"dl_csv_btn_{download_key_suffix_render}", use_container_width=True):
                    pass  # Download completed, no additional flags needed
            else:
                st.button("📊 Download Cycles CSV", disabled=True, use_container_width=True, help="Cycles CSV data not available.")

        with col3: # Combined CSV
            if st.session_state.get('cached_combined_csv_download_bytes'):
                if st.download_button("📈 Download Combined CSV", 
                                   data=st.session_state.cached_combined_csv_download_bytes,
                                   file_name=st.session_state.cached_combined_csv_filename,
                                   mime="text/csv", 
                                   key=f"dl_combined_btn_{download_key_suffix_render}", use_container_width=True,
                                   help="Download price data with cycle wave values"):
                    pass  # Download completed, no additional flags needed
            else:
                st.button("📈 Download Combined CSV", disabled=True, use_container_width=True, help="Combined CSV data not available.")

    with st.expander("🗄️ Database Management", expanded=False):
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            st.write("**Database Statistics**")
            if 'cached_db_stats_display' in st.session_state:
                for stat_line in st.session_state.cached_db_stats_display:
                    st.write(stat_line)
            else:
                st.write("Database stats will appear after first full analysis.")
        with col_db2:
            st.write("**Recent Analysis Runs**")
            if 'cached_recent_runs_display' in st.session_state:
                for run_line in st.session_state.cached_recent_runs_display:
                    st.write(run_line)
            else:
                st.write("Recent runs will appear after first full analysis.")
            st.write(f"• Displayed/Saved Run ID: {run_id_for_display}")

# --- Analysis Results Display Function ---
def display_analysis_results(current_ohlc_data, current_data_source_name, perform_full_computation=True):
    """Display analysis results with downloads, database management, and scheduling"""
    
    if not perform_full_computation and 'cached_fig_for_display' in st.session_state and 'cached_table_for_display' in st.session_state:
        print(f"🔍 DEBUG: Displaying results from cache for {current_data_source_name}")
        st.subheader("Cycle Analysis Results (Cached)")
        st.dataframe(st.session_state.cached_table_for_display)
        st.subheader(f"Indicator Plot (Cached - Plot Time: {st.session_state.get('cached_plot_timestamp', 'N/A')})")
        st.pyplot(st.session_state.cached_fig_for_display)
        
        # Call the helper for downloads and DB status (passing False for full computation)
        _render_downloads_and_db_management(current_ohlc_data, current_data_source_name, is_full_computation_run=False)
        return # Exit early
    
    print(f"🔍 DEBUG: Performing full computation and display for {current_data_source_name}")
    # ... (the rest of the function, i.e., the full computation path) ...
    
    print(f"🔍 DEBUG: display_analysis_results() called for {current_data_source_name}")
    print(f"🔍 DEBUG: Data length: {len(current_ohlc_data) if current_ohlc_data is not None else 'None'}")
    
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

            # Store core components in session state for Group 2 automatic updates
            st.session_state.cached_core_components_output = core_components

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

                    # DEBUG: Trace what's being passed to plotting
                    print(f"DEBUG APP: About to call plot_indicator_lines with:")
                    print(f"  full_results keys: {list(full_results.keys())}")
                    print(f"  calc_bar_idx: {calc_bar_idx}")
                    print(f"  plot_ws_past: {plot_ws_past}, plot_ws_future: {plot_ws_future}")
                    print(f"  analysis_sample_size: {analysis_sample_size}")
                    
                    past_wave_from_results = full_results.get('goertzel_past_wave')
                    future_wave_from_results = full_results.get('goertzel_future_wave')
                    print(f"  past_wave_from_results: {type(past_wave_from_results)}, length: {len(past_wave_from_results) if past_wave_from_results is not None else 'None'}")
                    print(f"  future_wave_from_results: {type(future_wave_from_results)}, length: {len(future_wave_from_results) if future_wave_from_results is not None else 'None'}")
                    
                    if past_wave_from_results is not None and len(past_wave_from_results) > 0:
                        print(f"  past_wave stats: min={np.min(past_wave_from_results):.6f}, max={np.max(past_wave_from_results):.6f}")
                        # Check if all values are the same (would cause single line)
                        unique_values = np.unique(past_wave_from_results)
                        print(f"  past_wave unique values count: {len(unique_values)}")
                        if len(unique_values) <= 3:
                            print(f"  past_wave unique values: {unique_values}")
                    
                    if future_wave_from_results is not None and len(future_wave_from_results) > 0:
                        print(f"  future_wave stats: min={np.min(future_wave_from_results):.6f}, max={np.max(future_wave_from_results):.6f}")
                        unique_values = np.unique(future_wave_from_results)
                        print(f"  future_wave unique values count: {len(unique_values)}")
                        if len(unique_values) <= 3:
                            print(f"  future_wave unique values: {unique_values}")

                    fig = plot_indicator_lines(
                        current_ohlc_data, 
                        full_results.get('goertzel_past_wave'), 
                        full_results.get('goertzel_future_wave'),
                        calc_bar_idx, 
                        plot_ws_past, 
                        plot_ws_future, 
                        title=f"{current_data_source_name} - Cycle Overlay (Analysis Window: {analysis_sample_size} bars)"
                    )
                    if fig: st.pyplot(fig)
                    else: st.warning("Plot generation failed or no figure returned.")

                    # Cache the figure and table for display
                    st.session_state.cached_fig_for_display = fig
                    st.session_state.cached_table_for_display = cycle_table_df
                    st.session_state.cached_plot_timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')

                    # --- Enhanced Download & Database Features (Milestone 3) ---
                    
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
                    
                    # Cache all necessary data for downloads and DB management section
                    st.session_state.cached_run_id_for_downloads = run_id 
                    st.session_state.cached_symbol_for_db_for_downloads = symbol_for_db
                    st.session_state.cached_timeframe_for_db_for_downloads = timeframe_for_db
                    st.session_state.cached_cycle_results_for_downloads = cycle_results # This is the list of dicts for JSON/DB
                    st.session_state.cached_wave_data_for_downloads = wave_data
                    st.session_state.cached_download_settings_for_downloads = download_settings.copy()
                    current_download_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.session_state.cached_download_timestamp = current_download_timestamp

                    # --- Pre-calculate and cache download bytes ---
                    # 1. JSON Download Bytes
                    json_dl_data_for_bytes = {
                                "analysis_metadata": {
                            "symbol": symbol_for_db, "timeframe": timeframe_for_db,
                            "timestamp": datetime.datetime.now().isoformat(), "run_id": run_id,
                            "data_length": len(current_ohlc_data) if current_ohlc_data is not None else 0
                        },
                        "cycle_results": cycle_results, # list of dicts
                        "past_wave": wave_data.get('past', []),
                        "future_wave": wave_data.get('future', []),
                        "price_data": current_ohlc_data.to_dict('records') if current_ohlc_data is not None else [],
                        "settings_used": {} # Will be populated next
                    }
                    serializable_settings_for_bytes = {}
                    for k, v_bytes in download_settings.items(): # Use a different loop variable here
                        if isinstance(v_bytes, np.integer): serializable_settings_for_bytes[k] = int(v_bytes)
                        elif isinstance(v_bytes, np.floating): serializable_settings_for_bytes[k] = float(v_bytes)
                        elif isinstance(v_bytes, np.bool_): serializable_settings_for_bytes[k] = bool(v_bytes)
                        elif isinstance(v_bytes, np.ndarray): serializable_settings_for_bytes[k] = v_bytes.tolist()
                        else: serializable_settings_for_bytes[k] = v_bytes
                    json_dl_data_for_bytes['settings_used'] = serializable_settings_for_bytes
                    
                    st.session_state.cached_json_filename = f"{symbol_for_db}_analysis_{current_download_timestamp}.json"
                    st.session_state.cached_json_download_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                        json_dl_data_for_bytes, st.session_state.cached_json_filename, 'json'
                    )

                    # 2. Cycles CSV Download Bytes (uses the DataFrame `cycle_table_df`)
                    if not cycle_table_df.empty:
                        st.session_state.cached_cycles_csv_filename = f"{symbol_for_db}_cycles_{current_download_timestamp}.csv"
                        st.session_state.cached_cycles_csv_download_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                            cycle_table_df, st.session_state.cached_cycles_csv_filename, 'csv'
                        )
                    else:
                        st.session_state.cached_cycles_csv_filename = None
                        st.session_state.cached_cycles_csv_download_bytes = None

                    # 3. Combined CSV Download Bytes
                    combined_df_for_bytes = pd.DataFrame()
                    if current_ohlc_data is not None:
                        combined_df_for_bytes = current_ohlc_data.reset_index().copy()
                        combined_df_for_bytes['Cycle'] = np.nan
                        # Simplified logic for marking cycles in combined CSV (can be expanded if needed)
                        cycle_periods_for_combined = [c.get('period') for c in cycle_results if c.get('period')] # from list of dicts
                        # Ensure `past_wave_data_list` and `future_wave_data_list` are correctly referenced or derived from `wave_data`
                        _past_wave_list = wave_data.get('past', [])
                        _future_wave_list = wave_data.get('future', [])

                        if _past_wave_list and len(_past_wave_list) > 1:
                            _past_wave_length = len(_past_wave_list)
                            if len(combined_df_for_bytes) >= _past_wave_length:
                                _data_start_idx = len(combined_df_for_bytes) - _past_wave_length
                                for i_wave in range(_past_wave_length - 1):
                                    _direction_change = (_past_wave_list[i_wave] > _past_wave_list[i_wave+1]) != \
                                                       (_past_wave_list[i_wave-1] > _past_wave_list[i_wave] if i_wave > 0 else False)
                                    if _direction_change and cycle_periods_for_combined:
                                        combined_df_for_bytes.loc[_data_start_idx + i_wave, 'Cycle'] = cycle_periods_for_combined[0]

                        if _future_wave_list:
                            _last_date = combined_df_for_bytes['Date'].iloc[-1] if not combined_df_for_bytes.empty else pd.Timestamp.now(tz='UTC')
                            _time_delta = pd.Timedelta(days=1) # Default
                            if timeframe_for_db == '1h': _time_delta = pd.Timedelta(hours=1)
                            elif timeframe_for_db == '5m': _time_delta = pd.Timedelta(minutes=5)
                            # Add more timeframes as needed
                            
                            _future_rows = []
                            for i_wave, val_wave in enumerate(_future_wave_list):
                                _future_date = _last_date + (_time_delta * (i_wave + 1))
                                _cycle_period_mark = np.nan
                                if i_wave > 0 and cycle_periods_for_combined and \
                                   (_future_wave_list[i_wave] > _future_wave_list[i_wave-1]) != \
                                   (_future_wave_list[i_wave-1] > _future_wave_list[i_wave-2] if i_wave > 1 else False) :
                                     _cycle_period_mark = cycle_periods_for_combined[0]
                                _future_rows.append({'Date': _future_date, 'Cycle': _cycle_period_mark})
                            if _future_rows:
                                _future_df_for_bytes = pd.DataFrame(_future_rows)
                                combined_df_for_bytes = pd.concat([combined_df_for_bytes, _future_df_for_bytes], ignore_index=True)

                    if not combined_df_for_bytes.empty:
                        st.session_state.cached_combined_csv_filename = f"{symbol_for_db}_combined_{current_download_timestamp}.csv"
                        st.session_state.cached_combined_csv_download_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                            combined_df_for_bytes, st.session_state.cached_combined_csv_filename, 'csv'
                        )
                    else:
                        st.session_state.cached_combined_csv_filename = None
                        st.session_state.cached_combined_csv_download_bytes = None

                    # --- Pre-cache DB Management Info ---
                    try:
                        db_stats_for_cache = st.session_state.db_manager.get_database_stats()
                        st.session_state.cached_db_stats_display = [f"• {k}: {v}" for k, v in db_stats_for_cache.items()]
                        
                        recent_runs_for_cache = st.session_state.db_manager.get_recent_runs(symbol=symbol_for_db, limit=5)
                        cached_recent_runs_display = []
                        if recent_runs_for_cache:
                            for run_item in recent_runs_for_cache:
                                _run_time = run_item.get('timestamp', 'Unknown') # Assuming 'timestamp' key from db_manager.get_recent_runs
                                if isinstance(_run_time, str):
                                    try: _dt = datetime.datetime.fromisoformat(_run_time.replace('Z', '+00:00')); _formatted_time = _dt.strftime('%m/%d %H:%M')
                                    except: _formatted_time = str(_run_time)[:16]
                                else: _formatted_time = str(_run_time)[:16]
                                cached_recent_runs_display.append(f"• {_formatted_time} - {run_item.get('symbol', 'Unknown')} ({run_item.get('timeframe', 'Unknown')})")
                        else:
                            cached_recent_runs_display.append("• No recent runs found")
                        st.session_state.cached_recent_runs_display = cached_recent_runs_display
                    except Exception as e_db_cache:
                        st.session_state.cached_db_stats_display = [f"Error caching DB stats: {e_db_cache}"]
                        st.session_state.cached_recent_runs_display = [f"Error caching recent runs: {e_db_cache}"]

                    # Call the new helper function for downloads and DB management
                    _render_downloads_and_db_management(current_ohlc_data, current_data_source_name, is_full_computation_run=True)

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
            'custom_symbol_input', 
            'user_intended_WindowSizePast_base', 'schedule_enabled', 'recalc_bars'
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

# Handle Group 2 changes automatically - re-run wave summation without full core calculation
if (st.session_state.get('rerun_wave_summation', False) and 
    st.session_state.cached_core_components_output is not None and 
    ohlc_data is not None and not ohlc_data.empty):
    
    if st.session_state.cached_core_components_output["status"] == "success":
        print("🔍 DEBUG: Rerunning wave summation due to Group 2 change.")
        
        # Get wave summation arguments from current session state
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
                    wave_sum_args[key] = default_val
                    st.session_state[key] = default_val
                else:
                    wave_settings_ok = False
                    break
        
        if wave_settings_ok:
            # Update core components with current WindowSizeFuture_base if it changed
            updated_core_components = st.session_state.cached_core_components_output.copy()
            
            # Calculate the new WindowSizeFuture_output
            max_per_from_core = len(updated_core_components.get('cyclebuffer', [])) - 1
            if max_per_from_core <= 0:
                max_per_from_core = st.session_state.get('MaxPer', 20)  # Fallback
            new_WindowSizeFuture_output = max(st.session_state.WindowSizeFuture_base, 2 * max_per_from_core)
            
            # Check if we need to regenerate the goeWorkFuture matrix
            current_future_size = updated_core_components.get('current_WindowSizeFuture', 0)
            if new_WindowSizeFuture_output != current_future_size:
                print(f"🔍 DEBUG: Regenerating goeWorkFuture matrix from {current_future_size} to {new_WindowSizeFuture_output}")
                
                # Regenerate goeWorkFuture matrix with new dimensions
                number_of_cycles = updated_core_components['number_of_cycles']
                amplitudebuffer = updated_core_components['amplitudebuffer']
                phasebuffer = updated_core_components['phasebuffer']
                cyclebuffer = updated_core_components['cyclebuffer']
                
                # Create new goeWorkFuture matrix
                new_goeWorkFuture = np.zeros((new_WindowSizeFuture_output, max_per_from_core + 1))
                
                # Get settings for wave generation
                useCosine = st.session_state.get('useCosine', False)
                useAddition = st.session_state.get('useAddition', False)
                
                # Regenerate future wave components using the same logic as in goertzel_core
                for i_cycle_num in range(1, number_of_cycles + 1):
                    if (i_cycle_num < len(amplitudebuffer) and 
                        i_cycle_num < len(phasebuffer) and 
                        i_cycle_num < len(cyclebuffer)):
                        
                        amplitude = amplitudebuffer[i_cycle_num - 1]  # Convert from 1-indexed
                        phase = phasebuffer[i_cycle_num - 1]
                        cycle_period = cyclebuffer[i_cycle_num - 1]
                        
                        if cycle_period == 0:
                            continue
                        
                        # Use inverted sign for future wave (as per original algorithm)
                        sign_val = 1.0 if not useAddition else -1.0
                        
                        # Generate future wave components
                        for k_time_future in range(new_WindowSizeFuture_output):
                            # Store future wave time-reversed (as per original algorithm)
                            array_idx_future = new_WindowSizeFuture_output - k_time_future - 1
                            if (array_idx_future >= 0 and 
                                array_idx_future < new_goeWorkFuture.shape[0] and 
                                i_cycle_num < new_goeWorkFuture.shape[1]):
                                
                                angle = phase + sign_val * k_time_future * 2.0 * np.pi / cycle_period
                                new_goeWorkFuture[array_idx_future, i_cycle_num] = amplitude * (
                                    np.cos(angle) if useCosine else np.sin(angle))
                
                # Update the core components with new matrix and size
                updated_core_components['goeWorkFuture'] = new_goeWorkFuture
                updated_core_components['current_WindowSizeFuture'] = new_WindowSizeFuture_output
            else:
                print("🔍 DEBUG: goeWorkFuture matrix size unchanged, using cached matrix")
            
            # Re-run wave summation with updated core components
            wave_results = sum_composite_waves(updated_core_components, **wave_sum_args)
            full_results = {**updated_core_components, **wave_results}
            
            # Re-generate the cycle table
            cycle_table_df = create_cycle_table(
                full_results['number_of_cycles'], full_results['cyclebuffer'], full_results['amplitudebuffer'],
                full_results['phasebuffer'], full_results['cycleBartelsBuffer'],
                st.session_state.UseCycleStrength, st.session_state.FilterBartels
            )
            
            # Re-generate the plot
            calc_bar_idx = full_results.get('calculation_bar_idx', len(ohlc_data) - 1)
            plot_ws_past = full_results.get('current_WindowSizePast', st.session_state.WindowSizePast_base)
            plot_ws_future = full_results.get('current_WindowSizeFuture', st.session_state.WindowSizeFuture_base)
            analysis_sample_size = full_results.get('calculated_SampleSize', st.session_state.WindowSizePast_base)
            
            plot_fig = plot_indicator_lines(
                ohlc_data, 
                full_results.get('goertzel_past_wave'), 
                full_results.get('goertzel_future_wave'),
                calc_bar_idx, 
                plot_ws_past, 
                plot_ws_future, 
                title=f"{data_source_name} - Cycle Overlay (Analysis Window: {analysis_sample_size} bars)"
            )
            
            # Update cached display items
            st.session_state.cached_fig_for_display = plot_fig
            st.session_state.cached_table_for_display = cycle_table_df
            st.session_state.cached_plot_timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')
            
            # Regenerate download data for Group 2 changes
            # Prepare cycle results for download
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
            
            # Prepare download settings
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
                symbol_for_db = data_source_name
            timeframe_for_db = st.session_state.selected_timeframe if st.session_state.data_source == 'live' else '1d'
            
            # Generate new run ID for Group 2 changes
            run_id = f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            current_download_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Cache all necessary data for downloads and DB management section
            st.session_state.cached_run_id_for_downloads = run_id 
            st.session_state.cached_symbol_for_db_for_downloads = symbol_for_db
            st.session_state.cached_timeframe_for_db_for_downloads = timeframe_for_db
            st.session_state.cached_cycle_results_for_downloads = cycle_results
            st.session_state.cached_wave_data_for_downloads = wave_data
            st.session_state.cached_download_settings_for_downloads = download_settings.copy()
            st.session_state.cached_download_timestamp = current_download_timestamp

            # Regenerate download bytes
            # 1. JSON Download Bytes
            json_dl_data_for_bytes = {
                "analysis_metadata": {
                    "symbol": symbol_for_db, "timeframe": timeframe_for_db,
                    "timestamp": datetime.datetime.now().isoformat(), "run_id": run_id,
                    "data_length": len(ohlc_data) if ohlc_data is not None else 0
                },
                "cycle_results": cycle_results,
                "past_wave": wave_data.get('past', []),
                "future_wave": wave_data.get('future', []),
                "price_data": ohlc_data.to_dict('records') if ohlc_data is not None else [],
                "settings_used": {}
            }
            serializable_settings_for_bytes = {}
            for k, v_bytes in download_settings.items():
                if isinstance(v_bytes, np.integer): serializable_settings_for_bytes[k] = int(v_bytes)
                elif isinstance(v_bytes, np.floating): serializable_settings_for_bytes[k] = float(v_bytes)
                elif isinstance(v_bytes, np.bool_): serializable_settings_for_bytes[k] = bool(v_bytes)
                elif isinstance(v_bytes, np.ndarray): serializable_settings_for_bytes[k] = v_bytes.tolist()
                else: serializable_settings_for_bytes[k] = v_bytes
            json_dl_data_for_bytes['settings_used'] = serializable_settings_for_bytes
            
            st.session_state.cached_json_filename = f"{symbol_for_db}_analysis_{current_download_timestamp}.json"
            st.session_state.cached_json_download_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                json_dl_data_for_bytes, st.session_state.cached_json_filename, 'json'
            )

            # 2. Cycles CSV Download Bytes
            if not cycle_table_df.empty:
                st.session_state.cached_cycles_csv_filename = f"{symbol_for_db}_cycles_{current_download_timestamp}.csv"
                st.session_state.cached_cycles_csv_download_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                    cycle_table_df, st.session_state.cached_cycles_csv_filename, 'csv'
                )
            else:
                st.session_state.cached_cycles_csv_filename = None
                st.session_state.cached_cycles_csv_download_bytes = None

            # 3. Combined CSV Download Bytes
            combined_df_for_bytes = pd.DataFrame()
            if ohlc_data is not None:
                combined_df_for_bytes = ohlc_data.reset_index().copy()
                combined_df_for_bytes['Cycle'] = np.nan
                cycle_periods_for_combined = [c.get('period') for c in cycle_results if c.get('period')]
                _past_wave_list = wave_data.get('past', [])
                _future_wave_list = wave_data.get('future', [])

                if _past_wave_list and len(_past_wave_list) > 1:
                    _past_wave_length = len(_past_wave_list)
                    if len(combined_df_for_bytes) >= _past_wave_length:
                        _data_start_idx = len(combined_df_for_bytes) - _past_wave_length
                        for i_wave in range(_past_wave_length - 1):
                            _direction_change = (_past_wave_list[i_wave] > _past_wave_list[i_wave+1]) != \
                                               (_past_wave_list[i_wave-1] > _past_wave_list[i_wave] if i_wave > 0 else False)
                            if _direction_change and cycle_periods_for_combined:
                                combined_df_for_bytes.loc[_data_start_idx + i_wave, 'Cycle'] = cycle_periods_for_combined[0]

                if _future_wave_list:
                    _last_date = combined_df_for_bytes['Date'].iloc[-1] if not combined_df_for_bytes.empty else pd.Timestamp.now(tz='UTC')
                    _time_delta = pd.Timedelta(days=1)
                    if timeframe_for_db == '1h': _time_delta = pd.Timedelta(hours=1)
                    elif timeframe_for_db == '5m': _time_delta = pd.Timedelta(minutes=5)
                    
                    _future_rows = []
                    for i_wave, val_wave in enumerate(_future_wave_list):
                        _future_date = _last_date + (_time_delta * (i_wave + 1))
                        _cycle_period_mark = np.nan
                        if i_wave > 0 and cycle_periods_for_combined and \
                           (_future_wave_list[i_wave] > _future_wave_list[i_wave-1]) != \
                           (_future_wave_list[i_wave-1] > _future_wave_list[i_wave-2] if i_wave > 1 else False):
                             _cycle_period_mark = cycle_periods_for_combined[0]
                        _future_rows.append({'Date': _future_date, 'Cycle': _cycle_period_mark})
                    if _future_rows:
                        _future_df_for_bytes = pd.DataFrame(_future_rows)
                        combined_df_for_bytes = pd.concat([combined_df_for_bytes, _future_df_for_bytes], ignore_index=True)

            if not combined_df_for_bytes.empty:
                st.session_state.cached_combined_csv_filename = f"{symbol_for_db}_combined_{current_download_timestamp}.csv"
                st.session_state.cached_combined_csv_download_bytes = st.session_state.data_exporter.export_to_streamlit_download(
                    combined_df_for_bytes, st.session_state.cached_combined_csv_filename, 'csv'
                )
            else:
                st.session_state.cached_combined_csv_filename = None
                st.session_state.cached_combined_csv_download_bytes = None
            
            # Reset the flag
            st.session_state.rerun_wave_summation = False
            
            print("🔍 DEBUG: Wave summation re-run completed.")

# Check and run auto-recalculation (runs globally, independent of analysis results)
check_and_run_auto_recalculation()

# Display auto-recalculation status if enabled
if st.session_state.get('schedule_enabled', False):
    with st.expander("⏰ Auto-Recalculation Status", expanded=False):
        display_auto_recalc_status()

# Show "Run Cycle Engine" instruction when data is loaded but calculation hasn't been run
if ohlc_data is not None and not ohlc_data.empty:
    print(f"🔍 DEBUG: Data available, checking run_calculation flag")
    print(f"🔍 DEBUG: run_calculation = {st.session_state.get('run_calculation', False)}")
    
    if not st.session_state.get('run_calculation', False): # If "Run Cycle Engine" was NOT just clicked
        if 'cached_fig_for_display' in st.session_state and \
           'cached_table_for_display' in st.session_state and \
           not st.session_state.get('calculation_needed', True) and \
           not st.session_state.get('rerun_wave_summation', False): # Also ensure no pending wave summation
            # Results are cached and no new calculation is needed, display from cache
            display_analysis_results(ohlc_data, data_source_name, perform_full_computation=False)
        else:
            # Data loaded, but no analysis run yet, or settings changed making prior cache invalid for display without re-run
            st.info("📊 **Data loaded successfully!** Click '🔄 Run Cycle Engine' in the sidebar to start analysis.")
            # Show basic data info
            st.write(f"**Data Source:** {data_source_name}")
            st.write(f"**Rows:** {len(ohlc_data):,}")
            if hasattr(ohlc_data.index, 'min'): # Check if ohlc_data.index exists and has min method
                try:
                    st.write(f"**Date Range:** {ohlc_data.index.min()} to {ohlc_data.index.max()}")
                except AttributeError: # Handle cases where index might not be datetime-like after all
                    st.write(f"**Date Range:** Index not suitable for min/max date display.")

            if st.session_state.get('calculation_needed', True): 
                # If settings changed or new data invalidates cache, clear old display cache items
                # (Download byte caches are cleared in on_setting_change_with_calc_flag)
                if 'cached_fig_for_display' in st.session_state: del st.session_state.cached_fig_for_display
                if 'cached_table_for_display' in st.session_state: del st.session_state.cached_table_for_display
                if 'cached_plot_timestamp' in st.session_state: del st.session_state.cached_plot_timestamp
    else: # st.session_state.run_calculation IS True (user clicked "Run" or auto-recalc)
        print(f"🔍 DEBUG: Running analysis because run_calculation = True")
        display_analysis_results(ohlc_data, data_source_name, perform_full_computation=True) # Perform full computation
        st.session_state.run_calculation = False # Reset the flag
else:
    print(f"🔍 DEBUG: No data available for analysis")
    print(f"🔍 DEBUG: ohlc_data is None: {ohlc_data is None}")
    if ohlc_data is not None:
        print(f"🔍 DEBUG: ohlc_data.empty: {ohlc_data.empty}")

print(f"🔍 DEBUG: App execution completed")

# Clean up session state when no data is available
if st.session_state.data_source == 'file' and uploaded_file is None:
    if 'ohlc_data_for_length_check_processed' in st.session_state:
        del st.session_state.ohlc_data_for_length_check_processed
    if 'live_data' in st.session_state:
        del st.session_state.live_data