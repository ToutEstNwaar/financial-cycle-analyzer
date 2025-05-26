# indicator_logic/settings.py

"""
Default settings and constants for the cycle indicator.
"""

import math # For pi if needed for any future default calculations

# --- Color Definitions ---
PAST_COLORS = {
    'up': '#2DD204',    # Green for upward slope in past wave
    'down': '#D2042D'   # Red for downward slope in past wave
}
FUTURE_COLORS = {
    'up': 'fuchsia',  # Color for upward slope in future wave
    'down': 'yellow' # Color for downward slope in future wave
}
FUTURE_FLIP_COLORS = {
    'up': 'fuchsia',   # If last past was down, first future is up
    'down': 'yellow'  # If last past was up, first future is down
}

GREEN_COLOR = '#2DD204'
RED_COLOR = '#D2042D'

# --- Method Name Constants ---
HPSMTH = "Hodrick-Prescott Filter Smoothing"
ZLAGSMTH = "Zero-lag Moving Average Smoothing"
NONE_SMTH_DT = "None" 

HPSMTHDT = "Hodrick-Prescott Filter Detrending"
ZLAGSMTHDT = "Zero-lag Moving Average Detrending"
LOG_ZLAG_REGRESSION_DT = "Logarithmic Zero-lag Moving Average Regression Detrending"

# --- Default Indicator Settings (Updated based on user feedback) ---

# General
DEFAULT_MAX_PERIOD = 50  # Updated
DEFAULT_WINDOW_SIZE_PAST = 120 # Updated
DEFAULT_WINDOW_SIZE_FUTURE = 120 # Updated
DEFAULT_START_AT_CYCLE = 1
DEFAULT_USE_TOP_CYCLES = 2
DEFAULT_BAR_TO_CALCULATE = 1 

# Compound Cycle Settings (if UseCycleList is True)
DEFAULT_CYCLE1 = 1
DEFAULT_CYCLE2 = 1
DEFAULT_CYCLE3 = 1
DEFAULT_CYCLE4 = 1
DEFAULT_CYCLE5 = 1

# Source Price Detrending/Smoothing Settings
DEFAULT_DETREND_MODE = ZLAGSMTH

DEFAULT_DT_ZL_PERIOD1 = 10      
DEFAULT_DT_ZL_PERIOD2 = 40      
DEFAULT_DT_HP_PERIOD1 = 20      
DEFAULT_DT_HP_PERIOD2 = 80      
DEFAULT_DT_REG_ZL_SMOOTH_PERIOD = 5 

DEFAULT_HP_SMOOTH_PERIOD = 20   
DEFAULT_ZLMA_SMOOTH_PERIOD = 10 

# Bartels Cycle Significance Settings
DEFAULT_FILTER_BARTELS = True
DEFAULT_BARTELS_NUM_CYCLES = 5
DEFAULT_BARTELS_SMOOTH_PERIOD = 2
DEFAULT_BARTELS_SIG_LIMIT = 50.0 
DEFAULT_SORT_BARTELS = False

# Miscellaneous Filter Settings
DEFAULT_SQUARED_AMPLITUDE = True
DEFAULT_USE_ADDITION = False
DEFAULT_USE_COSINE = True
DEFAULT_SUBTRACT_NOISE = False
DEFAULT_USE_CYCLE_LIST = False
DEFAULT_USE_CYCLE_STRENGTH = True 

DEFAULT_SHOW_TABLE = True

DEFAULT_SETTINGS_DICT = {
    "MaxPer": DEFAULT_MAX_PERIOD,
    # The keys here should match what streamlit_app.py expects for session state keys
    # If streamlit_app.py uses "WindowSizePast_base", then map it here or use that key directly.
    # Based on previous iterations, streamlit_app.py was modified to use "WindowSizePast_base".
    "WindowSizePast_base": DEFAULT_WINDOW_SIZE_PAST, # Changed key to _base
    "WindowSizeFuture_base": DEFAULT_WINDOW_SIZE_FUTURE, # Changed key to _base
    "StartAtCycle": DEFAULT_START_AT_CYCLE,
    "UseTopCycles": DEFAULT_USE_TOP_CYCLES,
    "BarToCalculate": DEFAULT_BAR_TO_CALCULATE,
    "Cycle1": DEFAULT_CYCLE1,
    "Cycle2": DEFAULT_CYCLE2,
    "Cycle3": DEFAULT_CYCLE3,
    "Cycle4": DEFAULT_CYCLE4,
    "Cycle5": DEFAULT_CYCLE5,
    "detrendornot": DEFAULT_DETREND_MODE,
    "DT_ZLper1": DEFAULT_DT_ZL_PERIOD1,
    "DT_ZLper2": DEFAULT_DT_ZL_PERIOD2,
    "DT_HPper1": DEFAULT_DT_HP_PERIOD1,
    "DT_HPper2": DEFAULT_DT_HP_PERIOD2,
    "DT_RegZLsmoothPer": DEFAULT_DT_REG_ZL_SMOOTH_PERIOD,
    "HPsmoothPer": DEFAULT_HP_SMOOTH_PERIOD,
    "ZLMAsmoothPer": DEFAULT_ZLMA_SMOOTH_PERIOD,
    "FilterBartels": DEFAULT_FILTER_BARTELS,
    "BartNoCycles": DEFAULT_BARTELS_NUM_CYCLES,
    "BartSmoothPer": DEFAULT_BARTELS_SMOOTH_PERIOD,
    "BartSigLimit": DEFAULT_BARTELS_SIG_LIMIT,
    "SortBartels": DEFAULT_SORT_BARTELS,
    "squaredAmp": DEFAULT_SQUARED_AMPLITUDE,
    "useAddition": DEFAULT_USE_ADDITION,
    "useCosine": DEFAULT_USE_COSINE,
    "SubtractNoise": DEFAULT_SUBTRACT_NOISE,
    "UseCycleList": DEFAULT_USE_CYCLE_LIST,
    "UseCycleStrength": DEFAULT_USE_CYCLE_STRENGTH,
    "showTable": DEFAULT_SHOW_TABLE 
}

if __name__ == '__main__':
    print(f"Default Max Period: {DEFAULT_MAX_PERIOD}")
    print(f"Default Window Size Past: {DEFAULT_SETTINGS_DICT['WindowSizePast_base']}")
    print(f"Default Window Size Future: {DEFAULT_SETTINGS_DICT['WindowSizeFuture_base']}")
    import json
    print(json.dumps(DEFAULT_SETTINGS_DICT, indent=2))