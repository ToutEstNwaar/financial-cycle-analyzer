# indicator_logic/settings.py

"""
Default settings and constants for the cycle indicator.
"""

import math # For pi if needed for any future default calculations

# --- Color Definitions ---
# Define colors similar to Pine Script (can be adjusted)
# Pine Script colors: greencolor = #2DD204, redcolor = #D2042D
PAST_COLORS = {
    'up': '#2DD204',    # Green for upward slope in past wave
    'down': '#D2042D'   # Red for downward slope in past wave
}
# Pine Script uses fuchsia/yellow for future lines based on a slightly complex logic
# Let's map based on slope direction directly for clarity here
FUTURE_COLORS = {
    'up': 'fuchsia',  # Color for upward slope in future wave
    'down': 'yellow' # Color for downward slope in future wave
}
# Color for the first future line if it flips direction from the last past segment
FUTURE_FLIP_COLORS = {
    'up': 'fuchsia',   # If last past was down, first future is up
    'down': 'yellow'  # If last past was up, first future is down
}

# --- Global Color Variables (Legacy, consider using dicts above) ---
GREEN_COLOR = '#2DD204'
RED_COLOR = '#D2042D'

# --- Method Name Constants ---
HPSMTH = "Hodrick-Prescott Filter Smoothing"
ZLAGSMTH = "Zero-lag Moving Average Smoothing"
NONE_SMTH_DT = "None" # Added for clarity if no smoothing/detrending

HPSMTHDT = "Hodrick-Prescott Filter Detrending"
ZLAGSMTHDT = "Zero-lag Moving Average Detrending"
LOG_ZLAG_REGRESSION_DT = "Logarithmic Zero-lag Moving Average Regression Detrending"

# --- Default Indicator Settings ---

# General
DEFAULT_MAX_PERIOD = 120
DEFAULT_WINDOW_SIZE_PAST = 100 # Will be adjusted relative to MaxPer in main logic
DEFAULT_WINDOW_SIZE_FUTURE = 100 # Will be adjusted relative to MaxPer in main logic
DEFAULT_START_AT_CYCLE = 1
DEFAULT_USE_TOP_CYCLES = 2
DEFAULT_BAR_TO_CALCULATE = 1 # Typically the most recent bar for analysis
# DEFAULT_RECALCULATE = 0 # This seems like a UI control, not a core setting for calculations

# Compound Cycle Settings (if UseCycleList is True)
DEFAULT_CYCLE1 = 1
DEFAULT_CYCLE2 = 1
DEFAULT_CYCLE3 = 1
DEFAULT_CYCLE4 = 1
DEFAULT_CYCLE5 = 1

# Source Price Detrending/Smoothing Settings
# Options: HPSMTH, ZLAGSMTH, HPSMTHDT, ZLAGSMTHDT, LOG_ZLAG_REGRESSION_DT, NONE_SMTH_DT
DEFAULT_DETREND_MODE = ZLAGSMTH

# Source Price Detrending Settings (used if mode is HPSMTHDT, ZLAGSMTHDT, LOG_ZLAG_REGRESSION_DT)
DEFAULT_DT_ZL_PERIOD1 = 10      # Zero-lag Moving Average Fast Period
DEFAULT_DT_ZL_PERIOD2 = 40      # Zero-lag Moving Average Slow Period
DEFAULT_DT_HP_PERIOD1 = 20      # Hodrick-Prescott Filter Fast Period
DEFAULT_DT_HP_PERIOD2 = 80      # Hodrick-Prescott Filter Slow Period
DEFAULT_DT_REG_ZL_SMOOTH_PERIOD = 5 # Logarithmic Zero-lag Moving Average Regression Period

# Source Price Smoothing Settings (used if mode is HPSMTH, ZLAGSMTH)
DEFAULT_HP_SMOOTH_PERIOD = 20   # Hodrick-Prescott Filter Period
DEFAULT_ZLMA_SMOOTH_PERIOD = 10 # Zero-lag Moving Average Period

# Bartels Cycle Significance Settings
DEFAULT_FILTER_BARTELS = True
DEFAULT_BARTELS_NUM_CYCLES = 5
DEFAULT_BARTELS_SMOOTH_PERIOD = 2
DEFAULT_BARTELS_SIG_LIMIT = 50.0 # Percentage
DEFAULT_SORT_BARTELS = False

# Miscellaneous Filter Settings
DEFAULT_SQUARED_AMPLITUDE = True
DEFAULT_USE_ADDITION = False
DEFAULT_USE_COSINE = True
DEFAULT_SUBTRACT_NOISE = False
DEFAULT_USE_CYCLE_LIST = False
DEFAULT_USE_CYCLE_STRENGTH = True # Use cycle strength instead of amplitude in Goertzel

# UI Options (These might be handled directly in Streamlit state, but defaults are good)
DEFAULT_SHOW_TABLE = True

# You can also define a dictionary to hold all default settings
# for easy passing to functions or initializing Streamlit widgets.
DEFAULT_SETTINGS_DICT = {
    "MaxPer": DEFAULT_MAX_PERIOD,
    "WindowSizePast": DEFAULT_WINDOW_SIZE_PAST,
    "WindowSizeFuture": DEFAULT_WINDOW_SIZE_FUTURE,
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
    "showTable": DEFAULT_SHOW_TABLE # UI specific, might not be passed to backend
}

if __name__ == '__main__':
    # Example of accessing a setting
    print(f"Default Max Period: {DEFAULT_MAX_PERIOD}")
    print(f"Default Detrend Mode: {DEFAULT_DETREND_MODE}")
    print(f"Past Up Color: {PAST_COLORS['up']}")
    print("\nAll default settings dictionary:")
    import json
    print(json.dumps(DEFAULT_SETTINGS_DICT, indent=2))