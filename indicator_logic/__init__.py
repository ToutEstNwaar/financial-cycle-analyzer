# indicator_logic/__init__.py

"""
Initialization file for the indicator_logic package.

This file makes Python treat the 'indicator_logic' directory as a package,
allowing for organized module imports.

You can also use this file to expose specific functions or classes
from submodules at the package level for easier access, e.g.:
  from .data_loader import load_ohlc_from_csv
  from .core_algorithms import goertzel
We will keep it simple for now.
"""

# You can leave this file empty if you don't need to execute any package-level
# initialization code or define what's imported when 'from indicator_logic import *' is used.

# Example of how you might expose functions later (optional):
# from .settings import default_settings
# from .data_loader import load_ohlc_from_csv
# from .core_algorithms import zero_lag_ma, hodrick_prescott_filter, goertzel, bartels_prob
# from .processing import detrend_centered_ma, detrend_ln_zero_lag_regression, bartels_cycle_test
# from .main_calculator import run_full_analysis

print("indicator_logic package initialized") # Optional: for confirming import