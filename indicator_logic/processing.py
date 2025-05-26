# indicator_logic/processing.py

"""
Data processing functions for cycle analysis, including detrending
and Bartels cycle significance testing.
"""

import pandas as pd
import numpy as np
import math

# Import necessary functions from core_algorithms
# Relative import since core_algorithms.py is in the same package
from .core_algorithms import zero_lag_ma, hodrick_prescott_filter, bartels_prob

# Import constants from settings (e.g., for DTmethod comparison)
from .settings import HPSMTHDT, ZLAGSMTHDT # Add other constants if needed by these funcs

# --- Detrend Logarithmic Zero-lag Regression ---
def detrend_ln_zero_lag_regression(src: np.ndarray | pd.Series,
                                   smooth_per: int,
                                   bars_taken: int) -> np.ndarray:
    """
    Detrends a source series using logarithmic zero-lag regression.
    Applies zeroLagMA, takes the log, performs linear regression,
    and subtracts the regression line.
    'src' is assumed to be [most_recent, ..., oldest].
    Identical to indicator_milestone_1.py.
    """
    # print(f"\n--- DEBUG DETREND_LN_ZLR ---")
    # print(f"Entering with smooth_per={smooth_per}, bars_taken={bars_taken}, src length={len(src)}")

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series): src = src.values
        else:
            try: src = np.array(src, dtype=float)
            except Exception as e: print(f"Error converting src to numpy array in detrend_ln_zlr: {e}"); return np.array([])

    if bars_taken > len(src):
        # print(f"Warning: bars_taken ({bars_taken}) > src length ({len(src)}) in detrend_ln_zlr. Adjusting.")
        bars_taken = len(src)
    if bars_taken <= 0 or smooth_per <= 0:
        print(f"Error: bars_taken ({bars_taken}) and smooth_per ({smooth_per}) must be positive in detrend_ln_zlr.")
        return np.array([])

    # Original script: src_subset = src[-bars_taken:]
    # If src is [d9, d8 ... d0] (most recent d0)
    # src[-5:] is [d4, d3, d2, d1, d0] which is correct for ZLMA (most recent of subset, first)
    # If srcVal (reversed) is passed, src[-bars_taken:] gets the OLDEST segment of srcVal.
    # However, the zero_lag_ma function expects its input `src` to be most_recent_first.
    # If src input to this detrend_ln_zlr is ALREADY most_recent_first (like srcVal),
    # then src_subset = src[:bars_taken] is correct for ZLMA.
    # The original code had:
    #   srcVal = srcVal_from_file[::-1] # reversed, most_recent_first
    #   srcVal = detrend_ln_zero_lag_regression(srcVal, DT_RegZLsmoothPer, SampleSize)
    #   Inside detrend_ln_zero_lag_regression:
    #     src_subset = src[-bars_taken:] # THIS IS THE KEY LINE FROM ORIGINAL
    #     calc_values = zero_lag_ma(src_subset, smooth_per, bars_taken)
    # If src is [d9,d8,d7,d6,d5,d4,d3,d2,d1,d0] (len 10) and bars_taken=5.
    # src_subset = src[-5:] = [d4,d3,d2,d1,d0]. This is correctly ordered for zero_lag_ma.
    # This logic from the original script is maintained.
    src_subset_for_calc = src[-bars_taken:]

    if len(src_subset_for_calc) == 0: return np.array([]) # Should be caught by bars_taken <=0

    calc_values = zero_lag_ma(src_subset_for_calc, smooth_per, bars_taken) # ZLMA expects most_recent_first

    # Handle potential log(0) or log(negative) issues by ensuring values are positive
    # np.maximum avoids changing original calc_values if already positive
    log_calc_values = np.log(np.maximum(calc_values, 1e-9)) * 100

    x = np.arange(bars_taken) # Independent variable for regression
    y = log_calc_values       # Dependent variable

    sumy = np.sum(y)
    sumx = np.sum(x)
    sumxy = np.sum(x * y)
    sumx2 = np.sum(x * x)

    val3 = sumx2 * bars_taken - sumx * sumx
    output = np.zeros(bars_taken)

    if val3 != 0.0:
        val2 = (sumxy * bars_taken - sumx * sumy) / val3 # Slope
        val1 = (sumy - sumx * val2) / bars_taken        # Intercept
        reg_line = val1 + val2 * x
        output = y - reg_line # Detrended values
    else: # Happens if bars_taken is small (e.g., 1) or x values are not distinct
        if bars_taken > 0:
            output = y - np.mean(y) # Detrend by subtracting the mean
        # else output remains zeros

    # print(f"--- END DEBUG DETREND_LN_ZLR ---\n")
    return output

# --- Bartels cycle significance test ---
def bartels_cycle_test(src: np.ndarray | pd.Series,
                       G: int, # Number of cycles to test
                       cyclebuffer: np.ndarray, # 1-indexed periods from Goertzel (cyclebuffer[0] unused or 0)
                       BartelsProb: np.ndarray, # Output array, 0-indexed by cycle rank (0 to G-1)
                       BartNoCycles: int, # Bartels N parameter (number of segments for test)
                       BartSmoothPer: int): # Bartels smoothing for internal detrend
    """
    Performs the Bartels cycle significance test.
    'src' is data (e.g. srcVal from Goertzel, most_recent_first).
    'cyclebuffer' is 1-indexed from Goertzel but accessed 0-indexed here.
    'BartelsProb' (output) is 0-indexed.
    Identical to indicator_milestone_1.py.
    """
    # print(f"\n--- DEBUG BARTELS_TEST ---")
    # print(f"Entering with G={G}, BartNoCycles={BartNoCycles}, BartSmoothPer={BartSmoothPer}")

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series): src = src.values
        else:
            try: src = np.array(src, dtype=float)
            except Exception as e: print(f"Error converting src to numpy array in bartels_test: {e}"); return BartelsProb

    # G is number_of_cycles. Loop k from 0 to G-1.
    # cyclebuffer is 1-indexed, but original script accesses cyclebuffer[k].
    # This means cyclebuffer[0] (first cycle period if k=0) is used.
    # Goertzel puts 1st cycle at cyclebuffer[1]. cyclebuffer[0] is usually 0.
    # This is a critical detail of the original script's indexing.
    
    # print(f"Testing {G} cycles...")
    for k_cycle_rank in range(G): # k_cycle_rank is 0 to G-1
        # Original: bpi = round(cyclebuffer[k])
        # If cyclebuffer[0] is 0, bpi becomes 0, and this cycle is skipped.
        # This implicitly handles the 1-indexing of meaningful data in cyclebuffer.
        if k_cycle_rank < len(cyclebuffer):
            bpi = round(cyclebuffer[k_cycle_rank]) # Period of the cycle being tested
        else:
            # print(f"Warning: k_cycle_rank {k_cycle_rank} out of bounds for cyclebuffer in bartels_test.")
            continue

        if bpi > 0:
            required_bval_len = bpi * BartNoCycles
            if required_bval_len > len(src):
                # print(f"Warning: Required length for bval ({required_bval_len}) > src length ({len(src)}). Skipping Bartels for cycle rank {k_cycle_rank}.")
                if k_cycle_rank < len(BartelsProb): BartelsProb[k_cycle_rank] = 0.0 # Explicitly set prob to 0
                continue

            # src is [most_recent, ..., oldest]. src[-N:] takes the OLDEST N points.
            src_for_bval = src[-required_bval_len:]

            # bval is detrended src_for_bval. detrend_ln_zlr expects its input to be most_recent_first.
            # src_for_bval is [oldest_N, oldest_N-1, ..., oldest_1_of_N]. This IS most_recent_first for that segment.
            bval = detrend_ln_zero_lag_regression(src_for_bval, BartSmoothPer, required_bval_len)

            if len(bval) == required_bval_len:
                # bartels_prob expects bval to be flat array for its n*N calculations.
                # n = bpi (period), N = BartNoCycles
                bp_value = bartels_prob(bpi, BartNoCycles, bval)
                if k_cycle_rank < len(BartelsProb):
                    BartelsProb[k_cycle_rank] = (1.0 - bp_value) * 100.0
            else:
                # print(f"Warning: detrend_ln_zlr returned unexpected length for bval. Skipping Bartels for cycle rank {k_cycle_rank}.")
                if k_cycle_rank < len(BartelsProb): BartelsProb[k_cycle_rank] = 0.0
        else: # bpi <= 0
            if k_cycle_rank < len(BartelsProb): BartelsProb[k_cycle_rank] = 0.0


    # print(f"--- END DEBUG BARTELS_TEST ---\n")
    return BartelsProb # Modified in place

# --- Detrending options using Centered Moving Averages ---
def detrend_centered_ma(src: np.ndarray | pd.Series,
                        period1: int,
                        period2: int,
                        calcBars: int,
                        DTmethod: str) -> np.ndarray:
    """
    Detrends a source series using centered moving averages.
    'src' is assumed to be [most_recent, ..., oldest] (e.g. srcVal).
    Identical to indicator_milestone_1.py.
    """
    # print(f"\n--- DEBUG DETREND_MA ---")
    # print(f"Entering with DTmethod='{DTmethod}', period1={period1}, period2={period2}, calcBars={calcBars}, src length={len(src)}")

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series): src = src.values
        else:
            try: src = np.array(src, dtype=float)
            except Exception as e: print(f"Error converting src to numpy array in detrend_centered_ma: {e}"); return np.array([])

    if calcBars > len(src):
        # print(f"Warning: calcBars ({calcBars}) > src length ({len(src)}) in detrend_ma. Adjusting.")
        calcBars = len(src)
    if calcBars <= 0:
        print(f"Error: calcBars ({calcBars}) must be positive in detrend_centered_ma.")
        return np.array([])

    # src is [d_calcBars-1, ..., d0] (most_recent_first order)
    # Original: src_subset = src[-calcBars:]
    # If src is already reversed (most_recent_first), src[-calcBars:] takes the oldest part.
    # However, ZLMA and HPF are then called with this src_subset.
    # ZLMA expects its input to be most_recent_first for the segment it processes.
    # HPF also processes the segment it's given.
    # If src is [d9,d8..d0] and calcBars=7, src_subset = src[-7:] = [d6,d5..d0]. This is correct.
    src_subset_for_calc = src[-calcBars:]
    if len(src_subset_for_calc) == 0 : return np.array([])

    output = np.copy(src_subset_for_calc) # Initialize output
    calcValues1 = np.array([])
    calcValues2 = np.array([])

    if DTmethod == HPSMTHDT:
        if period1 <= 0 or period2 <= 0:
             print(f"Error: period1 ({period1}) and period2 ({period2}) must be positive for HP filter in detrend_ma.")
             return np.zeros(calcBars) # Or return src_subset_for_calc

        sin_pi_p1 = math.sin(math.pi / period1) if period1 > 0 else 0
        sin_pi_p2 = math.sin(math.pi / period2) if period2 > 0 else 0

        # Avoid division by zero for lambda calculation
        lamb1 = 0.0625 / math.pow(sin_pi_p1, 4) if sin_pi_p1 != 0 else float('inf')
        lamb2 = 0.0625 / math.pow(sin_pi_p2, 4) if sin_pi_p2 != 0 else float('inf')

        if lamb1 == float('inf') or lamb2 == float('inf'):
             print(f"Error: Division by zero calculating lambda for HP filter in detrend_ma. Check periods.")
             return np.zeros(calcBars)

        # HPF expects src, lambda, num_bars. It internally takes src[-num_bars:].
        # src_subset_for_calc is already the correct segment and order for HPF to process.
        calcValues1 = hodrick_prescott_filter(src_subset_for_calc, lamb1, calcBars)
        calcValues2 = hodrick_prescott_filter(src_subset_for_calc, lamb2, calcBars)

    elif DTmethod == ZLAGSMTHDT:
        if period1 <= 0 or period2 <= 0:
             print(f"Error: period1 ({period1}) and period2 ({period2}) must be positive for ZLMA in detrend_ma.")
             return np.zeros(calcBars)

        # ZLMA expects src (most_recent_first), smooth_per, num_bars.
        # src_subset_for_calc is already the correct segment and order.
        calcValues1 = zero_lag_ma(src_subset_for_calc, period1, calcBars)
        calcValues2 = zero_lag_ma(src_subset_for_calc, period2, calcBars)
    else:
        # print(f"Warning: Unknown detrending method '{DTmethod}' in detrend_centered_ma. No detrending applied.")
        return src_subset_for_calc # Return original subset if method is unknown

    if len(calcValues1) == calcBars and len(calcValues2) == calcBars:
        # output = calcValues1 - calcValues2 # Element-wise subtraction
        for i in range(calcBars): # Original script loop
            output[i] = calcValues1[i] - calcValues2[i]
    else:
         print(f"Error: Filter returned unexpected length in detrend_centered_ma. Expected {calcBars}.")
         return np.zeros(calcBars) # Or return src_subset_for_calc

    # print(f"--- END DEBUG DETREND_MA ---\n")
    return output


if __name__ == '__main__':
    print("--- Testing processing.py ---")

    # Test detrend_ln_zero_lag_regression
    print("\nTesting detrend_ln_zero_lag_regression...")
    # src_test_ln should be most_recent_first
    src_test_ln = np.array([110, 108, 105, 106, 104, 102, 100, 98, 95, 96.0, 94, 92, 90, 88, 85])
    smooth_per_ln = 5
    bars_taken_ln = 10 # Process last 10 points of src_test_ln
    
    # src_test_ln is [d14, d13, ..., d0].
    # detrend_ln_zero_lag_regression gets src_subset = src_test_ln[-10:] = [d9, d8, ..., d0]
    # This subset is [96.0, 94, 92, 90, 88, 85]. This is correct "most_recent_first" for ZLMA.
    ln_detrended = detrend_ln_zero_lag_regression(src_test_ln, smooth_per_ln, bars_taken_ln)
    print(f"Original data for ln_detrend (last {bars_taken_ln}): {src_test_ln[-bars_taken_ln:]}")
    print(f"Log-ZLR Detrended (first 5): {np.round(ln_detrended[:5], 4)}")
    print(f"Log-ZLR Detrended (all): {np.round(ln_detrended, 4)}")


    # Test detrend_centered_ma with ZLMA
    print("\nTesting detrend_centered_ma (ZLMA method)...")
    src_test_cma = np.array([100,102,101,103,105,104,106,108,107,109,110.0, 109,108,107,106]) # most_recent_first
    p1_cma = 5
    p2_cma = 10
    bars_cma = 12 # Process last 12 points of src_test_cma
    cma_detrended_zl = detrend_centered_ma(src_test_cma, p1_cma, p2_cma, bars_cma, ZLAGSMTHDT)
    print(f"Original data for cma_detrend (last {bars_cma}): {src_test_cma[-bars_cma:]}")
    print(f"Centered MA Detrended (ZLMA) (first 5): {np.round(cma_detrended_zl[:5], 4)}")

    # Test bartels_cycle_test (mocking dependencies)
    print("\nTesting bartels_cycle_test (mock setup)...")
    G_test = 2
    # cyclebuffer needs to be 1-indexed for meaning, but bartels_cycle_test accesses it 0-indexed.
    # So, cyclebuffer[0] = period of 1st cycle, cyclebuffer[1] = period of 2nd cycle.
    # This matches original if goertzel populates cyclebuffer[0] for 1st cycle (which it does not, it starts at [1]).
    # This implies a slight mismatch in how cyclebuffer is populated vs used by bartels_cycle_test in original.
    # Let's follow indicator_milestone_1.py's bartels_cycle_test which does `round(cyclebuffer[k])` where k is 0-indexed.
    # If Goertzel puts 1st cycle at cyclebuffer[1], 2nd at cyclebuffer[2], then Bartels Test using cyclebuffer[0] and cyclebuffer[1]
    # would effectively test cyclebuffer[0] (usually 0) and cyclebuffer[1] (1st real cycle).
    # This is how the original script behaved.
    
    test_cyclebuffer = np.zeros(G_test + 1) # +1 to allow 1-based indexing if needed elsewhere
    test_cyclebuffer[0] = 20 # Mimicking original script: bartels_cycle_test access cyclebuffer[0]
    test_cyclebuffer[1] = 30 # Mimicking original script: bartels_cycle_test access cyclebuffer[1]

    test_bartelsprob_arr = np.zeros(G_test)
    test_src_bartels = np.sin(np.arange(200) * 2 * np.pi / 20) + 0.5 * np.random.randn(200) # recent first
    test_src_bartels = test_src_bartels[::-1] # ensure recent first

    result_bartelsprob = bartels_cycle_test(test_src_bartels, G_test, test_cyclebuffer, test_bartelsprob_arr, 5, 2)
    print(f"Bartels Probabilities from test: {np.round(result_bartelsprob, 2)}")
    print(f"Cycle periods tested (cyclebuffer[0], cyclebuffer[1]): {test_cyclebuffer[0]}, {test_cyclebuffer[1]}")