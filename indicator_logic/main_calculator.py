# indicator_logic/main_calculator.py

"""
Main calculation orchestrator for the cycle indicator.
Separates core cycle component calculation from wave summation for optimized caching.
"""

import numpy as np
import pandas as pd
import math

from . import settings as ind_settings
from .core_algorithms import zero_lag_ma as zlma_core, \
                             hodrick_prescott_filter as hpf_core, \
                             goertzel as goertzel_core
from .processing import detrend_centered_ma, \
                        detrend_ln_zero_lag_regression, \
                        bartels_cycle_test

def _calculate_src_series(ohlc_data_df: pd.DataFrame, source_price_type: str) -> np.ndarray | None:
    """Helper to calculate the source series based on user selection."""
    if not all(col in ohlc_data_df.columns for col in ['open', 'high', 'low', 'close']):
        print("Error: OHLC data must contain 'open', 'high', 'low', 'close' columns.")
        return None

    if source_price_type == "Open": src = ohlc_data_df['open'].values
    elif source_price_type == "High": src = ohlc_data_df['high'].values
    elif source_price_type == "Low": src = ohlc_data_df['low'].values
    elif source_price_type == "Close": src = ohlc_data_df['close'].values
    elif source_price_type == "(H+L)/2": src = ((ohlc_data_df['high'] + ohlc_data_df['low']) / 2).values
    elif source_price_type == "(H+L+C)/3": src = ((ohlc_data_df['high'] + ohlc_data_df['low'] + ohlc_data_df['close']) / 3).values
    elif source_price_type == "(O+H+L+C)/4": src = ((ohlc_data_df['open'] + ohlc_data_df['high'] + ohlc_data_df['low'] + ohlc_data_df['close']) / 4).values
    elif source_price_type == "(H+L+C+C)/4": src = ((ohlc_data_df['high'] + ohlc_data_df['low'] + ohlc_data_df['close'] + ohlc_data_df['close']) / 4).values
    else:
        print(f"Warning: Unknown source_price_type '{source_price_type}', defaulting to 'Close'.")
        src = ohlc_data_df['close'].values
    
    return np.array(src) if src is not None else None


def calculate_core_cycle_components(
    ohlc_data_df: pd.DataFrame,
    # --- Group A Settings (Affect core cycle detection) ---
    source_price_type: str,
    MaxPer: int,
    WindowSizePast_base: int, # Base value before adjustment
    WindowSizeFuture_base: int, # Base value before adjustment
    detrendornot: str,
    DT_ZLper1: int, DT_ZLper2: int, DT_HPper1: int, DT_HPper2: int, DT_RegZLsmoothPer: int,
    HPsmoothPer: int, ZLMAsmoothPer: int,
    BarToCalculate: int,
    FilterBartels: bool, BartNoCycles: int, BartSmoothPer: int, BartSigLimit: float, SortBartels: bool,
    squaredAmp: bool, UseCycleStrength: bool, useAddition: bool, useCosine: bool
    ):
    """
    Performs the expensive part of the calculation: data prep, detrending/smoothing,
    and Goertzel analysis to find cycle components and individual wave matrices.
    This function's output can be cached by Streamlit.
    """
    core_results = {"status": "success", "message": ""}

    src_series = _calculate_src_series(ohlc_data_df, source_price_type)
    if src_series is None:
        core_results["status"] = "error"
        core_results["message"] = "Failed to calculate source series from OHLC data."
        return core_results

    # Adjust window sizes
    current_WindowSizePast = max(WindowSizePast_base, 2 * MaxPer)
    current_WindowSizeFuture = max(WindowSizeFuture_base, 2 * MaxPer)
    core_results['current_WindowSizePast'] = current_WindowSizePast # Pass adjusted size out
    core_results['current_WindowSizeFuture'] = current_WindowSizeFuture # Pass adjusted size out


    calculated_SampleSize = BartNoCycles * MaxPer + BarToCalculate
    core_results['calculated_SampleSize'] = calculated_SampleSize
    
    calculation_bar_idx = len(ohlc_data_df) - 1 # Based on the full dataset
    core_results['calculation_bar_idx'] = calculation_bar_idx

    if len(src_series) < calculated_SampleSize:
        core_results["status"] = "error"
        core_results["message"] = f"Not enough data ({len(src_series)}) for SampleSize {calculated_SampleSize}."
        return core_results

    goeWorkPast = np.zeros((current_WindowSizePast, MaxPer + 1))
    goeWorkFuture = np.zeros((current_WindowSizeFuture, MaxPer + 1))
    cyclebuffer = np.zeros(MaxPer + 1)
    amplitudebuffer = np.zeros(MaxPer + 1)
    phasebuffer = np.zeros(MaxPer + 1)
    cycleBartelsBuffer = np.zeros(MaxPer * 3)

    srcVal_from_file = src_series[-calculated_SampleSize:]
    srcVal = srcVal_from_file[::-1]
    
    processed_srcVal = np.copy(srcVal)
    if detrendornot == ind_settings.HPSMTHDT: processed_srcVal = detrend_centered_ma(srcVal, DT_HPper1, DT_HPper2, calculated_SampleSize, detrendornot)
    elif detrendornot == ind_settings.ZLAGSMTHDT: processed_srcVal = detrend_centered_ma(srcVal, DT_ZLper1, DT_ZLper2, calculated_SampleSize, detrendornot)
    elif detrendornot == ind_settings.LOG_ZLAG_REGRESSION_DT: processed_srcVal = detrend_ln_zero_lag_regression(srcVal, DT_RegZLsmoothPer, calculated_SampleSize)
    elif detrendornot == ind_settings.HPSMTH:
        sin_pi_per = math.sin(math.pi / HPsmoothPer) if HPsmoothPer > 0 else 0
        lamb = 0.0625 / math.pow(sin_pi_per, 4) if sin_pi_per != 0 else float('inf')
        if lamb != float('inf') and lamb >=0: processed_srcVal = hpf_core(srcVal, lamb, calculated_SampleSize)
        else: print(f"Warning: Cannot apply HP smoothing. Lambda invalid ({lamb}). Using original srcVal.")
    elif detrendornot == ind_settings.ZLAGSMTH: processed_srcVal = zlma_core(srcVal, ZLMAsmoothPer, calculated_SampleSize)
    # else: NONE_SMTH_DT or unknown, processed_srcVal remains as srcVal copy

    core_results['final_srcVal_for_debug'] = processed_srcVal # For debug if needed

    if processed_srcVal is None or len(processed_srcVal) != calculated_SampleSize:
        core_results["status"] = "error"
        core_results["message"] = "srcVal processing failed or returned incorrect length."
        return core_results

    number_of_cycles = goertzel_core(
        processed_srcVal, BarToCalculate, calculated_SampleSize, MaxPer,
        squaredAmp, useAddition, useCosine, UseCycleStrength,
        current_WindowSizePast, current_WindowSizeFuture,
        FilterBartels, BartNoCycles, BartSmoothPer, BartSigLimit, SortBartels,
        goeWorkPast, goeWorkFuture,
        cyclebuffer, amplitudebuffer, phasebuffer, cycleBartelsBuffer,
        bartels_cycle_test_func=bartels_cycle_test
    )

    core_results.update({
        'number_of_cycles': number_of_cycles,
        'cyclebuffer': cyclebuffer, 'amplitudebuffer': amplitudebuffer,
        'phasebuffer': phasebuffer, 'cycleBartelsBuffer': cycleBartelsBuffer,
        'goeWorkPast': goeWorkPast, 'goeWorkFuture': goeWorkFuture
    })
    return core_results


def sum_composite_waves(
    core_components: dict,
    # --- Group B Settings (Affect only wave summation) ---
    UseCycleList: bool,
    Cycle1: int, Cycle2: int, Cycle3: int, Cycle4: int, Cycle5: int,
    StartAtCycle: int,
    UseTopCycles: int,
    SubtractNoise: bool
    ):
    """
    Calculates the final composite past and future waves using the
    outputs from calculate_core_cycle_components and Group B settings.
    """
    wave_results = {"status": "success", "message": ""}
    
    # Unpack from core_components
    number_of_cycles = core_components['number_of_cycles']
    goeWorkPast = core_components['goeWorkPast']
    goeWorkFuture = core_components['goeWorkFuture']
    current_WindowSizePast = core_components['current_WindowSizePast']
    current_WindowSizeFuture = core_components['current_WindowSizeFuture']

    goertzel_past_wave = np.zeros(current_WindowSizePast)
    goertzel_future_wave = np.zeros(current_WindowSizeFuture)

    # Calculate past composite wave
    for i in range(current_WindowSizePast):
        sum_val = 0.0
        if UseCycleList:
            cycles_to_use = [c for c in [Cycle1, Cycle2, Cycle3, Cycle4, Cycle5] if c > 0 and c <= number_of_cycles]
            for cycle_idx in cycles_to_use:
                 if i < goeWorkPast.shape[0] and cycle_idx < goeWorkPast.shape[1]: sum_val += goeWorkPast[i, cycle_idx]
        else:
            end_cycle_loop = min(StartAtCycle + UseTopCycles, number_of_cycles + 1)
            for k_rank in range(StartAtCycle, end_cycle_loop):
                 if i < goeWorkPast.shape[0] and k_rank < goeWorkPast.shape[1]: sum_val += goeWorkPast[i, k_rank]
            if SubtractNoise and number_of_cycles >= 1:
                start_noise = StartAtCycle + UseTopCycles
                for k_rank in range(start_noise, number_of_cycles + 1):
                     if i < goeWorkPast.shape[0] and k_rank < goeWorkPast.shape[1]: sum_val -= goeWorkPast[i, k_rank]
        goertzel_past_wave[i] = sum_val

    # Calculate future composite wave (from time-reversed goeWorkFuture)
    for i in range(current_WindowSizeFuture):
        sum_val = 0.0
        if UseCycleList:
            cycles_to_use = [c for c in [Cycle1, Cycle2, Cycle3, Cycle4, Cycle5] if c > 0 and c <= number_of_cycles]
            for cycle_idx in cycles_to_use:
                 if i < goeWorkFuture.shape[0] and cycle_idx < goeWorkFuture.shape[1]: sum_val += goeWorkFuture[i, cycle_idx]
        else:
            end_cycle_loop = min(StartAtCycle + UseTopCycles, number_of_cycles + 1)
            for k_rank in range(StartAtCycle, end_cycle_loop):
                 if i < goeWorkFuture.shape[0] and k_rank < goeWorkFuture.shape[1]: sum_val += goeWorkFuture[i, k_rank]
            if SubtractNoise and number_of_cycles >= 1:
                start_noise = StartAtCycle + UseTopCycles
                for k_rank in range(start_noise, number_of_cycles + 1):
                     if i < goeWorkFuture.shape[0] and k_rank < goeWorkFuture.shape[1]: sum_val -= goeWorkFuture[i, k_rank]
        goertzel_future_wave[i] = sum_val
    
    goertzel_future_wave = goertzel_future_wave[::-1] # Make chronological

    wave_results['goertzel_past_wave'] = goertzel_past_wave
    wave_results['goertzel_future_wave'] = goertzel_future_wave
    return wave_results

# Kept for direct testing or if a single-call interface is ever needed outside Streamlit
def run_indicator_calculations(ohlc_data_df: pd.DataFrame, **all_settings):
    """
    Original single-call function, now orchestrates the two new functions.
    Not typically called directly by Streamlit app if using two-stage caching.
    """
    core_args = {
        "source_price_type": all_settings["source_price_type"], "MaxPer": all_settings["MaxPer"],
        "WindowSizePast_base": all_settings["WindowSizePast"], "WindowSizeFuture_base": all_settings["WindowSizeFuture"],
        "detrendornot": all_settings["detrendornot"], "DT_ZLper1": all_settings["DT_ZLper1"],
        "DT_ZLper2": all_settings["DT_ZLper2"], "DT_HPper1": all_settings["DT_HPper1"],
        "DT_HPper2": all_settings["DT_HPper2"], "DT_RegZLsmoothPer": all_settings["DT_RegZLsmoothPer"],
        "HPsmoothPer": all_settings["HPsmoothPer"], "ZLMAsmoothPer": all_settings["ZLMAsmoothPer"],
        "BarToCalculate": all_settings["BarToCalculate"], "FilterBartels": all_settings["FilterBartels"],
        "BartNoCycles": all_settings["BartNoCycles"], "BartSmoothPer": all_settings["BartSmoothPer"],
        "BartSigLimit": all_settings["BartSigLimit"], "SortBartels": all_settings["SortBartels"],
        "squaredAmp": all_settings["squaredAmp"], "UseCycleStrength": all_settings["UseCycleStrength"],
        "useAddition": all_settings["useAddition"], "useCosine": all_settings["useCosine"]
    }
    core_components = calculate_core_cycle_components(ohlc_data_df, **core_args)

    if core_components["status"] == "error":
        return core_components # Return error dict

    wave_args = {
        "UseCycleList": all_settings["UseCycleList"], "Cycle1": all_settings["Cycle1"],
        "Cycle2": all_settings["Cycle2"], "Cycle3": all_settings["Cycle3"],
        "Cycle4": all_settings["Cycle4"], "Cycle5": all_settings["Cycle5"],
        "StartAtCycle": all_settings["StartAtCycle"], "UseTopCycles": all_settings["UseTopCycles"],
        "SubtractNoise": all_settings["SubtractNoise"]
    }
    wave_results = sum_composite_waves(core_components, **wave_args)

    # Combine results
    final_results = {**core_components, **wave_results}
    return final_results


if __name__ == '__main__':
    print("--- Testing main_calculator.py (with two-stage calculation) ---")
    from .data_loader import load_ohlc_from_csv # For direct test
    import os as test_os # Alias for os specific to this test block

    data_file_path_test = test_os.path.join('indicator_logic', '..', 'btc_daily.csv') # Path relative to this file for testing
    print(f"Attempting to load data from: {data_file_path_test}")
    actual_ohlc_data = load_ohlc_from_csv(data_file_path_test)

    if actual_ohlc_data is not None:
        print(f"Loaded {len(actual_ohlc_data)} rows for testing.")
        test_settings_group_a = {
            "source_price_type": "Close", "MaxPer": ind_settings.DEFAULT_MAX_PERIOD,
            "WindowSizePast_base": ind_settings.DEFAULT_WINDOW_SIZE_PAST,
            "WindowSizeFuture_base": ind_settings.DEFAULT_WINDOW_SIZE_FUTURE,
            "detrendornot": ind_settings.DEFAULT_DETREND_MODE, "DT_ZLper1": ind_settings.DEFAULT_DT_ZL_PERIOD1,
            "DT_ZLper2": ind_settings.DEFAULT_DT_ZL_PERIOD2, "DT_HPper1": ind_settings.DEFAULT_DT_HP_PERIOD1,
            "DT_HPper2": ind_settings.DEFAULT_DT_HP_PERIOD2, "DT_RegZLsmoothPer": ind_settings.DEFAULT_DT_REG_ZL_SMOOTH_PERIOD,
            "HPsmoothPer": ind_settings.DEFAULT_HP_SMOOTH_PERIOD, "ZLMAsmoothPer": ind_settings.DEFAULT_ZLMA_SMOOTH_PERIOD,
            "BarToCalculate": ind_settings.DEFAULT_BAR_TO_CALCULATE, "FilterBartels": ind_settings.DEFAULT_FILTER_BARTELS,
            "BartNoCycles": ind_settings.DEFAULT_BARTELS_NUM_CYCLES, "BartSmoothPer": ind_settings.DEFAULT_BARTELS_SMOOTH_PERIOD,
            "BartSigLimit": ind_settings.DEFAULT_BARTELS_SIG_LIMIT, "SortBartels": ind_settings.DEFAULT_SORT_BARTELS,
            "squaredAmp": ind_settings.DEFAULT_SQUARED_AMPLITUDE, "UseCycleStrength": ind_settings.DEFAULT_USE_CYCLE_STRENGTH,
            "useAddition": ind_settings.DEFAULT_USE_ADDITION, "useCosine": ind_settings.DEFAULT_USE_COSINE
        }
        core_output = calculate_core_cycle_components(actual_ohlc_data, **test_settings_group_a)

        if core_output["status"] == "success":
            print("Core components calculated successfully.")
            test_settings_group_b = {
                "UseCycleList": ind_settings.DEFAULT_USE_CYCLE_LIST, "Cycle1": ind_settings.DEFAULT_CYCLE1,
                "Cycle2": ind_settings.DEFAULT_CYCLE2, "Cycle3": ind_settings.DEFAULT_CYCLE3,
                "Cycle4": ind_settings.DEFAULT_CYCLE4, "Cycle5": ind_settings.DEFAULT_CYCLE5,
                "StartAtCycle": ind_settings.DEFAULT_START_AT_CYCLE, "UseTopCycles": ind_settings.DEFAULT_USE_TOP_CYCLES,
                "SubtractNoise": ind_settings.DEFAULT_SUBTRACT_NOISE
            }
            wave_output = sum_composite_waves(core_output, **test_settings_group_b)
            print(f"Number of cycles: {core_output['number_of_cycles']}")
            print(f"Past wave (first 5): {np.round(wave_output['goertzel_past_wave'][:5], 4)}")
            print(f"Future wave (first 5): {np.round(wave_output['goertzel_future_wave'][:5], 4)}")
        else:
            print(f"Core component calculation failed: {core_output['message']}")
    else:
        print(f"Failed to load data for testing main_calculator.py from {data_file_path_test}")