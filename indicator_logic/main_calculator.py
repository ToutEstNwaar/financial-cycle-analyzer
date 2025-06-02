<<<<<<< HEAD
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
        # print("Error: OHLC data must contain 'open', 'high', 'low', 'close' columns.")
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
        # print(f"Warning: Unknown source_price_type '{source_price_type}', defaulting to 'Close'.")
        src = ohlc_data_df['close'].values
    
    return np.array(src) if src is not None else None


def calculate_core_cycle_components(
    ohlc_data_df: pd.DataFrame,
    # --- Group A Settings (Affect core cycle detection) ---
    source_price_type: str,
    MaxPer: int,
    WindowSizePast_base: int, # User input: now defines analysis window size AND base for past wave display length
    WindowSizeFuture_base: int, # User input: base for future wave display length
    detrendornot: str,
    DT_ZLper1: int, DT_ZLper2: int, DT_HPper1: int, DT_HPper2: int, DT_RegZLsmoothPer: int,
    HPsmoothPer: int, ZLMAsmoothPer: int,
    BarToCalculate: int, # Offset for Goertzel calculation start within the sample
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

    # --- MODIFIED LOGIC FOR SampleSize AND MaxPer VALIDATION ---
    # WindowSizePast_base (user input) now directly determines the SampleSize for analysis.
    calculated_SampleSize = WindowSizePast_base
    core_results['calculated_SampleSize'] = calculated_SampleSize # For reference/debug

    # Validate BarToCalculate against SampleSize
    if not (1 <= BarToCalculate <= calculated_SampleSize):
        core_results["status"] = "error"
        core_results["message"] = (
            f"BarToCalculate ({BarToCalculate}) is out of valid range [1, {calculated_SampleSize}] "
            f"for the chosen WindowSizePast_base ({WindowSizePast_base})."
        )
        return core_results

    # Validate MaxPer against calculated_SampleSize and other settings
    min_allowable_max_per = 2 # A practical minimum for MaxPer
    
    # Constraint 1: From Goertzel's internal processing (needs 2*MaxPer data segment starting at BarToCalculate)
    # BarToCalculate + 2 * MaxPer <= calculated_SampleSize  => MaxPer <= (calculated_SampleSize - BarToCalculate) / 2
    if (calculated_SampleSize - BarToCalculate) < (2 * min_allowable_max_per) : # Not enough room for even the smallest MaxPer
         max_per_constraint_goertzel = min_allowable_max_per -1 # Will cause MaxPer to be invalid
    else:
        max_per_constraint_goertzel = math.floor((calculated_SampleSize - BarToCalculate) / 2)

    # Constraint 2: From Bartels test data requirement (needs BartNoCycles * MaxPer data)
    max_per_constraint_bartels = float('inf')
    if FilterBartels and BartNoCycles > 0:
        if calculated_SampleSize < (BartNoCycles * min_allowable_max_per):
             max_per_constraint_bartels = min_allowable_max_per -1 # Will cause MaxPer to be invalid
        else:
            max_per_constraint_bartels = math.floor(calculated_SampleSize / BartNoCycles)
    
    effective_allowable_max_per = min(max_per_constraint_goertzel, max_per_constraint_bartels)

    if not (min_allowable_max_per <= MaxPer <= effective_allowable_max_per):
        core_results["status"] = "error"
        error_message_parts = [
            f"Max Period ({MaxPer}) is out of the valid range [{min_allowable_max_per}, {effective_allowable_max_per}]."
            f" Based on WindowSizePast_base={WindowSizePast_base} and BarToCalculate={BarToCalculate}.",
            f"  - Goertzel requires MaxPer <= {max_per_constraint_goertzel}.",
        ]
        if FilterBartels and BartNoCycles > 0:
            error_message_parts.append(f"  - Bartels Test (with {BartNoCycles} cycles) requires MaxPer <= {max_per_constraint_bartels}.")
        core_results["message"] = " ".join(error_message_parts)
        return core_results
    # --- END OF MODIFIED LOGIC ---

    # Adjust window sizes for OUTPUT WAVES (goeWorkPast/Future matrices dimensions)
    # These ensure output waves can represent at least one full cycle of MaxPer.
    current_WindowSizePast_output = max(WindowSizePast_base, 2 * MaxPer)
    current_WindowSizeFuture_output = max(WindowSizeFuture_base, 2 * MaxPer)
    core_results['current_WindowSizePast'] = current_WindowSizePast_output
    core_results['current_WindowSizeFuture'] = current_WindowSizeFuture_output
    
    calculation_bar_idx = len(ohlc_data_df) - 1 
    core_results['calculation_bar_idx'] = calculation_bar_idx

    if len(src_series) < calculated_SampleSize:
        # This check might be redundant if ohlc_data length is checked before calling,
        # but good for safety. WindowSizePast_base UI should also be limited by data length.
        core_results["status"] = "error"
        core_results["message"] = (
            f"Not enough data in source ({len(src_series)} bars) "
            f"for the chosen WindowSizePast_base ({calculated_SampleSize} bars)."
        )
        return core_results

    # Initialize buffers and matrices
    # Dimensions depend on MaxPer for buffers, and current_WindowSize_output for work matrices
    goeWorkPast = np.zeros((current_WindowSizePast_output, MaxPer + 1))
    goeWorkFuture = np.zeros((current_WindowSizeFuture_output, MaxPer + 1))
    cyclebuffer = np.zeros(MaxPer + 1)
    amplitudebuffer = np.zeros(MaxPer + 1)
    phasebuffer = np.zeros(MaxPer + 1)
    # MaxPer * 3 for cycleBartelsBuffer was from original, check if still appropriate
    # BartelsProb array in processing.bartels_cycle_test is 0-indexed by G (number_of_cycles found, up to MaxPer).
    # Original Goertzel filled cycleBartelsBuffer based on G.
    # Let's assume MaxPer*3 is generous enough. If number_of_cycles can be up to MaxPer,
    # then cycleBartelsBuffer needs to be at least MaxPer. MaxPer * 3 seems safe.
    cycleBartelsBuffer = np.zeros(MaxPer * 3) 

    # Prepare srcVal for analysis (length = calculated_SampleSize)
    srcVal_from_file = src_series[-calculated_SampleSize:]
    srcVal = srcVal_from_file[::-1] # Reverse to most_recent_first
    
    processed_srcVal = np.copy(srcVal)
    # Detrending/Smoothing applied to the window of size calculated_SampleSize
    if detrendornot == ind_settings.HPSMTHDT: processed_srcVal = detrend_centered_ma(srcVal, DT_HPper1, DT_HPper2, calculated_SampleSize, detrendornot)
    elif detrendornot == ind_settings.ZLAGSMTHDT: processed_srcVal = detrend_centered_ma(srcVal, DT_ZLper1, DT_ZLper2, calculated_SampleSize, detrendornot)
    elif detrendornot == ind_settings.LOG_ZLAG_REGRESSION_DT: processed_srcVal = detrend_ln_zero_lag_regression(srcVal, DT_RegZLsmoothPer, calculated_SampleSize)
    elif detrendornot == ind_settings.HPSMTH:
        if HPsmoothPer <= 0: # Prevent math error
            print(f"Warning: HPsmoothPer ({HPsmoothPer}) must be positive. Skipping HP smoothing.")
        else:
            sin_pi_per = math.sin(math.pi / HPsmoothPer)
            lamb = 0.0625 / math.pow(sin_pi_per, 4) if sin_pi_per != 0 else float('inf')
            print(f"DEBUG HPSMTH: HPsmoothPer={HPsmoothPer}, sin_pi_per={sin_pi_per:.6f}, lamb={lamb:.6f}")
            if lamb != float('inf') and lamb >=0: 
                hp_result = hpf_core(srcVal, lamb, calculated_SampleSize)
                if hp_result is not None and len(hp_result) == calculated_SampleSize:
                    # Original PineScript uses the HP filtered values directly for cycle analysis
                    processed_srcVal = hp_result
                    print(f"DEBUG HPSMTH: Success - processed {len(processed_srcVal)} values (using HP smoothed data)")
                    print(f"DEBUG HPSMTH: HP smoothed stats - min={np.min(processed_srcVal):.6f}, max={np.max(processed_srcVal):.6f}, mean={np.mean(processed_srcVal):.6f}")
                else:
                    print(f"DEBUG HPSMTH: Failed - result length: {len(hp_result) if hp_result is not None else 'None'}")
            else: 
                print(f"Warning: Cannot apply HP smoothing. Lambda invalid ({lamb}). Using original srcVal.")
    elif detrendornot == ind_settings.ZLAGSMTH:
        if ZLMAsmoothPer <=0:
            print(f"Warning: ZLMAsmoothPer ({ZLMAsmoothPer}) must be positive. Skipping ZLMA smoothing.")
        else:
            print(f"DEBUG ZLAGSMTH: ZLMAsmoothPer={ZLMAsmoothPer}, calculated_SampleSize={calculated_SampleSize}")
            zlma_result = zlma_core(srcVal, ZLMAsmoothPer, calculated_SampleSize)
            if zlma_result is not None and len(zlma_result) == calculated_SampleSize:
                # Original PineScript uses the ZLMA smoothed values directly for cycle analysis
                processed_srcVal = zlma_result
                print(f"DEBUG ZLAGSMTH: Success - processed {len(processed_srcVal)} values (using ZLMA smoothed data)")
                print(f"DEBUG ZLAGSMTH: ZLMA smoothed stats - min={np.min(processed_srcVal):.6f}, max={np.max(processed_srcVal):.6f}, mean={np.mean(processed_srcVal):.6f}")
            else:
                print(f"DEBUG ZLAGSMTH: Failed - result length: {len(zlma_result) if zlma_result is not None else 'None'}")
                # Use original srcVal if ZLMA fails
                print(f"DEBUG ZLAGSMTH: Using original srcVal as fallback")
    # else: NONE_SMTH_DT or unknown, processed_srcVal remains as srcVal copy

    core_results['final_srcVal_for_debug'] = processed_srcVal 
    
    # Debug: Check final processed_srcVal state
    print(f"DEBUG FINAL: detrendornot='{detrendornot}', processed_srcVal length={len(processed_srcVal) if processed_srcVal is not None else 'None'}")
    if processed_srcVal is not None and len(processed_srcVal) > 0:
        print(f"DEBUG FINAL: processed_srcVal stats - min={np.min(processed_srcVal):.6f}, max={np.max(processed_srcVal):.6f}, mean={np.mean(processed_srcVal):.6f}")
        if len(processed_srcVal) >= 10:
            print(f"DEBUG FINAL: processed_srcVal first 10: {processed_srcVal[:10]}")
    else:
        print(f"DEBUG FINAL: processed_srcVal is invalid!")

    if processed_srcVal is None or len(processed_srcVal) != calculated_SampleSize:
        core_results["status"] = "error"
        core_results["message"] = "srcVal processing failed or returned incorrect length."
        return core_results

    # Call Goertzel algorithm
    # The 'samplesize' argument for goertzel_core is our calculated_SampleSize.
    # WindowSizePast/Future for goertzel_core are for the dimensions of goeWorkPast/Future matrices.
    number_of_cycles = goertzel_core(
        processed_srcVal, BarToCalculate, calculated_SampleSize, MaxPer,
        squaredAmp, useAddition, useCosine, UseCycleStrength,
        current_WindowSizePast_output, current_WindowSizeFuture_output, # For goeWork matrices
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
    # DEBUG: Trace wave summation inputs
    print(f"DEBUG WAVE_SUM: sum_composite_waves called with:")
    print(f"  UseCycleList: {UseCycleList}")
    print(f"  Cycles: [{Cycle1}, {Cycle2}, {Cycle3}, {Cycle4}, {Cycle5}]")
    print(f"  StartAtCycle: {StartAtCycle}, UseTopCycles: {UseTopCycles}")
    print(f"  SubtractNoise: {SubtractNoise}")
    
    wave_results = {"status": "success", "message": ""}
    
    # Unpack from core_components
    number_of_cycles = core_components['number_of_cycles']
    goeWorkPast = core_components['goeWorkPast']       # Matrix of individual past cycle waves
    goeWorkFuture = core_components['goeWorkFuture']     # Matrix of individual future cycle waves
    # These are the actual dimensions of goeWorkPast/Future and the length of summed waves
    current_WindowSizePast_output = core_components['current_WindowSizePast'] 
    current_WindowSizeFuture_output = core_components['current_WindowSizeFuture']

    print(f"DEBUG WAVE_SUM: Core components unpacked:")
    print(f"  number_of_cycles: {number_of_cycles}")
    print(f"  goeWorkPast shape: {goeWorkPast.shape}")
    print(f"  goeWorkFuture shape: {goeWorkFuture.shape}")
    print(f"  current_WindowSizePast_output: {current_WindowSizePast_output}")
    print(f"  current_WindowSizeFuture_output: {current_WindowSizeFuture_output}")
    
    # Check if we have any valid cycles
    if number_of_cycles <= 0:
        print(f"DEBUG WAVE_SUM: No cycles found (number_of_cycles={number_of_cycles}), returning zeros")

    goertzel_past_wave = np.zeros(current_WindowSizePast_output)
    goertzel_future_wave = np.zeros(current_WindowSizeFuture_output)

    # DEBUG: Check goeWork matrices content
    if number_of_cycles > 0:
        print(f"DEBUG WAVE_SUM: Examining goeWork matrices content:")
        for cycle_idx in range(min(3, number_of_cycles, goeWorkPast.shape[1]-1)):  # Check first few cycles
            cycle_rank = cycle_idx + 1  # Convert to 1-based rank
            if cycle_rank < goeWorkPast.shape[1]:
                past_cycle_data = goeWorkPast[:, cycle_rank]
                future_cycle_data = goeWorkFuture[:, cycle_rank]
                
                print(f"  Cycle {cycle_rank} (index {cycle_idx}):")
                print(f"    Past data length: {len(past_cycle_data)}, unique values: {len(np.unique(past_cycle_data))}")
                if len(past_cycle_data) > 0:
                    print(f"    Past stats: min={np.min(past_cycle_data):.6f}, max={np.max(past_cycle_data):.6f}, mean={np.mean(past_cycle_data):.6f}")
                    print(f"    Past first 5: {past_cycle_data[:5]}")
                    if np.all(past_cycle_data == 0):
                        print(f"    ⚠️ Past cycle {cycle_rank} is ALL ZEROS!")
                
                print(f"    Future data length: {len(future_cycle_data)}, unique values: {len(np.unique(future_cycle_data))}")
                if len(future_cycle_data) > 0:
                    print(f"    Future stats: min={np.min(future_cycle_data):.6f}, max={np.max(future_cycle_data):.6f}, mean={np.mean(future_cycle_data):.6f}")
                    print(f"    Future first 5: {future_cycle_data[:5]}")
                    if np.all(future_cycle_data == 0):
                        print(f"    ⚠️ Future cycle {cycle_rank} is ALL ZEROS!")

    # Calculate past composite wave
    # Summing columns of goeWorkPast matrix
    for i_time_step in range(current_WindowSizePast_output):
        sum_val = 0.0
        if UseCycleList:
            # Use specific cycles by their rank (1-indexed in UI, adjust for 0-indexed matrix column if needed by goeWork matrix structure)
            # Assuming goeWorkPast columns are 1-indexed for cycle rank (Cycle1=1st col, etc.) based on original Pine
            cycles_to_use = [c for c in [Cycle1, Cycle2, Cycle3, Cycle4, Cycle5] if c > 0 and c <= number_of_cycles and c < goeWorkPast.shape[1]]
            for cycle_rank_col_idx in cycles_to_use: # cycle_rank_col_idx is 1-based
                 if i_time_step < goeWorkPast.shape[0] : sum_val += goeWorkPast[i_time_step, cycle_rank_col_idx]
        else:
            # Use top N cycles starting from StartAtCycle rank
            # Assuming StartAtCycle is 1-indexed rank
            actual_start_rank = StartAtCycle # 1-indexed
            # Max rank to sum is StartAtCycle + UseTopCycles - 1
            # Loop up to min(actual_start_rank + UseTopCycles, number_of_cycles + 1)
            end_cycle_loop = min(actual_start_rank + UseTopCycles, number_of_cycles + 1) 
            for k_rank in range(actual_start_rank, end_cycle_loop): # k_rank is 1-indexed
                 if i_time_step < goeWorkPast.shape[0] and k_rank < goeWorkPast.shape[1]: sum_val += goeWorkPast[i_time_step, k_rank]
            
            if SubtractNoise and number_of_cycles >= 1:
                # Noise cycles are those after the 'UseTopCycles'
                start_noise_rank = actual_start_rank + UseTopCycles # 1-indexed
                for k_rank_noise in range(start_noise_rank, number_of_cycles + 1): # k_rank_noise is 1-indexed
                     if i_time_step < goeWorkPast.shape[0] and k_rank_noise < goeWorkPast.shape[1]: sum_val -= goeWorkPast[i_time_step, k_rank_noise]
        goertzel_past_wave[i_time_step] = sum_val

    # Calculate future composite wave (from time-reversed goeWorkFuture)
    # Summing columns of goeWorkFuture matrix
    for i_time_step in range(current_WindowSizeFuture_output):
        sum_val = 0.0
        if UseCycleList:
            cycles_to_use = [c for c in [Cycle1, Cycle2, Cycle3, Cycle4, Cycle5] if c > 0 and c <= number_of_cycles and c < goeWorkFuture.shape[1]]
            for cycle_rank_col_idx in cycles_to_use:
                 if i_time_step < goeWorkFuture.shape[0]: sum_val += goeWorkFuture[i_time_step, cycle_rank_col_idx]
        else:
            actual_start_rank = StartAtCycle
            end_cycle_loop = min(actual_start_rank + UseTopCycles, number_of_cycles + 1)
            for k_rank in range(actual_start_rank, end_cycle_loop):
                 if i_time_step < goeWorkFuture.shape[0] and k_rank < goeWorkFuture.shape[1]: sum_val += goeWorkFuture[i_time_step, k_rank]

            if SubtractNoise and number_of_cycles >= 1:
                start_noise_rank = actual_start_rank + UseTopCycles
                for k_rank_noise in range(start_noise_rank, number_of_cycles + 1):
                     if i_time_step < goeWorkFuture.shape[0] and k_rank_noise < goeWorkFuture.shape[1]: sum_val -= goeWorkFuture[i_time_step, k_rank_noise]
        goertzel_future_wave[i_time_step] = sum_val
    
    # goeWorkFuture was stored time-reversed by Goertzel function to match Pine logic.
    # So, the sum is also time-reversed. Reverse it back for chronological future wave.
    goertzel_future_wave = goertzel_future_wave[::-1] 

    # DEBUG: Check final wave results
    print(f"DEBUG WAVE_SUM: Final wave results:")
    print(f"  goertzel_past_wave length: {len(goertzel_past_wave)}")
    print(f"  goertzel_future_wave length: {len(goertzel_future_wave)}")
    
    if len(goertzel_past_wave) > 0:
        unique_past = np.unique(goertzel_past_wave)
        print(f"  Past wave unique values: {len(unique_past)} (first 5: {unique_past[:5] if len(unique_past) > 0 else 'None'})")
        print(f"  Past wave stats: min={np.min(goertzel_past_wave):.6f}, max={np.max(goertzel_past_wave):.6f}, mean={np.mean(goertzel_past_wave):.6f}")
        if np.all(goertzel_past_wave == 0):
            print(f"  ⚠️ FINAL PAST WAVE IS ALL ZEROS!")
        elif len(unique_past) == 1:
            print(f"  ⚠️ FINAL PAST WAVE IS ALL THE SAME VALUE: {unique_past[0]}")
    
    if len(goertzel_future_wave) > 0:
        unique_future = np.unique(goertzel_future_wave)
        print(f"  Future wave unique values: {len(unique_future)} (first 5: {unique_future[:5] if len(unique_future) > 0 else 'None'})")
        print(f"  Future wave stats: min={np.min(goertzel_future_wave):.6f}, max={np.max(goertzel_future_wave):.6f}, mean={np.mean(goertzel_future_wave):.6f}")
        if np.all(goertzel_future_wave == 0):
            print(f"  ⚠️ FINAL FUTURE WAVE IS ALL ZEROS!")
        elif len(unique_future) == 1:
            print(f"  ⚠️ FINAL FUTURE WAVE IS ALL THE SAME VALUE: {unique_future[0]}")

    wave_results['goertzel_past_wave'] = goertzel_past_wave
    wave_results['goertzel_future_wave'] = goertzel_future_wave
    return wave_results

# Kept for direct testing or if a single-call interface is ever needed outside Streamlit
def run_indicator_calculations(ohlc_data_df: pd.DataFrame, **all_settings):
    """
    Original single-call function, now orchestrates the two new functions.
    Not typically called directly by Streamlit app if using two-stage caching.
    """
    # Note: The keys for WindowSizePast/Future in all_settings might be the original ones.
    # The calculate_core_cycle_components expects 'WindowSizePast_base' and 'WindowSizeFuture_base'.
    # This mapping should align with how settings are stored/passed if this function is used.
    # Assuming all_settings uses the '_base' suffix for consistency with streamlit_app.py state keys.
    core_args = {
        "source_price_type": all_settings["source_price_type"], "MaxPer": all_settings["MaxPer"],
        "WindowSizePast_base": all_settings.get("WindowSizePast_base", all_settings.get("WindowSizePast")), # Adapt if key name varies
        "WindowSizeFuture_base": all_settings.get("WindowSizeFuture_base", all_settings.get("WindowSizeFuture")), # Adapt
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
    print("--- Testing main_calculator.py (with two-stage calculation and new SampleSize logic) ---")
    from .data_loader import load_ohlc_from_csv 
    import os as test_os 

    # Assuming btc_daily.csv is in the parent directory of indicator_logic
    # Adjust path if your project structure is different
    current_dir = test_os.path.dirname(test_os.path.abspath(__file__))
    data_file_path_test = test_os.path.join(current_dir, '..', 'btc_daily.csv') 
    
    print(f"Attempting to load data from: {data_file_path_test}")
    actual_ohlc_data = load_ohlc_from_csv(data_file_path_test)

    if actual_ohlc_data is not None and not actual_ohlc_data.empty:
        print(f"Loaded {len(actual_ohlc_data)} rows for testing.")

        # Test Case 1: Valid settings based on new logic
        print("\n--- Test Case 1: Valid Settings ---")
        test_settings_valid = {
            "source_price_type": "Close", 
            "MaxPer": 20, # MaxPer=20, WindowSizePast_base=100, BarToCalculate=1. Goertzel: (100-1)/2=49. Bartels(5): 100/5=20. Valid.
            "WindowSizePast_base": 100, 
            "WindowSizeFuture_base": 100,
            "detrendornot": ind_settings.DEFAULT_DETREND_MODE, "DT_ZLper1": ind_settings.DEFAULT_DT_ZL_PERIOD1,
            "DT_ZLper2": ind_settings.DEFAULT_DT_ZL_PERIOD2, "DT_HPper1": ind_settings.DEFAULT_DT_HP_PERIOD1,
            "DT_HPper2": ind_settings.DEFAULT_DT_HP_PERIOD2, "DT_RegZLsmoothPer": ind_settings.DEFAULT_DT_REG_ZL_SMOOTH_PERIOD,
            "HPsmoothPer": ind_settings.DEFAULT_HP_SMOOTH_PERIOD, "ZLMAsmoothPer": ind_settings.DEFAULT_ZLMA_SMOOTH_PERIOD,
            "BarToCalculate": 1, "FilterBartels": True, # DEFAULT_FILTER_BARTELS,
            "BartNoCycles": 5, # DEFAULT_BARTELS_NUM_CYCLES, 
            "BartSmoothPer": ind_settings.DEFAULT_BARTELS_SMOOTH_PERIOD,
            "BartSigLimit": ind_settings.DEFAULT_BARTELS_SIG_LIMIT, "SortBartels": ind_settings.DEFAULT_SORT_BARTELS,
            "squaredAmp": ind_settings.DEFAULT_SQUARED_AMPLITUDE, "UseCycleStrength": ind_settings.DEFAULT_USE_CYCLE_STRENGTH,
            "useAddition": ind_settings.DEFAULT_USE_ADDITION, "useCosine": ind_settings.DEFAULT_USE_COSINE
        }
        core_output_valid = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_valid)

        if core_output_valid["status"] == "success":
            print("VALID: Core components calculated successfully.")
            print(f"  Calculated SampleSize: {core_output_valid.get('calculated_SampleSize')}")
            print(f"  current_WindowSizePast for output: {core_output_valid.get('current_WindowSizePast')}")
            test_settings_group_b = {
                "UseCycleList": ind_settings.DEFAULT_USE_CYCLE_LIST, "Cycle1": ind_settings.DEFAULT_CYCLE1,
                "Cycle2": ind_settings.DEFAULT_CYCLE2, "Cycle3": ind_settings.DEFAULT_CYCLE3,
                "Cycle4": ind_settings.DEFAULT_CYCLE4, "Cycle5": ind_settings.DEFAULT_CYCLE5,
                "StartAtCycle": ind_settings.DEFAULT_START_AT_CYCLE, "UseTopCycles": ind_settings.DEFAULT_USE_TOP_CYCLES,
                "SubtractNoise": ind_settings.DEFAULT_SUBTRACT_NOISE
            }
            wave_output = sum_composite_waves(core_output_valid, **test_settings_group_b)
            print(f"  Number of cycles: {core_output_valid['number_of_cycles']}")
            if wave_output['goertzel_past_wave'] is not None and len(wave_output['goertzel_past_wave']) > 0:
                 print(f"  Past wave (first 5): {np.round(wave_output['goertzel_past_wave'][:5], 4)}")
            if wave_output['goertzel_future_wave'] is not None and len(wave_output['goertzel_future_wave']) > 0:
                 print(f"  Future wave (first 5): {np.round(wave_output['goertzel_future_wave'][:5], 4)}")
        else:
            print(f"VALID TEST FAILED: Core component calculation failed: {core_output_valid['message']}")

        # Test Case 2: Invalid MaxPer (too high for Goertzel)
        print("\n--- Test Case 2: Invalid MaxPer (too high for Goertzel) ---")
        test_settings_invalid_mp_goertzel = test_settings_valid.copy()
        test_settings_invalid_mp_goertzel["MaxPer"] = 60 # (100-1)/2 = 49. 60 is too high.
        core_output_invalid_g = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_invalid_mp_goertzel)
        if core_output_invalid_g["status"] == "error":
            print(f"INVALID MAXPER (GOERTZEL): Correctly identified error: {core_output_invalid_g['message']}")
        else:
            print(f"INVALID MAXPER (GOERTZEL) TEST FAILED: Calculation succeeded unexpectedly.")

        # Test Case 3: Invalid MaxPer (too high for Bartels)
        print("\n--- Test Case 3: Invalid MaxPer (too high for Bartels) ---")
        test_settings_invalid_mp_bartels = test_settings_valid.copy()
        test_settings_invalid_mp_bartels["MaxPer"] = 25 # Bartels: 100/5 = 20. 25 is too high. Goertzel okay: (100-1)/2 = 49
        test_settings_invalid_mp_bartels["FilterBartels"] = True
        test_settings_invalid_mp_bartels["BartNoCycles"] = 5
        
        core_output_invalid_b = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_invalid_mp_bartels)
        if core_output_invalid_b["status"] == "error":
            print(f"INVALID MAXPER (BARTELS): Correctly identified error: {core_output_invalid_b['message']}")
        else:
            print(f"INVALID MAXPER (BARTELS) TEST FAILED: Calculation succeeded unexpectedly.")

        # Test Case 4: WindowSizePast_base too small for any reasonable MaxPer
        print("\n--- Test Case 4: WindowSizePast_base too small ---")
        test_settings_small_wsp = test_settings_valid.copy()
        test_settings_small_wsp["WindowSizePast_base"] = 10 
        test_settings_small_wsp["MaxPer"] = 5 # Try MaxPer = 5. (10-1)/2 = 4. Bartels: 10/5 = 2. MaxPer=5 invalid.
                                            # Try MaxPer = 2. (10-1)/2 = 4. Bartels: 10/5 = 2. MaxPer=2 Valid.
        test_settings_small_wsp["MaxPer"] = 2 # This should be valid
        
        core_output_small_wsp_valid_mp = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_small_wsp)
        if core_output_small_wsp_valid_mp["status"] == "success":
            print(f"SMALL WSP (Valid MP=2): Correctly succeeded. SampleSize: {core_output_small_wsp_valid_mp.get('calculated_SampleSize')}")
        else:
            print(f"SMALL WSP (Valid MP=2) TEST FAILED: {core_output_small_wsp_valid_mp['message']}")

        test_settings_small_wsp["MaxPer"] = 3 # This should be invalid due to Bartels (10/5 = 2)
        core_output_small_wsp_invalid_mp = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_small_wsp)
        if core_output_small_wsp_invalid_mp["status"] == "error":
            print(f"SMALL WSP (Invalid MP=3): Correctly identified error: {core_output_small_wsp_invalid_mp['message']}")
        else:
            print(f"SMALL WSP (Invalid MP=3) TEST FAILED: Calculation succeeded unexpectedly.")


    else:
=======
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
        # print("Error: OHLC data must contain 'open', 'high', 'low', 'close' columns.")
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
        # print(f"Warning: Unknown source_price_type '{source_price_type}', defaulting to 'Close'.")
        src = ohlc_data_df['close'].values
    
    return np.array(src) if src is not None else None


def calculate_core_cycle_components(
    ohlc_data_df: pd.DataFrame,
    # --- Group A Settings (Affect core cycle detection) ---
    source_price_type: str,
    MaxPer: int,
    WindowSizePast_base: int, # User input: now defines analysis window size AND base for past wave display length
    WindowSizeFuture_base: int, # User input: base for future wave display length
    detrendornot: str,
    DT_ZLper1: int, DT_ZLper2: int, DT_HPper1: int, DT_HPper2: int, DT_RegZLsmoothPer: int,
    HPsmoothPer: int, ZLMAsmoothPer: int,
    BarToCalculate: int, # Offset for Goertzel calculation start within the sample
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

    # --- MODIFIED LOGIC FOR SampleSize AND MaxPer VALIDATION ---
    # WindowSizePast_base (user input) now directly determines the SampleSize for analysis.
    calculated_SampleSize = WindowSizePast_base
    core_results['calculated_SampleSize'] = calculated_SampleSize # For reference/debug

    # Validate BarToCalculate against SampleSize
    if not (1 <= BarToCalculate <= calculated_SampleSize):
        core_results["status"] = "error"
        core_results["message"] = (
            f"BarToCalculate ({BarToCalculate}) is out of valid range [1, {calculated_SampleSize}] "
            f"for the chosen WindowSizePast_base ({WindowSizePast_base})."
        )
        return core_results

    # Validate MaxPer against calculated_SampleSize and other settings
    min_allowable_max_per = 2 # A practical minimum for MaxPer
    
    # Constraint 1: From Goertzel's internal processing (needs 2*MaxPer data segment starting at BarToCalculate)
    # BarToCalculate + 2 * MaxPer <= calculated_SampleSize  => MaxPer <= (calculated_SampleSize - BarToCalculate) / 2
    if (calculated_SampleSize - BarToCalculate) < (2 * min_allowable_max_per) : # Not enough room for even the smallest MaxPer
         max_per_constraint_goertzel = min_allowable_max_per -1 # Will cause MaxPer to be invalid
    else:
        max_per_constraint_goertzel = math.floor((calculated_SampleSize - BarToCalculate) / 2)

    # Constraint 2: From Bartels test data requirement (needs BartNoCycles * MaxPer data)
    max_per_constraint_bartels = float('inf')
    if FilterBartels and BartNoCycles > 0:
        if calculated_SampleSize < (BartNoCycles * min_allowable_max_per):
             max_per_constraint_bartels = min_allowable_max_per -1 # Will cause MaxPer to be invalid
        else:
            max_per_constraint_bartels = math.floor(calculated_SampleSize / BartNoCycles)
    
    effective_allowable_max_per = min(max_per_constraint_goertzel, max_per_constraint_bartels)

    if not (min_allowable_max_per <= MaxPer <= effective_allowable_max_per):
        core_results["status"] = "error"
        error_message_parts = [
            f"Max Period ({MaxPer}) is out of the valid range [{min_allowable_max_per}, {effective_allowable_max_per}]."
            f" Based on WindowSizePast_base={WindowSizePast_base} and BarToCalculate={BarToCalculate}.",
            f"  - Goertzel requires MaxPer <= {max_per_constraint_goertzel}.",
        ]
        if FilterBartels and BartNoCycles > 0:
            error_message_parts.append(f"  - Bartels Test (with {BartNoCycles} cycles) requires MaxPer <= {max_per_constraint_bartels}.")
        core_results["message"] = " ".join(error_message_parts)
        return core_results
    # --- END OF MODIFIED LOGIC ---

    # Adjust window sizes for OUTPUT WAVES (goeWorkPast/Future matrices dimensions)
    # These ensure output waves can represent at least one full cycle of MaxPer.
    current_WindowSizePast_output = max(WindowSizePast_base, 2 * MaxPer)
    current_WindowSizeFuture_output = max(WindowSizeFuture_base, 2 * MaxPer)
    core_results['current_WindowSizePast'] = current_WindowSizePast_output
    core_results['current_WindowSizeFuture'] = current_WindowSizeFuture_output
    
    calculation_bar_idx = len(ohlc_data_df) - 1 
    core_results['calculation_bar_idx'] = calculation_bar_idx

    if len(src_series) < calculated_SampleSize:
        # This check might be redundant if ohlc_data length is checked before calling,
        # but good for safety. WindowSizePast_base UI should also be limited by data length.
        core_results["status"] = "error"
        core_results["message"] = (
            f"Not enough data in source ({len(src_series)} bars) "
            f"for the chosen WindowSizePast_base ({calculated_SampleSize} bars)."
        )
        return core_results

    # Initialize buffers and matrices
    # Dimensions depend on MaxPer for buffers, and current_WindowSize_output for work matrices
    goeWorkPast = np.zeros((current_WindowSizePast_output, MaxPer + 1))
    goeWorkFuture = np.zeros((current_WindowSizeFuture_output, MaxPer + 1))
    cyclebuffer = np.zeros(MaxPer + 1)
    amplitudebuffer = np.zeros(MaxPer + 1)
    phasebuffer = np.zeros(MaxPer + 1)
    # MaxPer * 3 for cycleBartelsBuffer was from original, check if still appropriate
    # BartelsProb array in processing.bartels_cycle_test is 0-indexed by G (number_of_cycles found, up to MaxPer).
    # Original Goertzel filled cycleBartelsBuffer based on G.
    # Let's assume MaxPer*3 is generous enough. If number_of_cycles can be up to MaxPer,
    # then cycleBartelsBuffer needs to be at least MaxPer. MaxPer * 3 seems safe.
    cycleBartelsBuffer = np.zeros(MaxPer * 3) 

    # Prepare srcVal for analysis (length = calculated_SampleSize)
    srcVal_from_file = src_series[-calculated_SampleSize:]
    srcVal = srcVal_from_file[::-1] # Reverse to most_recent_first
    
    processed_srcVal = np.copy(srcVal)
    # Detrending/Smoothing applied to the window of size calculated_SampleSize
    if detrendornot == ind_settings.HPSMTHDT: processed_srcVal = detrend_centered_ma(srcVal, DT_HPper1, DT_HPper2, calculated_SampleSize, detrendornot)
    elif detrendornot == ind_settings.ZLAGSMTHDT: processed_srcVal = detrend_centered_ma(srcVal, DT_ZLper1, DT_ZLper2, calculated_SampleSize, detrendornot)
    elif detrendornot == ind_settings.LOG_ZLAG_REGRESSION_DT: processed_srcVal = detrend_ln_zero_lag_regression(srcVal, DT_RegZLsmoothPer, calculated_SampleSize)
    elif detrendornot == ind_settings.HPSMTH:
        if HPsmoothPer <= 0: # Prevent math error
            print(f"Warning: HPsmoothPer ({HPsmoothPer}) must be positive. Skipping HP smoothing.")
        else:
            sin_pi_per = math.sin(math.pi / HPsmoothPer)
            lamb = 0.0625 / math.pow(sin_pi_per, 4) if sin_pi_per != 0 else float('inf')
            if lamb != float('inf') and lamb >=0: processed_srcVal = hpf_core(srcVal, lamb, calculated_SampleSize)
            else: print(f"Warning: Cannot apply HP smoothing. Lambda invalid ({lamb}). Using original srcVal.")
    elif detrendornot == ind_settings.ZLAGSMTH:
        if ZLMAsmoothPer <=0:
            print(f"Warning: ZLMAsmoothPer ({ZLMAsmoothPer}) must be positive. Skipping ZLMA smoothing.")
        else:
            processed_srcVal = zlma_core(srcVal, ZLMAsmoothPer, calculated_SampleSize)
    # else: NONE_SMTH_DT or unknown, processed_srcVal remains as srcVal copy

    core_results['final_srcVal_for_debug'] = processed_srcVal 

    if processed_srcVal is None or len(processed_srcVal) != calculated_SampleSize:
        core_results["status"] = "error"
        core_results["message"] = "srcVal processing failed or returned incorrect length."
        return core_results

    # Call Goertzel algorithm
    # The 'samplesize' argument for goertzel_core is our calculated_SampleSize.
    # WindowSizePast/Future for goertzel_core are for the dimensions of goeWorkPast/Future matrices.
    number_of_cycles = goertzel_core(
        processed_srcVal, BarToCalculate, calculated_SampleSize, MaxPer,
        squaredAmp, useAddition, useCosine, UseCycleStrength,
        current_WindowSizePast_output, current_WindowSizeFuture_output, # For goeWork matrices
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
    goeWorkPast = core_components['goeWorkPast']       # Matrix of individual past cycle waves
    goeWorkFuture = core_components['goeWorkFuture']     # Matrix of individual future cycle waves
    # These are the actual dimensions of goeWorkPast/Future and the length of summed waves
    current_WindowSizePast_output = core_components['current_WindowSizePast'] 
    current_WindowSizeFuture_output = core_components['current_WindowSizeFuture']

    goertzel_past_wave = np.zeros(current_WindowSizePast_output)
    goertzel_future_wave = np.zeros(current_WindowSizeFuture_output)

    # Calculate past composite wave
    # Summing columns of goeWorkPast matrix
    for i_time_step in range(current_WindowSizePast_output):
        sum_val = 0.0
        if UseCycleList:
            # Use specific cycles by their rank (1-indexed in UI, adjust for 0-indexed matrix column if needed by goeWork matrix structure)
            # Assuming goeWorkPast columns are 1-indexed for cycle rank (Cycle1=1st col, etc.) based on original Pine
            cycles_to_use = [c for c in [Cycle1, Cycle2, Cycle3, Cycle4, Cycle5] if c > 0 and c <= number_of_cycles and c < goeWorkPast.shape[1]]
            for cycle_rank_col_idx in cycles_to_use: # cycle_rank_col_idx is 1-based
                 if i_time_step < goeWorkPast.shape[0] : sum_val += goeWorkPast[i_time_step, cycle_rank_col_idx]
        else:
            # Use top N cycles starting from StartAtCycle rank
            # Assuming StartAtCycle is 1-indexed rank
            actual_start_rank = StartAtCycle # 1-indexed
            # Max rank to sum is StartAtCycle + UseTopCycles - 1
            # Loop up to min(actual_start_rank + UseTopCycles, number_of_cycles + 1)
            end_cycle_loop = min(actual_start_rank + UseTopCycles, number_of_cycles + 1) 
            for k_rank in range(actual_start_rank, end_cycle_loop): # k_rank is 1-indexed
                 if i_time_step < goeWorkPast.shape[0] and k_rank < goeWorkPast.shape[1]: sum_val += goeWorkPast[i_time_step, k_rank]
            
            if SubtractNoise and number_of_cycles >= 1:
                # Noise cycles are those after the 'UseTopCycles'
                start_noise_rank = actual_start_rank + UseTopCycles # 1-indexed
                for k_rank_noise in range(start_noise_rank, number_of_cycles + 1): # k_rank_noise is 1-indexed
                     if i_time_step < goeWorkPast.shape[0] and k_rank_noise < goeWorkPast.shape[1]: sum_val -= goeWorkPast[i_time_step, k_rank_noise]
        goertzel_past_wave[i_time_step] = sum_val

    # Calculate future composite wave (from time-reversed goeWorkFuture)
    # Summing columns of goeWorkFuture matrix
    for i_time_step in range(current_WindowSizeFuture_output):
        sum_val = 0.0
        if UseCycleList:
            cycles_to_use = [c for c in [Cycle1, Cycle2, Cycle3, Cycle4, Cycle5] if c > 0 and c <= number_of_cycles and c < goeWorkFuture.shape[1]]
            for cycle_rank_col_idx in cycles_to_use:
                 if i_time_step < goeWorkFuture.shape[0]: sum_val += goeWorkFuture[i_time_step, cycle_rank_col_idx]
        else:
            actual_start_rank = StartAtCycle
            end_cycle_loop = min(actual_start_rank + UseTopCycles, number_of_cycles + 1)
            for k_rank in range(actual_start_rank, end_cycle_loop):
                 if i_time_step < goeWorkFuture.shape[0] and k_rank < goeWorkFuture.shape[1]: sum_val += goeWorkFuture[i_time_step, k_rank]

            if SubtractNoise and number_of_cycles >= 1:
                start_noise_rank = actual_start_rank + UseTopCycles
                for k_rank_noise in range(start_noise_rank, number_of_cycles + 1):
                     if i_time_step < goeWorkFuture.shape[0] and k_rank_noise < goeWorkFuture.shape[1]: sum_val -= goeWorkFuture[i_time_step, k_rank_noise]
        goertzel_future_wave[i_time_step] = sum_val
    
    # goeWorkFuture was stored time-reversed by Goertzel function to match Pine logic.
    # So, the sum is also time-reversed. Reverse it back for chronological future wave.
    goertzel_future_wave = goertzel_future_wave[::-1] 

    wave_results['goertzel_past_wave'] = goertzel_past_wave
    wave_results['goertzel_future_wave'] = goertzel_future_wave
    return wave_results

# Kept for direct testing or if a single-call interface is ever needed outside Streamlit
def run_indicator_calculations(ohlc_data_df: pd.DataFrame, **all_settings):
    """
    Original single-call function, now orchestrates the two new functions.
    Not typically called directly by Streamlit app if using two-stage caching.
    """
    # Note: The keys for WindowSizePast/Future in all_settings might be the original ones.
    # The calculate_core_cycle_components expects 'WindowSizePast_base' and 'WindowSizeFuture_base'.
    # This mapping should align with how settings are stored/passed if this function is used.
    # Assuming all_settings uses the '_base' suffix for consistency with streamlit_app.py state keys.
    core_args = {
        "source_price_type": all_settings["source_price_type"], "MaxPer": all_settings["MaxPer"],
        "WindowSizePast_base": all_settings.get("WindowSizePast_base", all_settings.get("WindowSizePast")), # Adapt if key name varies
        "WindowSizeFuture_base": all_settings.get("WindowSizeFuture_base", all_settings.get("WindowSizeFuture")), # Adapt
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
    print("--- Testing main_calculator.py (with two-stage calculation and new SampleSize logic) ---")
    from .data_loader import load_ohlc_from_csv 
    import os as test_os 

    # Assuming btc_daily.csv is in the parent directory of indicator_logic
    # Adjust path if your project structure is different
    current_dir = test_os.path.dirname(test_os.path.abspath(__file__))
    data_file_path_test = test_os.path.join(current_dir, '..', 'btc_daily.csv') 
    
    print(f"Attempting to load data from: {data_file_path_test}")
    actual_ohlc_data = load_ohlc_from_csv(data_file_path_test)

    if actual_ohlc_data is not None and not actual_ohlc_data.empty:
        print(f"Loaded {len(actual_ohlc_data)} rows for testing.")

        # Test Case 1: Valid settings based on new logic
        print("\n--- Test Case 1: Valid Settings ---")
        test_settings_valid = {
            "source_price_type": "Close", 
            "MaxPer": 20, # MaxPer=20, WindowSizePast_base=100, BarToCalculate=1. Goertzel: (100-1)/2=49. Bartels(5): 100/5=20. Valid.
            "WindowSizePast_base": 100, 
            "WindowSizeFuture_base": 100,
            "detrendornot": ind_settings.DEFAULT_DETREND_MODE, "DT_ZLper1": ind_settings.DEFAULT_DT_ZL_PERIOD1,
            "DT_ZLper2": ind_settings.DEFAULT_DT_ZL_PERIOD2, "DT_HPper1": ind_settings.DEFAULT_DT_HP_PERIOD1,
            "DT_HPper2": ind_settings.DEFAULT_DT_HP_PERIOD2, "DT_RegZLsmoothPer": ind_settings.DEFAULT_DT_REG_ZL_SMOOTH_PERIOD,
            "HPsmoothPer": ind_settings.DEFAULT_HP_SMOOTH_PERIOD, "ZLMAsmoothPer": ind_settings.DEFAULT_ZLMA_SMOOTH_PERIOD,
            "BarToCalculate": 1, "FilterBartels": True, # DEFAULT_FILTER_BARTELS,
            "BartNoCycles": 5, # DEFAULT_BARTELS_NUM_CYCLES, 
            "BartSmoothPer": ind_settings.DEFAULT_BARTELS_SMOOTH_PERIOD,
            "BartSigLimit": ind_settings.DEFAULT_BARTELS_SIG_LIMIT, "SortBartels": ind_settings.DEFAULT_SORT_BARTELS,
            "squaredAmp": ind_settings.DEFAULT_SQUARED_AMPLITUDE, "UseCycleStrength": ind_settings.DEFAULT_USE_CYCLE_STRENGTH,
            "useAddition": ind_settings.DEFAULT_USE_ADDITION, "useCosine": ind_settings.DEFAULT_USE_COSINE
        }
        core_output_valid = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_valid)

        if core_output_valid["status"] == "success":
            print("VALID: Core components calculated successfully.")
            print(f"  Calculated SampleSize: {core_output_valid.get('calculated_SampleSize')}")
            print(f"  current_WindowSizePast for output: {core_output_valid.get('current_WindowSizePast')}")
            test_settings_group_b = {
                "UseCycleList": ind_settings.DEFAULT_USE_CYCLE_LIST, "Cycle1": ind_settings.DEFAULT_CYCLE1,
                "Cycle2": ind_settings.DEFAULT_CYCLE2, "Cycle3": ind_settings.DEFAULT_CYCLE3,
                "Cycle4": ind_settings.DEFAULT_CYCLE4, "Cycle5": ind_settings.DEFAULT_CYCLE5,
                "StartAtCycle": ind_settings.DEFAULT_START_AT_CYCLE, "UseTopCycles": ind_settings.DEFAULT_USE_TOP_CYCLES,
                "SubtractNoise": ind_settings.DEFAULT_SUBTRACT_NOISE
            }
            wave_output = sum_composite_waves(core_output_valid, **test_settings_group_b)
            print(f"  Number of cycles: {core_output_valid['number_of_cycles']}")
            if wave_output['goertzel_past_wave'] is not None and len(wave_output['goertzel_past_wave']) > 0:
                 print(f"  Past wave (first 5): {np.round(wave_output['goertzel_past_wave'][:5], 4)}")
            if wave_output['goertzel_future_wave'] is not None and len(wave_output['goertzel_future_wave']) > 0:
                 print(f"  Future wave (first 5): {np.round(wave_output['goertzel_future_wave'][:5], 4)}")
        else:
            print(f"VALID TEST FAILED: Core component calculation failed: {core_output_valid['message']}")

        # Test Case 2: Invalid MaxPer (too high for Goertzel)
        print("\n--- Test Case 2: Invalid MaxPer (too high for Goertzel) ---")
        test_settings_invalid_mp_goertzel = test_settings_valid.copy()
        test_settings_invalid_mp_goertzel["MaxPer"] = 60 # (100-1)/2 = 49. 60 is too high.
        core_output_invalid_g = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_invalid_mp_goertzel)
        if core_output_invalid_g["status"] == "error":
            print(f"INVALID MAXPER (GOERTZEL): Correctly identified error: {core_output_invalid_g['message']}")
        else:
            print(f"INVALID MAXPER (GOERTZEL) TEST FAILED: Calculation succeeded unexpectedly.")

        # Test Case 3: Invalid MaxPer (too high for Bartels)
        print("\n--- Test Case 3: Invalid MaxPer (too high for Bartels) ---")
        test_settings_invalid_mp_bartels = test_settings_valid.copy()
        test_settings_invalid_mp_bartels["MaxPer"] = 25 # Bartels: 100/5 = 20. 25 is too high. Goertzel okay: (100-1)/2 = 49
        test_settings_invalid_mp_bartels["FilterBartels"] = True
        test_settings_invalid_mp_bartels["BartNoCycles"] = 5
        
        core_output_invalid_b = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_invalid_mp_bartels)
        if core_output_invalid_b["status"] == "error":
            print(f"INVALID MAXPER (BARTELS): Correctly identified error: {core_output_invalid_b['message']}")
        else:
            print(f"INVALID MAXPER (BARTELS) TEST FAILED: Calculation succeeded unexpectedly.")

        # Test Case 4: WindowSizePast_base too small for any reasonable MaxPer
        print("\n--- Test Case 4: WindowSizePast_base too small ---")
        test_settings_small_wsp = test_settings_valid.copy()
        test_settings_small_wsp["WindowSizePast_base"] = 10 
        test_settings_small_wsp["MaxPer"] = 5 # Try MaxPer = 5. (10-1)/2 = 4. Bartels: 10/5 = 2. MaxPer=5 invalid.
                                            # Try MaxPer = 2. (10-1)/2 = 4. Bartels: 10/5 = 2. MaxPer=2 Valid.
        test_settings_small_wsp["MaxPer"] = 2 # This should be valid
        
        core_output_small_wsp_valid_mp = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_small_wsp)
        if core_output_small_wsp_valid_mp["status"] == "success":
            print(f"SMALL WSP (Valid MP=2): Correctly succeeded. SampleSize: {core_output_small_wsp_valid_mp.get('calculated_SampleSize')}")
        else:
            print(f"SMALL WSP (Valid MP=2) TEST FAILED: {core_output_small_wsp_valid_mp['message']}")

        test_settings_small_wsp["MaxPer"] = 3 # This should be invalid due to Bartels (10/5 = 2)
        core_output_small_wsp_invalid_mp = calculate_core_cycle_components(actual_ohlc_data.copy(), **test_settings_small_wsp)
        if core_output_small_wsp_invalid_mp["status"] == "error":
            print(f"SMALL WSP (Invalid MP=3): Correctly identified error: {core_output_small_wsp_invalid_mp['message']}")
        else:
            print(f"SMALL WSP (Invalid MP=3) TEST FAILED: Calculation succeeded unexpectedly.")


    else:
>>>>>>> 07e747c20443eb8cbfdc5b93205384046f029227
        print(f"Failed to load data for testing main_calculator.py from {data_file_path_test}")