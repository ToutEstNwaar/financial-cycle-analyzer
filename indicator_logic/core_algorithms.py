<<<<<<< HEAD
# indicator_logic/core_algorithms.py

"""
Core algorithms for cycle analysis, including Zero-lag MA, Bartels Probability,
Hodrick-Prescott Filter, and Goertzel algorithm.
Logic is intended to be a direct refactor from indicator_milestone_1.py.
"""

import pandas as pd
import numpy as np
import math

# --- Zero-lag Moving Average (Replicating Flawed PineScript Logic) ---
def zero_lag_ma(src, smooth_per, bars_taken): # Matches original signature
    """
    Calculates the Zero-lag Moving Average (ZLMA) by replicating the
    *original flawed logic* found in the provided PineScript code.
    WARNING: This function intentionally replicates a bug where the sum
    and weight accumulators are NOT reset correctly within each pass.
    It will NOT produce a mathematically correct ZLMA. It is intended
    only for matching the output of the specific flawed PineScript version.
    Args:
        src (np.ndarray or pd.Series): The source data array. Assumed to be
                                     ordered with the most recent value at index 0.
        smooth_per (int): The smoothing period.
        bars_taken (int): The number of bars (elements) in the source array.
    Returns:
        np.ndarray: An array containing the ZLMA values.
    """
    # --- Start ZLMA Debug ---
    # print(f"\n--- DEBUG ZLMA (Flawed Replication) ---")
    # print(f"Entering with smooth_per={smooth_per}, bars_taken={bars_taken}, src length={len(src)}")
    # ---

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series):
            src = src.values
        else:
            try:
                src = np.array(src, dtype=float)
            except Exception as e:
                print(f"Error converting src to numpy array in zero_lag_ma: {e}")
                return np.array([])

    if bars_taken > len(src):
        # print(f"Warning: bars_taken ({bars_taken}) > src length ({len(src)}). Adjusting.")
        bars_taken = len(src)
    if bars_taken <= 0 or smooth_per <= 0:
        # print(f"Error: bars_taken ({bars_taken}) and smooth_per ({smooth_per}) must be positive.")
        return np.array([])

    src_subset = src[:bars_taken]
    # --- ZLMA Debug ---
    # print(f"Using src_subset of length: {len(src_subset)}")
    # ---

    lwma1 = np.zeros(bars_taken)
    output = np.zeros(bars_taken)

    # --- First LWMA pass (Replicating PineScript Flaw) ---
    # print("Starting Pass 1 (Flawed Logic)...")
    sum_val = 0.0
    sum_w = 0.0
    for i in range(bars_taken - 1, -1, -1):
        # inner_sum_added = 0.0 # From original debug
        # inner_w_added = 0.0 # From original debug
        for k in range(smooth_per):
            index = i + k
            if index >= 0 and index < bars_taken - 1: # Replicate exact condition
                 if index < len(src_subset):
                     src_value = src_subset[index]
                     if not np.isnan(src_value):
                        weight = float(smooth_per - k)
                        sum_w += weight
                        sum_val += weight * src_value
                        # inner_sum_added += weight * src_value # From original debug
                        # inner_w_added += weight # From original debug
        if sum_w != 0.0:
            lwma1[i] = sum_val / sum_w
        else:
            lwma1[i] = 0.0

    # --- ZLMA Debug ---
    # print(f"DEBUG ZLMA (Flawed Replication): lwma1 calculated (shape: {lwma1.shape})")
    # print(f"DEBUG ZLMA (Flawed Replication): lwma1 (first 10): {np.round(lwma1[:10], 8)}")
    # print(f"DEBUG ZLMA (Flawed Replication): lwma1 (last 10): {np.round(lwma1[-10:], 8)}")
    # ---

    # --- Second LWMA pass (Replicating PineScript Flaw) ---
    # print("Starting Pass 2 (Flawed Logic)...")
    sum_val = 0.0
    sum_w = 0.0
    for i in range(bars_taken):
        # inner_sum_added = 0.0 # From original debug
        # inner_w_added = 0.0 # From original debug
        for k in range(smooth_per):
            index = i - k
            if index >= 0: # PineScript only checked this
                 if index < len(lwma1):
                     lwma1_value = lwma1[index]
                     if not np.isnan(lwma1_value):
                         weight = float(smooth_per - k)
                         sum_w += weight
                         sum_val += weight * lwma1_value
                         # inner_sum_added += weight * lwma1_value # From original debug
                         # inner_w_added += weight # From original debug
        if sum_w != 0.0:
            output[i] = sum_val / sum_w
        else:
            output[i] = 0.0

    # --- ZLMA Debug ---
    # print(f"DEBUG ZLMA (Flawed Replication): output calculated (shape: {output.shape})")
    # print(f"DEBUG ZLMA (Flawed Replication): output (first 10): {np.round(output[:10], 8)}")
    # print(f"DEBUG ZLMA (Flawed Replication): output (last 10): {np.round(output[-10:], 8)}")
    # print(f"DEBUG ZLMA (Flawed Replication): Exiting")
    # print(f"--- END DEBUG ZLMA (Flawed Replication) ---\n")
    return output

# --- Bartels probability ---
def bartels_prob(n, N, Bvalues): # Matches original signature
    """
    Calculates the Bartels probability.
    Identical to indicator_milestone_1.py.
    """
    # --- Start BartelsProb Debug ---
    # print(f"\n--- DEBUG BARTELS_PROB ---")
    # print(f"Entering with n={n}, N={N}")
    # ---

    if not isinstance(Bvalues, np.ndarray):
        try:
            Bvalues = np.array(Bvalues, dtype=float)
        except Exception as e:
            # print(f"Error converting Bvalues to numpy array: {e}")
            return 0.0

    # --- BartelsProb Debug ---
    # print(f"Bvalues shape: {Bvalues.shape}")
    # print(f"Bvalues (first 10): {np.round(Bvalues[:10], 8)}")
    # ---
    if N <= 0 or n <= 0: # Added basic check, original relied on shape
        return 0.0
    if Bvalues.shape[0] != N * n:
        # print(f"Error: Bvalues array shape ({Bvalues.shape[0]}) does not match expected shape ({N * n}).")
        return 0.0

    teta = np.zeros(n)
    vsin = np.zeros(n)
    vcos = np.zeros(n)
    CoeffA = np.zeros(N)
    CoeffB = np.zeros(N)
    IndAmplit = np.zeros(N)

    for i in range(n):
        teta[i] = 1.0 * (i + 1) / n * 2 * math.pi
        vsin[i] = math.sin(teta[i])
        vcos[i] = math.cos(teta[i])
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: teta (first 5): {np.round(teta[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: vsin (first 5): {np.round(vsin[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: vcos (first 5): {np.round(vcos[:5], 8)}")
    # ---

    for t_loop in range(N): # Renamed t to t_loop
        for i in range(n):
            # Ensure index is within bounds for Bvalues (already checked by Bvalues.shape[0] != N * n)
            CoeffA[t_loop] += vsin[i] * Bvalues[t_loop * n + i]
            CoeffB[t_loop] += vcos[i] * Bvalues[t_loop * n + i]
        IndAmplit[t_loop] = math.pow(CoeffA[t_loop], 2) + math.pow(CoeffB[t_loop], 2)
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: CoeffA (first 5): {np.round(CoeffA[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: CoeffB (first 5): {np.round(CoeffB[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: IndAmplit (first 5): {np.round(IndAmplit[:5], 8)}")
    # ---

    AvgCoeffA = np.sum(CoeffA) / N
    AvgCoeffB = np.sum(CoeffB) / N
    AvgIndAmplit = np.sum(IndAmplit) / N

    AvgAmpl = math.sqrt(math.pow(AvgCoeffA, 2) + math.pow(AvgCoeffB, 2))
    AvgIndAmplit_sqrt = math.sqrt(AvgIndAmplit) # Pine script takes sqrt after averaging
    ExptAmpl = AvgIndAmplit_sqrt / math.sqrt(1.0 * N)
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: AvgCoeffA={AvgCoeffA:.6f}, AvgCoeffB={AvgCoeffB:.6f}, AvgIndAmplit={AvgIndAmplit:.6f}")
    # print(f"DEBUG BARTELS_PROB: AvgAmpl={AvgAmpl:.6f}, AvgIndAmplit_sqrt={AvgIndAmplit_sqrt:.6f}, ExptAmpl={ExptAmpl:.6f}")
    # ---

    if ExptAmpl == 0:
        ARatio = 0.0
    else:
        ARatio = AvgAmpl / ExptAmpl
    
    BP = 0.0
    try:
        BP = 1 / math.exp(math.pow(ARatio, 2))
    except OverflowError: # If ARatio is too large, exp overflows
        BP = 0.0
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: ARatio={ARatio:.6f}, Final BP={BP:.8f}")
    # print(f"DEBUG BARTELS_PROB: Exiting")
    # print(f"--- END DEBUG BARTELS_PROB ---\n")
    return BP

# --- Hodrick-Prescott Filter ---
def hodrick_prescott_filter(src, lamb, per): # Matches original signature
    """
    Implements the Hodrick-Prescott filter.
    Identical to indicator_milestone_1.py.
    """
    # --- Start HPF Debug ---
    # print(f"\n--- DEBUG HPF ---")
    # print(f"Entering with lamb={lamb:.6f}, per={per}, src length={len(src)}")
    # ---

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series):
            src = src.values
        else:
            try:
                src = np.array(src, dtype=float)
            except Exception as e:
                # print(f"Error converting src to numpy array: {e}")
                return np.array([])

    if per > len(src):
        # print(f"Warning: per ({per}) is greater than source length ({len(src)}). Adjusting to source length.")
        per = len(src)
    if per <= 0:
        # print(f"Error: per ({per}) must be positive.")
        return np.array([])

    src_subset = src[-per:]

    H1, H2, H3, H4, H5 = 0., 0., 0., 0., 0.
    HH1, HH2, HH3, HH5 = 0., 0., 0., 0.
    # HB, HC, Z were local loop vars in original

    a = np.zeros(per)
    b = np.zeros(per)
    c_arr = np.zeros(per) # Renamed from 'c'
    output = np.copy(src_subset)
    # --- HPF Debug ---
    # print(f"DEBUG HPF: Initialized arrays a, b, c, output with shape ({per},)")
    # print(f"DEBUG HPF: Initial output (first 10): {np.round(output[:10], 6)}")
    # print(f"DEBUG HPF: Initial src_subset (first 10): {np.round(src_subset[:10], 6)}")
    # ---

    # Coefficient initialization IDENTICAL to indicator_milestone_1.py
    if per > 0:
        a[0] = 1.0 + lamb
        b[0] = -2.0 * lamb
        c_arr[0] = lamb

    for i in range(1, per - 3): # Original loop was 1 to per-3 (exclusive end)
        a[i] = 6.0 * lamb + 1.0
        b[i] = -4.0 * lamb
        c_arr[i] = lamb
    
    # Boundary conditions IDENTICAL to indicator_milestone_1.py
    if per >= 2:
        a[1] = 5.0 * lamb + 1.0
    if per >= 1:
        a[per - 1] = 1.0 + lamb
    if per >= 2:
        a[per - 2] = 5.0 * lamb + 1.0
        b[per - 2] = -2.0 * lamb
    # These assignments will overwrite previous ones if per is small
    if per >= 1: # Ensure per-1 is a valid index
      b[per - 1] = 0.0
    if per >= 2: # Ensure per-2 is valid
      c_arr[per - 2] = 0.0
    if per >= 1:
      c_arr[per - 1] = 0.0
      
    # --- HPF Debug ---
    # print(f"DEBUG HPF: Initialized a (first 10): {np.round(a[:10], 6)}")
    # print(f"DEBUG HPF: Initialized b (first 10): {np.round(b[:10], 6)}")
    # print(f"DEBUG HPF: Initialized c (first 10): {np.round(c_arr[:10], 6)}")
    # print("DEBUG HPF: Starting forward pass...")
    # ---

    # Forward pass
    for i in range(per):
        # a_i, b_i, c_i from original debug, not needed for logic
        Z_val = a[i] - H4 * H1 - HH5 * HH2 # Use a[i] as per original
        if Z_val == 0:
            # print(f"  DEBUG HPF (Forward i={i}): Z is ZERO! Aborting.")
            # print(f"Error: Division by zero in Hodrick-Prescott forward pass at index {i}.")
            return np.zeros(per)

        HB_val = b[i] # Use b[i] as per original
        HH1 = H1
        H1 = (HB_val - H4 * H2) / Z_val
        b[i] = H1 # Store modified b

        HC_val = c_arr[i] # Use c_arr[i] as per original
        HH2 = H2
        H2 = HC_val / Z_val
        c_arr[i] = H2 # Store modified c_arr

        src_val_iter = src_subset[i] # src_val_iter instead of src_val
        # Original: a_val = (src_val - HH3_val * HH5_val - H3_val * H4_val) / Z
        # HH3_val, HH5_val etc. were from previous iteration's H3, H5.
        # So, current HH3, HH5, H3, H4 are used here.
        a_val_modified = (src_val_iter - HH3 * HH5 - H3 * H4) / Z_val
        a[i] = a_val_modified # Store modified a

        HH3 = H3
        H3 = a[i] 
        H4 = HB_val - H5 * HH1 # H4 uses b[i] before modification in this iteration
        HH5 = H5
        H5 = HC_val # H5 uses c_arr[i] before modification in this iteration
    
    # Backward pass (original script H1, H2 were reset before this loop)
    H2_bp = 0.0 # Using _bp suffix for clarity in backward pass
    H1_bp = 0.0
    if per > 0:
        H1_bp = a[per-1] # a has been modified by forward pass
        output[per-1] = H1_bp

    # --- HPF Debug ---
    # print("DEBUG HPF: Starting backward pass...")
    # if per > 0:
        # print(f"  DEBUG HPF (Backward Start): Initial output[{per-1}]={output[per-1]:.4f}, H1={H1_bp:.4f}, H2={H2_bp:.4f}")
    # ---

    for i in range(per - 2, -1, -1):
        # output_i = a[i] - b[i] * H1_bp - c_arr[i] * H2_bp # Uses modified a,b,c_arr
        output[i] = a[i] - b[i] * H1_bp - c_arr[i] * H2_bp
        H2_bp = H1_bp
        H1_bp = output[i]

    # --- HPF Debug ---
    # print("DEBUG HPF: Backward pass complete.")
    # print(f"DEBUG HPF: Final output (first 10): {np.round(output[:10], 6)}")
    # print(f"DEBUG HPF: Final output (last 10): {np.round(output[-10:], 6)}")
    # print(f"--- END DEBUG HPF ---\n")
    return output

# --- Goertzel Browser ---
# Matches original signature, but bartels_cycle_test is now dependency injected
def goertzel(src, forBar, samplesize, per, squaredAmp, useAddition, useCosine, UseCycleStrength, WindowSizePast,
             WindowSizeFuture, FilterBartels, BartNoCycles, BartSmoothPer, BartSigLimit,
             SortBartels, goeWorkPast, goeWorkFuture, cyclebuffer,
             amplitudebuffer, phasebuffer, cycleBartelsBuffer,
             bartels_cycle_test_func # Dependency injection
             ):
    """
    Implements the Goertzel algorithm for spectral analysis.
    Identical to indicator_milestone_1.py, with bartels_cycle_test injected.
    """
    # DEBUG: Goertzel entry point
    print(f"DEBUG GOERTZEL: Entry with:")
    print(f"  forBar={forBar}, samplesize={samplesize}, per={per}")
    print(f"  src length={len(src)}")
    print(f"  squaredAmp={squaredAmp}, UseCycleStrength={UseCycleStrength}")
    print(f"  FilterBartels={FilterBartels}")
    
    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series): src = src.values
        else:
            try: src = np.array(src, dtype=float)
            except Exception as e: 
                print(f"DEBUG GOERTZEL: Error converting src to numpy array: {e}")
                return 0
    
    # Original script sample = 2 * per (MaxPer)
    sample_calc_len = 2 * per # Renamed 'sample' to 'sample_calc_len' for clarity
    if sample_calc_len <= 0: 
        print(f"DEBUG GOERTZEL: Invalid sample_calc_len={sample_calc_len}, returning 0")
        return 0 # MaxPer must be >0

    print(f"DEBUG GOERTZEL: sample_calc_len (2*per) = {sample_calc_len}")

    goeWork1 = np.zeros(sample_calc_len + 1)
    goeWork2 = np.zeros(sample_calc_len + 1)
    goeWork3 = np.zeros(sample_calc_len + 1) # Peaks. Index is period.
    goeWork4 = np.zeros(sample_calc_len + 1) # Detrended src. Index is time within 2*MaxPer window.

    # Ensure indices are within bounds for src.
    # forBar is offset. sample_calc_len is window size for Goertzel transform.
    if forBar + sample_calc_len - 1 >= len(src) or forBar < 0:
         print(f"DEBUG GOERTZEL: Bounds error - forBar ({forBar}) or sample size ({sample_calc_len}) out of bounds for src length ({len(src)}).")
         return 0

    print(f"DEBUG GOERTZEL: Bounds check passed. forBar={forBar}, sample_calc_len={sample_calc_len}, src length={len(src)}")
    print(f"DEBUG GOERTZEL: Will analyze src[{forBar}:{forBar + sample_calc_len}]")

    # print("Calculating initial detrending (goeWork4)...")
    # Linear detrend over the sample_calc_len window of src
    # src is [recent, ..., oldest]
    # temp1 is oldest value in window: src[forBar + sample_calc_len - 1]
    # temp2 is slope: (src[forBar] - temp1) / (sample_calc_len - 1)
    if sample_calc_len > 1: # Original script did not check for sample_calc_len=1 here for temp2
        temp1_val = src[forBar + sample_calc_len - 1] # Oldest value in window
        temp2_val = (src[forBar] - temp1_val) / (sample_calc_len - 1) # Slope
    elif sample_calc_len == 1: # Handle if only 1 point in window
        temp1_val = src[forBar]
        temp2_val = 0 # No slope for a single point
    else: # sample_calc_len <= 0, already returned
        return 0


    # Pine: for k = sample - 1 down to 0. Python: sample_calc_len-1 down to 0
    # Original: for k in range(sample - 1, 0, -1): -> k from sample-1 down to 1
    # This populates goeWork4[k] where k is 1-indexed time in Pine.
    for k_gw4 in range(sample_calc_len - 1, -1, -1): # k_gw4 from sample_calc_len-1 down to 0
        if forBar + k_gw4 < len(src) : # Check src index
            # Original: temp = src[forBar + k - 1] - (temp1 + temp2 * (sample - k))
            # k_gw4 is distance from "end" of window (0 is oldest, sample_calc_len-1 is newest in this loop)
            # Pine k was sample-1 down to 1. Let's use Pine's k for easier mapping
            # For k_pine = sample_calc_len-1 down to 1 (as in original `goeWork4[k] = temp`)
            # Equivalent python k_gw4 here would be k_pine-1
            # Let's replicate original directly:
            # Loop k from sample_calc_len-1 down to 1 for goeWork4[k_pine]
            if k_gw4 > 0 : # Only for k_pine from sample-1 down to 1
                src_idx = forBar + k_gw4 -1 # src[forBar + (k_pine-style)-1]
                # trend = temp1_val(oldest) + slope * (distance from oldest point)
                # distance from oldest point for src[forBar + k_pine-1] is ( (forBar+k_pine-1) - (forBar+sample_calc_len-1) ) in original time
                # Pine: temp1 + temp2 * (sample - k_pine)
                # (sample_calc_len - k_gw4) term for Pine's k_pine
                trend_val = temp1_val + temp2_val * (sample_calc_len - k_gw4)
                goeWork4[k_gw4] = src[src_idx] - trend_val
            # goeWork3[k_gw4] = 0. (already initialized)
        # else: out of bounds, goeWork4[k_gw4] remains 0.

    # goeWork3[0] = 0 (already initialized)

    # Main Goertzel calculation loop (k_period from 2 to MaxPer)
    for k_period in range(2, per + 1):
        w, x, y = 0.0, 0.0, 0.0
        z_freq = 1.0 / k_period
        cos_term = 2.0 * math.cos(2.0 * math.pi * z_freq)

        # Pine loop: for i = sample down to 0 (inclusive of 0 for goeWork4)
        # Original Python: for i in range(sample, 0, -1) -> sample down to 1 for goeWork4[i]
        # Let's use Pine's range: sample_calc_len down to 0 (inclusive)
        for i_g_idx in range(sample_calc_len, 0, -1): # i_g_idx from sample_calc_len down to 1
            # Ensure i_g_idx is a valid index for goeWork4, though loop range should ensure this
            goe4_val = goeWork4[i_g_idx] if i_g_idx < len(goeWork4) and i_g_idx > 0 else 0.0
            w = cos_term * x - y + goe4_val
            y = x
            x = w
            
        term2 = x - y / 2.0 * cos_term
        if term2 == 0.0: term2 = 0.0000001 # Original was 1e-7

        term3 = y * math.sin(2.0 * math.pi * z_freq)

        # Cycle Strength/Amplitude calculation
        if UseCycleStrength:
            raw_amp_sq = math.pow(term2, 2) + math.pow(term3, 2)
            if k_period != 0:
                goeWork1[k_period] = (raw_amp_sq / k_period) if squaredAmp else (math.sqrt(raw_amp_sq) / k_period)
            else: goeWork1[k_period] = 0.0
        else: # Use Amplitude
            goeWork1[k_period] = math.pow(term2, 2) + math.pow(term3, 2) if squaredAmp \
                               else math.sqrt(math.pow(term2, 2) + math.pow(term3, 2))
        
        # Phase calculation
        current_phase = 0.0
        if term2 != 0.0: current_phase = math.atan(term3 / term2)
        else: current_phase = math.pi / 2.0 if term3 > 0 else -math.pi / 2.0

        if term2 < 0.0: current_phase += math.pi
        elif term3 < 0.0: current_phase += 2.0 * math.pi # term2 >= 0 and temp3 < 0
        goeWork2[k_period] = current_phase


    # Extract peaks (potential cycles)
    # Pine: for k = 3 to per -1. goeWork3[k] = k * 10^-4 if peak
    print(f"DEBUG GOERTZEL: Starting peak detection for periods 3 to {per-1}")
    peaks_found = 0
    for k_peak_check in range(3, per): # Checks period k_peak_check (3 to MaxPer-1)
        if k_peak_check + 1 < len(goeWork1) and k_peak_check -1 >=0 :
            amp_curr = goeWork1[k_peak_check] # np.nan_to_num not in original
            amp_next = goeWork1[k_peak_check + 1]
            amp_prev = goeWork1[k_peak_check - 1]
            
            # DEBUG: Show amplitude values for first few periods
            if k_peak_check <= 6:
                print(f"  Period {k_peak_check}: amp_prev={amp_prev:.6f}, amp_curr={amp_curr:.6f}, amp_next={amp_next:.6f}")
                
            if amp_curr > amp_next and amp_curr > amp_prev:
                goeWork3[k_peak_check] = k_peak_check * 1e-4
                peaks_found += 1
                print(f"  ✓ PEAK found at period {k_peak_check}, amplitude={amp_curr:.6f}")
            else:
                goeWork3[k_peak_check] = 0.0 # Explicitly set to 0 if not a peak
        # else: goeWork3[k_peak_check] remains 0.0 if bounds fail

    print(f"DEBUG GOERTZEL: Peak detection complete. Found {peaks_found} peaks")
    
    # Show some amplitude statistics
    valid_amps = [goeWork1[i] for i in range(2, per+1) if i < len(goeWork1)]
    if valid_amps:
        print(f"DEBUG GOERTZEL: Amplitude stats - min={np.min(valid_amps):.6f}, max={np.max(valid_amps):.6f}, mean={np.mean(valid_amps):.6f}")
        # Show first 10 amplitudes
        first_10_amps = valid_amps[:10]
        print(f"DEBUG GOERTZEL: First 10 amplitudes: {first_10_amps}")

    # Extract cycles
    number_of_cycles = 0
    # Original script: for i in range(len(goeWork3)):
    # goeWork3 length is 2*per+1. Peaks stored at index = period. Max period is 'per'.
    print(f"DEBUG GOERTZEL: Starting cycle extraction from goeWork3 (length={len(goeWork3)})")
    for i_extract_val_at_period in range(len(goeWork3)): # Iterate through goeWork3 (0 to 2*per)
        # i_extract_val_at_period here is the period number.
        goe3_val = goeWork3[i_extract_val_at_period] # Value at goeWork3[period]
        if goe3_val > 0.0: # If it's a peak marker
            extracted_period = round(10000.0 * goe3_val) # This is the actual period
            print(f"  Found peak marker at period {i_extract_val_at_period}: goe3_val={goe3_val:.8f}, extracted_period={extracted_period}")
            if extracted_period <= 0: 
                print(f"    Skipping - invalid extracted_period ({extracted_period})")
                continue

            number_of_cycles += 1
            print(f"    ✓ Adding cycle #{number_of_cycles}: period={extracted_period}")
            if number_of_cycles < len(cyclebuffer): # Buffers are 1-indexed
                cyclebuffer[number_of_cycles] = extracted_period
                # Amplitude and Phase are from goeWork1/2 at index = actual period
                # which is i_extract_val_at_period (the index of goeWork3 where peak was found)
                if i_extract_val_at_period < len(goeWork1): # Check bounds for safety
                    amplitudebuffer[number_of_cycles] = goeWork1[i_extract_val_at_period]
                    print(f"      Amplitude: {goeWork1[i_extract_val_at_period]:.6f}")
                if i_extract_val_at_period < len(goeWork2):
                    phasebuffer[number_of_cycles] = goeWork2[i_extract_val_at_period]
                    print(f"      Phase: {goeWork2[i_extract_val_at_period]:.6f}")
            else:
                print(f"    Warning: cyclebuffer size ({len(cyclebuffer)}) exceeded.")
                break

    print(f"DEBUG GOERTZEL: Cycle extraction complete. Found {number_of_cycles} valid cycles")

    # Order cycles by amplitude/strength (descending) - 1-indexed buffers
    if number_of_cycles >= 1: # Original was just `if number_of_cycles:`
        for i_sort in range(1, number_of_cycles): # Pine 1 to N-1
            for k_sort in range(i_sort + 1, number_of_cycles + 1): # Pine i+1 to N
                if k_sort < len(amplitudebuffer) and i_sort < len(amplitudebuffer): # Bounds from original
                    if amplitudebuffer[k_sort] > amplitudebuffer[i_sort]:
                        # y, w, x are temp vars for swap
                        y_amp = amplitudebuffer[i_sort]
                        w_cycle = cyclebuffer[i_sort]
                        x_phase = phasebuffer[i_sort]
                        amplitudebuffer[i_sort] = amplitudebuffer[k_sort]
                        cyclebuffer[i_sort] = cyclebuffer[k_sort]
                        phasebuffer[i_sort] = phasebuffer[k_sort]
                        amplitudebuffer[k_sort] = y_amp
                        cyclebuffer[k_sort] = w_cycle
                        phasebuffer[k_sort] = x_phase
                # else: print warning from original (index out of bounds) - omitted for brevity

    # Execute Bartels test
    if FilterBartels and number_of_cycles > 0: # Ensure number_of_cycles is positive
        print(f"DEBUG GOERTZEL: Applying Bartels Filter...")
        print(f"  BartSigLimit: {BartSigLimit}")
        print(f"  Cycles before Bartels: {number_of_cycles}")
        if len(cycleBartelsBuffer) >= number_of_cycles: # Check if buffer is large enough for results
            # bartels_cycle_test_func expects G=number_of_cycles,
            # and will access cyclebuffer[k] where k is 0 to G-1.
            # src for bartels_cycle_test is the main 'src' input to goertzel.
            bartels_cycle_test_func(src, number_of_cycles, cyclebuffer, cycleBartelsBuffer, BartNoCycles, BartSmoothPer)
            
            # DEBUG: Show Bartels scores
            print(f"  Bartels scores calculated:")
            for i in range(number_of_cycles):
                if i < len(cycleBartelsBuffer):
                    print(f"    Cycle {i+1} (Period {cyclebuffer[i]}): Bartels score = {cycleBartelsBuffer[i]:.2f}, BartSigLimit = {BartSigLimit}")
            
            no_Bcycles = 0
            # Original Pine loop for filtering was 0 to number_of_cycles - 1 (0-indexed i)
            # It compacted amplitudebuffer[no_Bcycles] = amplitudebuffer[i]
            # This implies a temporary 0-indexed treatment of these buffers for this step.
            for i_filter in range(number_of_cycles): # i_filter is 0-indexed
                if i_filter < len(cycleBartelsBuffer) and no_Bcycles < len(amplitudebuffer): # Bounds
                    if cycleBartelsBuffer[i_filter] > BartSigLimit:
                        print(f"    ✓ Cycle {i_filter+1} PASSED Bartels filter (score: {cycleBartelsBuffer[i_filter]:.2f} > {BartSigLimit})")
                        # Compaction using 0-indexed access for this block
                        amplitudebuffer[no_Bcycles] = amplitudebuffer[i_filter]
                        cyclebuffer[no_Bcycles] = cyclebuffer[i_filter]
                        phasebuffer[no_Bcycles] = phasebuffer[i_filter]
                        cycleBartelsBuffer[no_Bcycles] = cycleBartelsBuffer[i_filter] # Also compact Bartels scores
                        no_Bcycles += 1
                    else:
                        print(f"    ❌ Cycle {i_filter+1} FAILED Bartels filter (score: {cycleBartelsBuffer[i_filter]:.2f} <= {BartSigLimit})")
                # else: print warning from original (index out of bounds) - omitted

            number_of_cycles = no_Bcycles # Original: if no_Bcycles != 0 else 0
            print(f"  Cycles after Bartels filtering: {number_of_cycles}")

            if SortBartels and number_of_cycles >= 1: # number_of_cycles is now the count of 0-indexed items
                # Sort the 0-indexed compacted buffers.
                # Valid indices are 0 to number_of_cycles - 1.
                for i_sort_b in range(number_of_cycles - 1):  # Corrected: Loop from index 0 to N-2
                    for k_sort_b in range(i_sort_b + 1, number_of_cycles): # Corrected: Loop from index i+1 to N-1
                        # cycleBartelsBuffer now contains the scores for the compacted, 0-indexed cycles.
                        # Accesses are within 0 to number_of_cycles-1.
                        if cycleBartelsBuffer[k_sort_b] > cycleBartelsBuffer[i_sort_b]: # Sorting descending by Bartels score
                            # Swap all corresponding buffer elements
                            y_amp_b = amplitudebuffer[i_sort_b]
                            w_cycle_b = cyclebuffer[i_sort_b]
                            x_phase_b = phasebuffer[i_sort_b]
                            v_bart_b = cycleBartelsBuffer[i_sort_b]
                            
                            amplitudebuffer[i_sort_b] = amplitudebuffer[k_sort_b]
                            cyclebuffer[i_sort_b] = cyclebuffer[k_sort_b]
                            phasebuffer[i_sort_b] = phasebuffer[k_sort_b]
                            cycleBartelsBuffer[i_sort_b] = cycleBartelsBuffer[k_sort_b]
                            
                            amplitudebuffer[k_sort_b] = y_amp_b
                            cyclebuffer[k_sort_b] = w_cycle_b
                            phasebuffer[k_sort_b] = x_phase_b
                            cycleBartelsBuffer[k_sort_b] = v_bart_b

    # Calculate wave components
    # goeWorkPast/Future: rows are time steps, cols are 1-indexed cycle_num
    for i_cycle_num in range(1, number_of_cycles + 1):
        if i_cycle_num < len(amplitudebuffer) and \
           i_cycle_num < len(phasebuffer) and \
           i_cycle_num < len(cyclebuffer): # Check all 1-indexed buffers

            amplitude = amplitudebuffer[i_cycle_num - 1]
            phase = phasebuffer[i_cycle_num - 1]
            cycle_period = cyclebuffer[i_cycle_num - 1]

            if cycle_period == 0: continue

            sign_val = -1.0 if not useAddition else 1.0 # Matched original type

            # Past Wave Components
            for k_time_past in range(WindowSizePast):
                if k_time_past < goeWorkPast.shape[0] and i_cycle_num < goeWorkPast.shape[1]:
                    angle = phase + sign_val * k_time_past * 2.0 * math.pi / cycle_period
                    goeWorkPast[k_time_past, i_cycle_num] = amplitude * (math.cos(angle) if useCosine else math.sin(angle))
                # else: print warning from original - omitted

            sign_val *= -1 # Invert sign for future wave, as per original
            # Future Wave Components
            for k_time_future in range(WindowSizeFuture):
                # Pine: goeWorkFuture[WindowSizeFuture - k - 1, i_cycle_num] = ...
                # This stores future wave time-reversed.
                array_idx_future = WindowSizeFuture - k_time_future - 1
                if array_idx_future >= 0 and array_idx_future < goeWorkFuture.shape[0] and \
                   i_cycle_num < goeWorkFuture.shape[1]:
                    angle = phase + sign_val * k_time_future * 2.0 * math.pi / cycle_period
                    goeWorkFuture[array_idx_future, i_cycle_num] = amplitude * (math.cos(angle) if useCosine else math.sin(angle))
                # else: print warning from original - omitted
        # else: print warning from original - omitted

    # --- End Goertzel Debug ---
    # print("--- END DEBUG GOERTZEL ---\n")
    return number_of_cycles


if __name__ == '__main__':
    print("--- Testing core_algorithms.py (Direct Refactor) ---")

    # Test zero_lag_ma
    print("\nTesting zero_lag_ma...")
    src_test_zlma = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.0] * 2) 
    zlma_result = zero_lag_ma(src_test_zlma, 3, 10)
    print(f"ZLMA Result (first 5): {np.round(zlma_result[:5], 4)}")

    # Test Hodrick-Prescott Filter
    print("\nTesting hodrick_prescott_filter...")
    src_chronological_hpf = np.array([10,10.2,10.5,10.3,10.8,11.5,11.2,11.8,12.5,12.3,12.8,13.5,13.2,13.8,14.5]*2)
    hpf_per_test = 15
    # Original HPF call in main_calculator.py used lambda calculated based on HPsmoothPer.
    # Let's use a fixed one for this direct test for simplicity.
    hpf_lambda_test = 1600 
    hpf_result = hodrick_prescott_filter(src_chronological_hpf, hpf_lambda_test, hpf_per_test)
    print(f"HPF Trend Result (first 5): {np.round(hpf_result[:5], 4)}")

    print("\nBartels & Goertzel require full setup (see main_calculator or integration tests for full context).")
=======
# indicator_logic/core_algorithms.py

"""
Core algorithms for cycle analysis, including Zero-lag MA, Bartels Probability,
Hodrick-Prescott Filter, and Goertzel algorithm.
Logic is intended to be a direct refactor from indicator_milestone_1.py.
"""

import pandas as pd
import numpy as np
import math

# --- Zero-lag Moving Average (Replicating Flawed PineScript Logic) ---
def zero_lag_ma(src, smooth_per, bars_taken): # Matches original signature
    """
    Calculates the Zero-lag Moving Average (ZLMA) by replicating the
    *original flawed logic* found in the provided PineScript code.
    WARNING: This function intentionally replicates a bug where the sum
    and weight accumulators are NOT reset correctly within each pass.
    It will NOT produce a mathematically correct ZLMA. It is intended
    only for matching the output of the specific flawed PineScript version.
    Args:
        src (np.ndarray or pd.Series): The source data array. Assumed to be
                                     ordered with the most recent value at index 0.
        smooth_per (int): The smoothing period.
        bars_taken (int): The number of bars (elements) in the source array.
    Returns:
        np.ndarray: An array containing the ZLMA values.
    """
    # --- Start ZLMA Debug ---
    # print(f"\n--- DEBUG ZLMA (Flawed Replication) ---")
    # print(f"Entering with smooth_per={smooth_per}, bars_taken={bars_taken}, src length={len(src)}")
    # ---

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series):
            src = src.values
        else:
            try:
                src = np.array(src, dtype=float)
            except Exception as e:
                print(f"Error converting src to numpy array in zero_lag_ma: {e}")
                return np.array([])

    if bars_taken > len(src):
        # print(f"Warning: bars_taken ({bars_taken}) > src length ({len(src)}). Adjusting.")
        bars_taken = len(src)
    if bars_taken <= 0 or smooth_per <= 0:
        # print(f"Error: bars_taken ({bars_taken}) and smooth_per ({smooth_per}) must be positive.")
        return np.array([])

    src_subset = src[:bars_taken]
    # --- ZLMA Debug ---
    # print(f"Using src_subset of length: {len(src_subset)}")
    # ---

    lwma1 = np.zeros(bars_taken)
    output = np.zeros(bars_taken)

    # --- First LWMA pass (Replicating PineScript Flaw) ---
    # print("Starting Pass 1 (Flawed Logic)...")
    sum_val = 0.0
    sum_w = 0.0
    for i in range(bars_taken - 1, -1, -1):
        # inner_sum_added = 0.0 # From original debug
        # inner_w_added = 0.0 # From original debug
        for k in range(smooth_per):
            index = i + k
            if index >= 0 and index < bars_taken - 1: # Replicate exact condition
                 if index < len(src_subset):
                     src_value = src_subset[index]
                     if not np.isnan(src_value):
                        weight = float(smooth_per - k)
                        sum_w += weight
                        sum_val += weight * src_value
                        # inner_sum_added += weight * src_value # From original debug
                        # inner_w_added += weight # From original debug
        if sum_w != 0.0:
            lwma1[i] = sum_val / sum_w
        else:
            lwma1[i] = 0.0

    # --- ZLMA Debug ---
    # print(f"DEBUG ZLMA (Flawed Replication): lwma1 calculated (shape: {lwma1.shape})")
    # print(f"DEBUG ZLMA (Flawed Replication): lwma1 (first 10): {np.round(lwma1[:10], 8)}")
    # print(f"DEBUG ZLMA (Flawed Replication): lwma1 (last 10): {np.round(lwma1[-10:], 8)}")
    # ---

    # --- Second LWMA pass (Replicating PineScript Flaw) ---
    # print("Starting Pass 2 (Flawed Logic)...")
    sum_val = 0.0
    sum_w = 0.0
    for i in range(bars_taken):
        # inner_sum_added = 0.0 # From original debug
        # inner_w_added = 0.0 # From original debug
        for k in range(smooth_per):
            index = i - k
            if index >= 0: # PineScript only checked this
                 if index < len(lwma1):
                     lwma1_value = lwma1[index]
                     if not np.isnan(lwma1_value):
                         weight = float(smooth_per - k)
                         sum_w += weight
                         sum_val += weight * lwma1_value
                         # inner_sum_added += weight * lwma1_value # From original debug
                         # inner_w_added += weight # From original debug
        if sum_w != 0.0:
            output[i] = sum_val / sum_w
        else:
            output[i] = 0.0

    # --- ZLMA Debug ---
    # print(f"DEBUG ZLMA (Flawed Replication): output calculated (shape: {output.shape})")
    # print(f"DEBUG ZLMA (Flawed Replication): output (first 10): {np.round(output[:10], 8)}")
    # print(f"DEBUG ZLMA (Flawed Replication): output (last 10): {np.round(output[-10:], 8)}")
    # print(f"DEBUG ZLMA (Flawed Replication): Exiting")
    # print(f"--- END DEBUG ZLMA (Flawed Replication) ---\n")
    return output

# --- Bartels probability ---
def bartels_prob(n, N, Bvalues): # Matches original signature
    """
    Calculates the Bartels probability.
    Identical to indicator_milestone_1.py.
    """
    # --- Start BartelsProb Debug ---
    # print(f"\n--- DEBUG BARTELS_PROB ---")
    # print(f"Entering with n={n}, N={N}")
    # ---

    if not isinstance(Bvalues, np.ndarray):
        try:
            Bvalues = np.array(Bvalues, dtype=float)
        except Exception as e:
            # print(f"Error converting Bvalues to numpy array: {e}")
            return 0.0

    # --- BartelsProb Debug ---
    # print(f"Bvalues shape: {Bvalues.shape}")
    # print(f"Bvalues (first 10): {np.round(Bvalues[:10], 8)}")
    # ---
    if N <= 0 or n <= 0: # Added basic check, original relied on shape
        return 0.0
    if Bvalues.shape[0] != N * n:
        # print(f"Error: Bvalues array shape ({Bvalues.shape[0]}) does not match expected shape ({N * n}).")
        return 0.0

    teta = np.zeros(n)
    vsin = np.zeros(n)
    vcos = np.zeros(n)
    CoeffA = np.zeros(N)
    CoeffB = np.zeros(N)
    IndAmplit = np.zeros(N)

    for i in range(n):
        teta[i] = 1.0 * (i + 1) / n * 2 * math.pi
        vsin[i] = math.sin(teta[i])
        vcos[i] = math.cos(teta[i])
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: teta (first 5): {np.round(teta[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: vsin (first 5): {np.round(vsin[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: vcos (first 5): {np.round(vcos[:5], 8)}")
    # ---

    for t_loop in range(N): # Renamed t to t_loop
        for i in range(n):
            # Ensure index is within bounds for Bvalues (already checked by Bvalues.shape[0] != N * n)
            CoeffA[t_loop] += vsin[i] * Bvalues[t_loop * n + i]
            CoeffB[t_loop] += vcos[i] * Bvalues[t_loop * n + i]
        IndAmplit[t_loop] = math.pow(CoeffA[t_loop], 2) + math.pow(CoeffB[t_loop], 2)
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: CoeffA (first 5): {np.round(CoeffA[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: CoeffB (first 5): {np.round(CoeffB[:5], 8)}")
    # print(f"DEBUG BARTELS_PROB: IndAmplit (first 5): {np.round(IndAmplit[:5], 8)}")
    # ---

    AvgCoeffA = np.sum(CoeffA) / N
    AvgCoeffB = np.sum(CoeffB) / N
    AvgIndAmplit = np.sum(IndAmplit) / N

    AvgAmpl = math.sqrt(math.pow(AvgCoeffA, 2) + math.pow(AvgCoeffB, 2))
    AvgIndAmplit_sqrt = math.sqrt(AvgIndAmplit) # Pine script takes sqrt after averaging
    ExptAmpl = AvgIndAmplit_sqrt / math.sqrt(1.0 * N)
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: AvgCoeffA={AvgCoeffA:.6f}, AvgCoeffB={AvgCoeffB:.6f}, AvgIndAmplit={AvgIndAmplit:.6f}")
    # print(f"DEBUG BARTELS_PROB: AvgAmpl={AvgAmpl:.6f}, AvgIndAmplit_sqrt={AvgIndAmplit_sqrt:.6f}, ExptAmpl={ExptAmpl:.6f}")
    # ---

    if ExptAmpl == 0:
        ARatio = 0.0
    else:
        ARatio = AvgAmpl / ExptAmpl
    
    BP = 0.0
    try:
        BP = 1 / math.exp(math.pow(ARatio, 2))
    except OverflowError: # If ARatio is too large, exp overflows
        BP = 0.0
    # --- BartelsProb Debug ---
    # print(f"DEBUG BARTELS_PROB: ARatio={ARatio:.6f}, Final BP={BP:.8f}")
    # print(f"DEBUG BARTELS_PROB: Exiting")
    # print(f"--- END DEBUG BARTELS_PROB ---\n")
    return BP

# --- Hodrick-Prescott Filter ---
def hodrick_prescott_filter(src, lamb, per): # Matches original signature
    """
    Implements the Hodrick-Prescott filter.
    Identical to indicator_milestone_1.py.
    """
    # --- Start HPF Debug ---
    # print(f"\n--- DEBUG HPF ---")
    # print(f"Entering with lamb={lamb:.6f}, per={per}, src length={len(src)}")
    # ---

    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series):
            src = src.values
        else:
            try:
                src = np.array(src, dtype=float)
            except Exception as e:
                # print(f"Error converting src to numpy array: {e}")
                return np.array([])

    if per > len(src):
        # print(f"Warning: per ({per}) is greater than source length ({len(src)}). Adjusting to source length.")
        per = len(src)
    if per <= 0:
        # print(f"Error: per ({per}) must be positive.")
        return np.array([])

    src_subset = src[-per:]

    H1, H2, H3, H4, H5 = 0., 0., 0., 0., 0.
    HH1, HH2, HH3, HH5 = 0., 0., 0., 0.
    # HB, HC, Z were local loop vars in original

    a = np.zeros(per)
    b = np.zeros(per)
    c_arr = np.zeros(per) # Renamed from 'c'
    output = np.copy(src_subset)
    # --- HPF Debug ---
    # print(f"DEBUG HPF: Initialized arrays a, b, c, output with shape ({per},)")
    # print(f"DEBUG HPF: Initial output (first 10): {np.round(output[:10], 6)}")
    # print(f"DEBUG HPF: Initial src_subset (first 10): {np.round(src_subset[:10], 6)}")
    # ---

    # Coefficient initialization IDENTICAL to indicator_milestone_1.py
    if per > 0:
        a[0] = 1.0 + lamb
        b[0] = -2.0 * lamb
        c_arr[0] = lamb

    for i in range(1, per - 3): # Original loop was 1 to per-3 (exclusive end)
        a[i] = 6.0 * lamb + 1.0
        b[i] = -4.0 * lamb
        c_arr[i] = lamb
    
    # Boundary conditions IDENTICAL to indicator_milestone_1.py
    if per >= 2:
        a[1] = 5.0 * lamb + 1.0
    if per >= 1:
        a[per - 1] = 1.0 + lamb
    if per >= 2:
        a[per - 2] = 5.0 * lamb + 1.0
        b[per - 2] = -2.0 * lamb
    # These assignments will overwrite previous ones if per is small
    if per >= 1: # Ensure per-1 is a valid index
      b[per - 1] = 0.0
    if per >= 2: # Ensure per-2 is valid
      c_arr[per - 2] = 0.0
    if per >= 1:
      c_arr[per - 1] = 0.0
      
    # --- HPF Debug ---
    # print(f"DEBUG HPF: Initialized a (first 10): {np.round(a[:10], 6)}")
    # print(f"DEBUG HPF: Initialized b (first 10): {np.round(b[:10], 6)}")
    # print(f"DEBUG HPF: Initialized c (first 10): {np.round(c_arr[:10], 6)}")
    # print("DEBUG HPF: Starting forward pass...")
    # ---

    # Forward pass
    for i in range(per):
        # a_i, b_i, c_i from original debug, not needed for logic
        Z_val = a[i] - H4 * H1 - HH5 * HH2 # Use a[i] as per original
        if Z_val == 0:
            # print(f"  DEBUG HPF (Forward i={i}): Z is ZERO! Aborting.")
            # print(f"Error: Division by zero in Hodrick-Prescott forward pass at index {i}.")
            return np.zeros(per)

        HB_val = b[i] # Use b[i] as per original
        HH1 = H1
        H1 = (HB_val - H4 * H2) / Z_val
        b[i] = H1 # Store modified b

        HC_val = c_arr[i] # Use c_arr[i] as per original
        HH2 = H2
        H2 = HC_val / Z_val
        c_arr[i] = H2 # Store modified c_arr

        src_val_iter = src_subset[i] # src_val_iter instead of src_val
        # Original: a_val = (src_val - HH3_val * HH5_val - H3_val * H4_val) / Z
        # HH3_val, HH5_val etc. were from previous iteration's H3, H5.
        # So, current HH3, HH5, H3, H4 are used here.
        a_val_modified = (src_val_iter - HH3 * HH5 - H3 * H4) / Z_val
        a[i] = a_val_modified # Store modified a

        HH3 = H3
        H3 = a[i] 
        H4 = HB_val - H5 * HH1 # H4 uses b[i] before modification in this iteration
        HH5 = H5
        H5 = HC_val # H5 uses c_arr[i] before modification in this iteration
    
    # Backward pass (original script H1, H2 were reset before this loop)
    H2_bp = 0.0 # Using _bp suffix for clarity in backward pass
    H1_bp = 0.0
    if per > 0:
        H1_bp = a[per-1] # a has been modified by forward pass
        output[per-1] = H1_bp

    # --- HPF Debug ---
    # print("DEBUG HPF: Starting backward pass...")
    # if per > 0:
        # print(f"  DEBUG HPF (Backward Start): Initial output[{per-1}]={output[per-1]:.4f}, H1={H1_bp:.4f}, H2={H2_bp:.4f}")
    # ---

    for i in range(per - 2, -1, -1):
        # output_i = a[i] - b[i] * H1_bp - c_arr[i] * H2_bp # Uses modified a,b,c_arr
        output[i] = a[i] - b[i] * H1_bp - c_arr[i] * H2_bp
        H2_bp = H1_bp
        H1_bp = output[i]

    # --- HPF Debug ---
    # print("DEBUG HPF: Backward pass complete.")
    # print(f"DEBUG HPF: Final output (first 10): {np.round(output[:10], 6)}")
    # print(f"DEBUG HPF: Final output (last 10): {np.round(output[-10:], 6)}")
    # print(f"--- END DEBUG HPF ---\n")
    return output

# --- Goertzel Browser ---
# Matches original signature, but bartels_cycle_test is now dependency injected
def goertzel(src, forBar, samplesize, per, squaredAmp, useAddition, useCosine, UseCycleStrength, WindowSizePast,
             WindowSizeFuture, FilterBartels, BartNoCycles, BartSmoothPer, BartSigLimit,
             SortBartels, goeWorkPast, goeWorkFuture, cyclebuffer,
             amplitudebuffer, phasebuffer, cycleBartelsBuffer,
             bartels_cycle_test_func # Dependency injection
             ):
    """
    Implements the Goertzel algorithm for spectral analysis.
    Identical to indicator_milestone_1.py, with bartels_cycle_test injected.
    """
    # --- Start Goertzel Debug ---
    # print("\n--- DEBUG GOERTZEL ---")
    # (Original extensive debug prints for args omitted for brevity but can be re-added)
    # ---
    if not isinstance(src, np.ndarray):
        if isinstance(src, pd.Series): src = src.values
        else:
            try: src = np.array(src, dtype=float)
            except Exception as e: 
                # print(f"Error converting src to numpy array: {e}")
                return 0
    
    # Original script sample = 2 * per (MaxPer)
    sample_calc_len = 2 * per # Renamed 'sample' to 'sample_calc_len' for clarity
    if sample_calc_len <= 0: return 0 # MaxPer must be >0

    goeWork1 = np.zeros(sample_calc_len + 1)
    goeWork2 = np.zeros(sample_calc_len + 1)
    goeWork3 = np.zeros(sample_calc_len + 1) # Peaks. Index is period.
    goeWork4 = np.zeros(sample_calc_len + 1) # Detrended src. Index is time within 2*MaxPer window.

    # Ensure indices are within bounds for src.
    # forBar is offset. sample_calc_len is window size for Goertzel transform.
    if forBar + sample_calc_len - 1 >= len(src) or forBar < 0:
         # print(f"Error: forBar ({forBar}) or sample size ({sample_calc_len}) out of bounds for src length ({len(src)}).")
         return 0

    # print("Calculating initial detrending (goeWork4)...")
    # Linear detrend over the sample_calc_len window of src
    # src is [recent, ..., oldest]
    # temp1 is oldest value in window: src[forBar + sample_calc_len - 1]
    # temp2 is slope: (src[forBar] - temp1) / (sample_calc_len - 1)
    if sample_calc_len > 1: # Original script did not check for sample_calc_len=1 here for temp2
        temp1_val = src[forBar + sample_calc_len - 1] # Oldest value in window
        temp2_val = (src[forBar] - temp1_val) / (sample_calc_len - 1) # Slope
    elif sample_calc_len == 1: # Handle if only 1 point in window
        temp1_val = src[forBar]
        temp2_val = 0 # No slope for a single point
    else: # sample_calc_len <= 0, already returned
        return 0


    # Pine: for k = sample - 1 down to 0. Python: sample_calc_len-1 down to 0
    # Original: for k in range(sample - 1, 0, -1): -> k from sample-1 down to 1
    # This populates goeWork4[k] where k is 1-indexed time in Pine.
    for k_gw4 in range(sample_calc_len - 1, -1, -1): # k_gw4 from sample_calc_len-1 down to 0
        if forBar + k_gw4 < len(src) : # Check src index
            # Original: temp = src[forBar + k - 1] - (temp1 + temp2 * (sample - k))
            # k_gw4 is distance from "end" of window (0 is oldest, sample_calc_len-1 is newest in this loop)
            # Pine k was sample-1 down to 1. Let's use Pine's k for easier mapping
            # For k_pine = sample_calc_len-1 down to 1 (as in original `goeWork4[k] = temp`)
            # Equivalent python k_gw4 here would be k_pine-1
            # Let's replicate original directly:
            # Loop k from sample_calc_len-1 down to 1 for goeWork4[k_pine]
            if k_gw4 > 0 : # Only for k_pine from sample-1 down to 1
                src_idx = forBar + k_gw4 -1 # src[forBar + (k_pine-style)-1]
                # trend = temp1_val(oldest) + slope * (distance from oldest point)
                # distance from oldest point for src[forBar + k_pine-1] is ( (forBar+k_pine-1) - (forBar+sample_calc_len-1) ) in original time
                # Pine: temp1 + temp2 * (sample - k_pine)
                # (sample_calc_len - k_gw4) term for Pine's k_pine
                trend_val = temp1_val + temp2_val * (sample_calc_len - k_gw4)
                goeWork4[k_gw4] = src[src_idx] - trend_val
            # goeWork3[k_gw4] = 0. (already initialized)
        # else: out of bounds, goeWork4[k_gw4] remains 0.

    # goeWork3[0] = 0 (already initialized)

    # Main Goertzel calculation loop (k_period from 2 to MaxPer)
    for k_period in range(2, per + 1):
        w, x, y = 0.0, 0.0, 0.0
        z_freq = 1.0 / k_period
        cos_term = 2.0 * math.cos(2.0 * math.pi * z_freq)

        # Pine loop: for i = sample down to 0 (inclusive of 0 for goeWork4)
        # Original Python: for i in range(sample, 0, -1) -> sample down to 1 for goeWork4[i]
        # Let's use Pine's range: sample_calc_len down to 0 (inclusive)
        for i_g_idx in range(sample_calc_len, 0, -1): # i_g_idx from sample_calc_len down to 1
            # Ensure i_g_idx is a valid index for goeWork4, though loop range should ensure this
            goe4_val = goeWork4[i_g_idx] if i_g_idx < len(goeWork4) and i_g_idx > 0 else 0.0
            w = cos_term * x - y + goe4_val
            y = x
            x = w
            
        term2 = x - y / 2.0 * cos_term
        if term2 == 0.0: term2 = 0.0000001 # Original was 1e-7

        term3 = y * math.sin(2.0 * math.pi * z_freq)

        # Cycle Strength/Amplitude calculation
        if UseCycleStrength:
            raw_amp_sq = math.pow(term2, 2) + math.pow(term3, 2)
            if k_period != 0:
                goeWork1[k_period] = (raw_amp_sq / k_period) if squaredAmp else (math.sqrt(raw_amp_sq) / k_period)
            else: goeWork1[k_period] = 0.0
        else: # Use Amplitude
            goeWork1[k_period] = math.pow(term2, 2) + math.pow(term3, 2) if squaredAmp \
                               else math.sqrt(math.pow(term2, 2) + math.pow(term3, 2))
        
        # Phase calculation
        current_phase = 0.0
        if term2 != 0.0: current_phase = math.atan(term3 / term2)
        else: current_phase = math.pi / 2.0 if term3 > 0 else -math.pi / 2.0

        if term2 < 0.0: current_phase += math.pi
        elif term3 < 0.0: current_phase += 2.0 * math.pi # term2 >= 0 and temp3 < 0
        goeWork2[k_period] = current_phase


    # Extract peaks (potential cycles)
    # Pine: for k = 3 to per -1. goeWork3[k] = k * 10^-4 if peak
    for k_peak_check in range(3, per): # Checks period k_peak_check (3 to MaxPer-1)
        if k_peak_check + 1 < len(goeWork1) and k_peak_check -1 >=0 :
            amp_curr = goeWork1[k_peak_check] # np.nan_to_num not in original
            amp_next = goeWork1[k_peak_check + 1]
            amp_prev = goeWork1[k_peak_check - 1]
            if amp_curr > amp_next and amp_curr > amp_prev:
                goeWork3[k_peak_check] = k_peak_check * 1e-4
            else:
                goeWork3[k_peak_check] = 0.0 # Explicitly set to 0 if not a peak
        # else: goeWork3[k_peak_check] remains 0.0 if bounds fail

    # Extract cycles
    number_of_cycles = 0
    # Original script: for i in range(len(goeWork3)):
    # goeWork3 length is 2*per+1. Peaks stored at index = period. Max period is 'per'.
    for i_extract_val_at_period in range(len(goeWork3)): # Iterate through goeWork3 (0 to 2*per)
        # i_extract_val_at_period here is the period number.
        goe3_val = goeWork3[i_extract_val_at_period] # Value at goeWork3[period]
        if goe3_val > 0.0: # If it's a peak marker
            extracted_period = round(10000.0 * goe3_val) # This is the actual period
            if extracted_period <= 0: continue

            number_of_cycles += 1
            if number_of_cycles < len(cyclebuffer): # Buffers are 1-indexed
                cyclebuffer[number_of_cycles] = extracted_period
                # Amplitude and Phase are from goeWork1/2 at index = actual period
                # which is i_extract_val_at_period (the index of goeWork3 where peak was found)
                if i_extract_val_at_period < len(goeWork1): # Check bounds for safety
                    amplitudebuffer[number_of_cycles] = goeWork1[i_extract_val_at_period]
                if i_extract_val_at_period < len(goeWork2):
                    phasebuffer[number_of_cycles] = goeWork2[i_extract_val_at_period]
            else:
                # print(f"Warning: cyclebuffer size ({len(cyclebuffer)}) exceeded.")
                break

    # Order cycles by amplitude/strength (descending) - 1-indexed buffers
    if number_of_cycles >= 1: # Original was just `if number_of_cycles:`
        for i_sort in range(1, number_of_cycles): # Pine 1 to N-1
            for k_sort in range(i_sort + 1, number_of_cycles + 1): # Pine i+1 to N
                if k_sort < len(amplitudebuffer) and i_sort < len(amplitudebuffer): # Bounds from original
                    if amplitudebuffer[k_sort] > amplitudebuffer[i_sort]:
                        # y, w, x are temp vars for swap
                        y_amp = amplitudebuffer[i_sort]
                        w_cycle = cyclebuffer[i_sort]
                        x_phase = phasebuffer[i_sort]
                        amplitudebuffer[i_sort] = amplitudebuffer[k_sort]
                        cyclebuffer[i_sort] = cyclebuffer[k_sort]
                        phasebuffer[i_sort] = phasebuffer[k_sort]
                        amplitudebuffer[k_sort] = y_amp
                        cyclebuffer[k_sort] = w_cycle
                        phasebuffer[k_sort] = x_phase
                # else: print warning from original (index out of bounds) - omitted for brevity

    # Execute Bartels test
    if FilterBartels and number_of_cycles > 0: # Ensure number_of_cycles is positive
        # print("\nDEBUG GOERTZEL: Applying Bartels Filter...")
        if len(cycleBartelsBuffer) >= number_of_cycles: # Check if buffer is large enough for results
            # bartels_cycle_test_func expects G=number_of_cycles,
            # and will access cyclebuffer[k] where k is 0 to G-1.
            # src for bartels_cycle_test is the main 'src' input to goertzel.
            bartels_cycle_test_func(src, number_of_cycles, cyclebuffer, cycleBartelsBuffer, BartNoCycles, BartSmoothPer)
            
            no_Bcycles = 0
            # Original Pine loop for filtering was 0 to number_of_cycles - 1 (0-indexed i)
            # It compacted amplitudebuffer[no_Bcycles] = amplitudebuffer[i]
            # This implies a temporary 0-indexed treatment of these buffers for this step.
            for i_filter in range(number_of_cycles): # i_filter is 0-indexed
                if i_filter < len(cycleBartelsBuffer) and no_Bcycles < len(amplitudebuffer): # Bounds
                    if cycleBartelsBuffer[i_filter] > BartSigLimit:
                        # print(f"  Cycle {i_filter+1} (Period {cyclebuffer[i_filter]}) PASSED Bartels...") from original debug
                        # Compaction using 0-indexed access for this block
                        amplitudebuffer[no_Bcycles] = amplitudebuffer[i_filter]
                        cyclebuffer[no_Bcycles] = cyclebuffer[i_filter]
                        phasebuffer[no_Bcycles] = phasebuffer[i_filter]
                        cycleBartelsBuffer[no_Bcycles] = cycleBartelsBuffer[i_filter] # Also compact Bartels scores
                        no_Bcycles += 1
                # else: print warning from original (index out of bounds) - omitted

            number_of_cycles = no_Bcycles # Original: if no_Bcycles != 0 else 0

            if SortBartels and number_of_cycles >= 1: # number_of_cycles is now the count of 0-indexed items
                # Sort the 0-indexed compacted buffers.
                # Valid indices are 0 to number_of_cycles - 1.
                for i_sort_b in range(number_of_cycles - 1):  # Corrected: Loop from index 0 to N-2
                    for k_sort_b in range(i_sort_b + 1, number_of_cycles): # Corrected: Loop from index i+1 to N-1
                        # cycleBartelsBuffer now contains the scores for the compacted, 0-indexed cycles.
                        # Accesses are within 0 to number_of_cycles-1.
                        if cycleBartelsBuffer[k_sort_b] > cycleBartelsBuffer[i_sort_b]: # Sorting descending by Bartels score
                            # Swap all corresponding buffer elements
                            y_amp_b = amplitudebuffer[i_sort_b]
                            w_cycle_b = cyclebuffer[i_sort_b]
                            x_phase_b = phasebuffer[i_sort_b]
                            v_bart_b = cycleBartelsBuffer[i_sort_b]
                            
                            amplitudebuffer[i_sort_b] = amplitudebuffer[k_sort_b]
                            cyclebuffer[i_sort_b] = cyclebuffer[k_sort_b]
                            phasebuffer[i_sort_b] = phasebuffer[k_sort_b]
                            cycleBartelsBuffer[i_sort_b] = cycleBartelsBuffer[k_sort_b]
                            
                            amplitudebuffer[k_sort_b] = y_amp_b
                            cyclebuffer[k_sort_b] = w_cycle_b
                            phasebuffer[k_sort_b] = x_phase_b
                            cycleBartelsBuffer[k_sort_b] = v_bart_b

    # Calculate wave components
    # goeWorkPast/Future: rows are time steps, cols are 1-indexed cycle_num
    for i_cycle_num in range(1, number_of_cycles + 1):
        if i_cycle_num < len(amplitudebuffer) and \
           i_cycle_num < len(phasebuffer) and \
           i_cycle_num < len(cyclebuffer): # Check all 1-indexed buffers

            amplitude = amplitudebuffer[i_cycle_num - 1]
            phase = phasebuffer[i_cycle_num - 1]
            cycle_period = cyclebuffer[i_cycle_num - 1]

            if cycle_period == 0: continue

            sign_val = -1.0 if not useAddition else 1.0 # Matched original type

            # Past Wave Components
            for k_time_past in range(WindowSizePast):
                if k_time_past < goeWorkPast.shape[0] and i_cycle_num < goeWorkPast.shape[1]:
                    angle = phase + sign_val * k_time_past * 2.0 * math.pi / cycle_period
                    goeWorkPast[k_time_past, i_cycle_num] = amplitude * (math.cos(angle) if useCosine else math.sin(angle))
                # else: print warning from original - omitted

            sign_val *= -1 # Invert sign for future wave, as per original
            # Future Wave Components
            for k_time_future in range(WindowSizeFuture):
                # Pine: goeWorkFuture[WindowSizeFuture - k - 1, i_cycle_num] = ...
                # This stores future wave time-reversed.
                array_idx_future = WindowSizeFuture - k_time_future - 1
                if array_idx_future >= 0 and array_idx_future < goeWorkFuture.shape[0] and \
                   i_cycle_num < goeWorkFuture.shape[1]:
                    angle = phase + sign_val * k_time_future * 2.0 * math.pi / cycle_period
                    goeWorkFuture[array_idx_future, i_cycle_num] = amplitude * (math.cos(angle) if useCosine else math.sin(angle))
                # else: print warning from original - omitted
        # else: print warning from original - omitted

    # --- End Goertzel Debug ---
    # print("--- END DEBUG GOERTZEL ---\n")
    return number_of_cycles


if __name__ == '__main__':
    print("--- Testing core_algorithms.py (Direct Refactor) ---")

    # Test zero_lag_ma
    print("\nTesting zero_lag_ma...")
    src_test_zlma = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20.0] * 2) 
    zlma_result = zero_lag_ma(src_test_zlma, 3, 10)
    print(f"ZLMA Result (first 5): {np.round(zlma_result[:5], 4)}")

    # Test Hodrick-Prescott Filter
    print("\nTesting hodrick_prescott_filter...")
    src_chronological_hpf = np.array([10,10.2,10.5,10.3,10.8,11.5,11.2,11.8,12.5,12.3,12.8,13.5,13.2,13.8,14.5]*2)
    hpf_per_test = 15
    # Original HPF call in main_calculator.py used lambda calculated based on HPsmoothPer.
    # Let's use a fixed one for this direct test for simplicity.
    hpf_lambda_test = 1600 
    hpf_result = hodrick_prescott_filter(src_chronological_hpf, hpf_lambda_test, hpf_per_test)
    print(f"HPF Trend Result (first 5): {np.round(hpf_result[:5], 4)}")

    print("\nBartels & Goertzel require full setup (see main_calculator or integration tests for full context).")
>>>>>>> 07e747c20443eb8cbfdc5b93205384046f029227
