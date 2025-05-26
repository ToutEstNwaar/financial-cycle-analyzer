# utils/plotting.py

"""
Utility functions for plotting, creating data tables, and saving indicator results.
"""

import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import mplfinance as mpf # Ensure this import is present
import datetime
import json

# Import color settings from the indicator_logic package
try:
    from indicator_logic import settings as ind_settings
except ImportError:
    print("Warning: Could not import 'indicator_logic.settings' in 'utils.plotting'. Using fallback local color defaults.")
    class FallbackSettings:
        PAST_COLORS = {'up': '#2DD204', 'down': '#D2042D'}
        FUTURE_COLORS = {'up': 'fuchsia', 'down': 'yellow'}
        FUTURE_FLIP_COLORS = {'up': 'fuchsia', 'down': 'yellow'}
    ind_settings = FallbackSettings()

DEFAULT_PAST_COLORS = ind_settings.PAST_COLORS
DEFAULT_FUTURE_COLORS = ind_settings.FUTURE_COLORS
DEFAULT_FUTURE_FLIP_COLORS = ind_settings.FUTURE_FLIP_COLORS


def plot_indicator_lines(ohlc_data: pd.DataFrame,
                         past_wave: np.ndarray,
                         future_wave: np.ndarray,
                         calc_bar_index: int,
                         window_past: int,
                         window_future: int,
                         past_colors_dict: dict | None = None,
                         future_colors_dict: dict | None = None,
                         future_flip_colors_dict: dict | None = None,
                         title: str = "Cycle Indicator Overlay") -> plt.Figure | None:
    """
    Plots OHLC data and overlays past/future cycle lines.
    Returns a matplotlib Figure object for use in Streamlit.
    Does NOT call mpf.show().
    """
    past_colors_to_use = past_colors_dict if past_colors_dict is not None else DEFAULT_PAST_COLORS
    future_colors_to_use = future_colors_dict if future_colors_dict is not None else DEFAULT_FUTURE_COLORS
    future_flip_colors_to_use = future_flip_colors_dict if future_flip_colors_dict is not None else DEFAULT_FUTURE_FLIP_COLORS

    if not isinstance(ohlc_data.index, pd.DatetimeIndex):
        print("Warning: ohlc_data.index is not a DatetimeIndex. Plotting may be affected.")

    if len(past_wave) < 2 or len(future_wave) < 2:
        if not ohlc_data.empty:
            # Increase warn_too_much_data here as well if plotting full data as fallback
            fig, _ = mpf.plot(ohlc_data, type='candle', style='yahoo', title=title + " (Wave data too short)", 
                              returnfig=True, figsize=(15,7), warn_too_much_data=len(ohlc_data)+1 if len(ohlc_data) > 250 else 250)
            return fig
        return None

    plot_start_idx = max(0, calc_bar_index - window_past - 20)
    plot_end_idx = min(len(ohlc_data), calc_bar_index + window_future + 20)
    plot_data = ohlc_data.iloc[plot_start_idx:plot_end_idx]

    if plot_data.empty:
        print("Warning: No data to plot after slicing for plot range.")
        return None

    # Set warn_too_much_data to a value slightly above mplfinance's default (200 for candles)
    # This should suppress the warning for typical default plots (like 141 points)
    # but still allow it for very large plots if the user expands the window significantly.
    custom_warn_too_much_data = 250 

    fig, axlist = mpf.plot(plot_data, type='candle', style='yahoo', title=title, 
                           volume=False, returnfig=True, figsize=(15, 7),
                           warn_too_much_data=custom_warn_too_much_data) # Added kwarg
    ax = axlist[0] if isinstance(axlist, list) else axlist

    numerical_calc_bar_index_in_plot = -1
    try:
        calc_bar_date = ohlc_data.index[calc_bar_index]
        plotted_dates_list = plot_data.index.tolist()
        if calc_bar_date in plotted_dates_list:
            numerical_calc_bar_index_in_plot = plotted_dates_list.index(calc_bar_date)
        else:
            pass 
    except (IndexError, ValueError) as e:
        print(f"Error mapping calc_bar_index to plotted data: {e}")
        pass 

    if numerical_calc_bar_index_in_plot != -1 and len(past_wave) >= 2:
        prev_past_color = None
        for i in range(len(past_wave) - 1):
            line_numerical_index = numerical_calc_bar_index_in_plot - i
            direction = 'up' if past_wave[i] > past_wave[i + 1] else 'down'
            current_color = past_colors_to_use.get(direction, 'gray')
            if prev_past_color is not None and prev_past_color != current_color:
                ax.axvline(x=line_numerical_index, color=current_color, linestyle='-', linewidth=1)
            prev_past_color = current_color
        last_past_direction = 'up' if past_wave[len(past_wave)-2] > past_wave[len(past_wave)-1] else 'down'
    else:
        last_past_direction = None

    if numerical_calc_bar_index_in_plot != -1 and len(future_wave) >= 2 and last_past_direction is not None:
        prev_future_color = None
        first_future_direction = 'up' if future_wave[0] > future_wave[1] else 'down'
        
        color_for_first_future_line = future_colors_to_use.get(first_future_direction, 'orange')
        if first_future_direction != last_past_direction:
            color_for_first_future_line = future_flip_colors_to_use.get(first_future_direction, 'purple')
        
        ax.axvline(x=numerical_calc_bar_index_in_plot + 1, color=color_for_first_future_line, linestyle='--', linewidth=1)
        prev_future_color = color_for_first_future_line
            
        for i in range(1, len(future_wave) - 1):
            line_idx = numerical_calc_bar_index_in_plot + i + 1
            direction = 'up' if future_wave[i] > future_wave[i+1] else 'down'
            current_color = future_colors_to_use.get(direction, 'orange')
            if prev_future_color != current_color: 
                ax.axvline(x=line_idx, color=current_color, linestyle='--', linewidth=1)
            prev_future_color = current_color
    return fig

def create_cycle_table(number_of_cycles: int, cyclebuffer: np.ndarray, amplitudebuffer: np.ndarray,
                       phasebuffer: np.ndarray, cycleBartelsBuffer: np.ndarray,
                       use_cycle_strength: bool, filter_bartels: bool) -> pd.DataFrame:
    if number_of_cycles <= 0: return pd.DataFrame()
    ranks, periods, bartels_probs_str, amps_or_strengths, phases_deg = [], [], [], [], []
    amp_col_name = "Cycle Strength" if use_cycle_strength else "Amplitude"
    
    for i in range(number_of_cycles): 
        if i >= len(cyclebuffer) or i >= len(amplitudebuffer) or \
           i >= len(phasebuffer) or (filter_bartels and i >= len(cycleBartelsBuffer)): 
            continue 

        ranks.append(i + 1) 
        periods.append(cyclebuffer[i])
        amps_or_strengths.append(amplitudebuffer[i])
        phase_rad = phasebuffer[i]
        phase_deg = math.degrees(phase_rad % (2 * math.pi)) 
        phases_deg.append(phase_deg)
        
        if filter_bartels:
            bartels_idx = i 
            if bartels_idx < len(cycleBartelsBuffer): 
                if isinstance(cycleBartelsBuffer[bartels_idx], (int, float)) and cycleBartelsBuffer[bartels_idx] != 0:
                    bartels_probs_str.append(f"{cycleBartelsBuffer[bartels_idx]:.2f}%")
                elif cycleBartelsBuffer[bartels_idx] == 0 and number_of_cycles > 0 : 
                     bartels_probs_str.append(f"{0.0:.2f}%")
                else: 
                    bartels_probs_str.append("N/A")
            else: 
                bartels_probs_str.append("N/A")
        else: 
            bartels_probs_str.append("N/A")
            
    if not periods: return pd.DataFrame() 
    
    data_dict = {"Period": periods}
    if filter_bartels: 
        data_dict["Bartel"] = bartels_probs_str
    data_dict[amp_col_name] = amps_or_strengths
    data_dict["Phase (deg)"] = phases_deg
    
    df_cycles = pd.DataFrame(data_dict)
    df_cycles.index = pd.Index(ranks, name="Rank") 
    return df_cycles

class NpEncoder(json.JSONEncoder): 
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)