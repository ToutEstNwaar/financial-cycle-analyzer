# utils/plotting.py

"""
Utility functions for plotting, creating data tables, and saving indicator results.
"""

import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import mplfinance as mpf
import datetime
import json

# Import color settings from the indicator_logic package
try:
    # Assuming 'indicator_logic' is a sibling package to 'utils'
    # and the project root (containing both) is in sys.path
    from indicator_logic import settings as ind_settings
except ImportError:
    print("Warning: Could not import 'indicator_logic.settings' in 'utils.plotting'. Using fallback local color defaults.")
    class FallbackSettings: # Define a fallback if import fails
        PAST_COLORS = {'up': '#2DD204', 'down': '#D2042D'}
        FUTURE_COLORS = {'up': 'fuchsia', 'down': 'yellow'}
        FUTURE_FLIP_COLORS = {'up': 'fuchsia', 'down': 'yellow'}
        # Add other settings if they were directly used from ind_settings here,
        # though it's better if streamlit_app.py passes all necessary values.
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
    # Use provided color dicts or fall back to defaults
    past_colors_to_use = past_colors_dict if past_colors_dict is not None else DEFAULT_PAST_COLORS
    future_colors_to_use = future_colors_dict if future_colors_dict is not None else DEFAULT_FUTURE_COLORS
    future_flip_colors_to_use = future_flip_colors_dict if future_flip_colors_dict is not None else DEFAULT_FUTURE_FLIP_COLORS

    if not isinstance(ohlc_data.index, pd.DatetimeIndex):
        print("Warning: ohlc_data.index is not a DatetimeIndex. Plotting may be affected.")

    if len(past_wave) < 2 or len(future_wave) < 2:
        # print("Warning: Wave arrays too short for plotting comparisons. Min length 2 required.")
        if not ohlc_data.empty:
            fig, _ = mpf.plot(ohlc_data, type='candle', style='yahoo', title=title + " (Wave data too short)", returnfig=True, figsize=(15,7))
            return fig
        return None

    plot_start_idx = max(0, calc_bar_index - window_past - 20)
    plot_end_idx = min(len(ohlc_data), calc_bar_index + window_future + 20)
    plot_data = ohlc_data.iloc[plot_start_idx:plot_end_idx]

    if plot_data.empty:
        print("Warning: No data to plot after slicing for plot range.")
        return None

    fig, axlist = mpf.plot(plot_data, type='candle', style='yahoo', title=title, volume=False, returnfig=True, figsize=(15, 7))
    ax = axlist[0] if isinstance(axlist, list) else axlist

    numerical_calc_bar_index_in_plot = -1
    try:
        calc_bar_date = ohlc_data.index[calc_bar_index]
        plotted_dates_list = plot_data.index.tolist()
        if calc_bar_date in plotted_dates_list:
            numerical_calc_bar_index_in_plot = plotted_dates_list.index(calc_bar_date)
        else:
            # print(f"Warning: Calculation bar date {calc_bar_date} not found in the plotted subset. Lines might be off.")
            # Try to find the closest date or use a default if critical
            pass # Will proceed with -1 if not found, line plotting will likely fail or be wrong.
                 # A more robust solution would be to not plot lines if this fails.
    except (IndexError, ValueError) as e:
        print(f"Error mapping calc_bar_index to plotted data: {e}")
        pass # Allow plot to render without lines if mapping fails

    # Plot Past Lines only if mapping was successful
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

    # Plot Future Lines only if mapping was successful and past direction is known
    if numerical_calc_bar_index_in_plot != -1 and len(future_wave) >= 2 and last_past_direction is not None:
        prev_future_color = None
        first_future_direction = 'up' if future_wave[0] > future_wave[1] else 'down'
        
        # Color for the first future line
        color_for_first_future_line = future_colors_to_use.get(first_future_direction, 'orange')
        if first_future_direction != last_past_direction:
            color_for_first_future_line = future_flip_colors_to_use.get(first_future_direction, 'purple')
        
        # Plot first future line (at numerical_calc_bar_index_in_plot + 1)
        ax.axvline(x=numerical_calc_bar_index_in_plot + 1, color=color_for_first_future_line, linestyle='--', linewidth=1)
        prev_future_color = color_for_first_future_line
            
        for i in range(1, len(future_wave) - 1):
            line_idx = numerical_calc_bar_index_in_plot + i + 1
            direction = 'up' if future_wave[i] > future_wave[i+1] else 'down'
            current_color = future_colors_to_use.get(direction, 'orange')
            if prev_future_color != current_color: # Only plot if color changes
                ax.axvline(x=line_idx, color=current_color, linestyle='--', linewidth=1)
            prev_future_color = current_color
    return fig

def create_cycle_table(number_of_cycles: int, cyclebuffer: np.ndarray, amplitudebuffer: np.ndarray,
                       phasebuffer: np.ndarray, cycleBartelsBuffer: np.ndarray,
                       use_cycle_strength: bool, filter_bartels: bool) -> pd.DataFrame:
    if number_of_cycles <= 0: return pd.DataFrame()
    ranks, periods, bartels_probs_str, amps_or_strengths, phases_deg = [], [], [], [], []
    amp_col_name = "Cycle Strength" if use_cycle_strength else "Amplitude"
    for i in range(1, number_of_cycles + 1):
        if i >= len(cyclebuffer) or i >= len(amplitudebuffer) or i >= len(phasebuffer): continue
        ranks.append(i)
        periods.append(cyclebuffer[i])
        amps_or_strengths.append(amplitudebuffer[i])
        phase_rad = phasebuffer[i]
        phase_deg = math.degrees(phase_rad % (2 * math.pi))
        phases_deg.append(phase_deg)
        if filter_bartels:
            bartels_idx = i - 1
            if bartels_idx < len(cycleBartelsBuffer): bartels_probs_str.append(f"{cycleBartelsBuffer[bartels_idx]:.2f}%")
            else: bartels_probs_str.append("N/A")
        else: bartels_probs_str.append("N/A")
    if not periods: return pd.DataFrame()
    df_cycles = pd.DataFrame({"Period": periods, "Bartel": bartels_probs_str, amp_col_name: amps_or_strengths, "Phase (deg)": phases_deg})
    df_cycles.index = pd.Index(ranks, name="Rank")
    return df_cycles

# save_indicator_data can remain as previously provided, as streamlit_app.py now handles JSON creation for download directly.
# If you want save_indicator_data to be used by the app, it would need to accept the data for the JSON string rather than a filename.
# For now, it's a utility that *could* save to server disk if called.

class NpEncoder(json.JSONEncoder): # Kept for completeness if save_indicator_data is used elsewhere
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

# (if __name__ == '__main__' block for testing utils/plotting.py can remain as before)