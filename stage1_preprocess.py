# This helper file implements the stage 1 bandpass filter

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from plotting import plot_stage1_validation

# butterworth filter function
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# apply filter to data set
def apply_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# main management function for stage 1.
def preprocess_data(ppg_df, f_low, f_high, fs=80, graph=False):
    print(f"Stage 1: Preprocessing (Bandpass {f_low}Hz - {f_high}Hz)")
    
    # Identify signal columns (ch1voltsWL1 to ch5voltsWL2)
    signal_cols = [col for col in ppg_df.columns if 'ch' in col and 'volts' in col]
    
    if not signal_cols:
        print("Error: No signal columns found in PPG data. Check CSV headers.")
        return ppg_df

    # Create a copy for filtered data
    filtered_df = ppg_df.copy()
    
    # Apply filter to each signal column
    for col in signal_cols:
        filtered_df[col] = apply_filter(ppg_df[col].values, f_low, f_high, fs)
    
    # Graph before-after ch3wl2 if user toggled flag
    if graph:
        # Use ch3voltsWL2 as the representative channel for validation
        channel_to_plot = 'ch3voltsWL2'
        if channel_to_plot in signal_cols:
            plot_stage1_validation(
                ppg_df[channel_to_plot].values,
                filtered_df[channel_to_plot].values,
                fs,
                channel_to_plot
            )
        
    return filtered_df
