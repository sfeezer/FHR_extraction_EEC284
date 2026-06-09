# this file implements peak selection via Peak Prominence. 
# all other non-prominence comments removed

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def extract_fhr_tracks(psd_results, bpm_min=110, bpm_max=270, helper_mode=False):
    print(f"Stage 4: Peak Picking (BPM Range: {bpm_min} - {bpm_max}, Helper: {helper_mode})")
    
    sensor_state = {}
    all_window_fhr = []
    
    num_windows = len(psd_results)
    
    for i in range(num_windows):
        window_fhr = {}
        window_psd_data = psd_results[i]
        
        for wl in window_psd_data:
            for det in window_psd_data[wl]:
                col_name = f"{det}_{wl}"
                if col_name not in sensor_state:
                    sensor_state[col_name] = {'seen': False, 'last': 140.0}

                freqs, psd = window_psd_data[wl][det]
                peak_bpm = np.nan
                
                bpms = freqs * 60
                mask = (bpms >= bpm_min) & (bpms <= bpm_max)
                
                found_valid = False
                if np.any(mask):
                    valid_bpms = bpms[mask]
                    valid_psd = psd[mask]
                    
                    peaks, properties = find_peaks(valid_psd, prominence=valid_psd.max() * 0.10)
                    if peaks.size > 0:
                        ## Capture peaks via prominence instead of raw argmax.
                        best_peak_idx_in_peaks = np.argmax(properties["prominences"])
                        peak_idx = peaks[best_peak_idx_in_peaks]
                        peak_bpm = valid_bpms[peak_idx]
                        
                        if peak_idx == 0 or peak_idx == len(valid_psd) - 1:
                            found_valid = False
                        else:
                            found_valid = True
                
                if helper_mode:
                    if found_valid:
                        sensor_state[col_name]['seen'] = True
                        sensor_state[col_name]['last'] = peak_bpm
                        window_fhr[col_name] = peak_bpm
                    else:

                        window_fhr[col_name] = sensor_state[col_name]['last']
                else:
                    window_fhr[col_name] = peak_bpm if found_valid else np.nan
        
        all_window_fhr.append(window_fhr)
        
        if (i + 1) % 20 == 0 or (i + 1) == num_windows:
            print(f"  Extracted peaks for window {i+1}/{num_windows}")

    fhr_tracks_df = pd.DataFrame(all_window_fhr)
    
    return fhr_tracks_df
