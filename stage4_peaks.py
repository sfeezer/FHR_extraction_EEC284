import numpy as np
import pandas as pd

def extract_fhr_tracks(psd_results, bpm_min=110, bpm_max=270):
    """
    Extracts the peak Fetal Heart Rate (FHR) from the PSD results for each window and detector.
    
    Args:
        psd_results: Dict mapping window_index -> wavelength -> detector -> (freqs, psd)
        bpm_min: Minimum physiological BPM for FHR.
        bpm_max: Maximum physiological BPM for FHR.
        
    Returns:
        pd.DataFrame: A DataFrame with windows as rows and {detector}_{wavelength} as columns.
    """
    print(f"Stage 4: Peak Picking (BPM Range: {bpm_min} - {bpm_max})")
    
    # Initialize results list
    all_window_fhr = []
    
    num_windows = len(psd_results)
    
    for i in range(num_windows):
        window_fhr = {}
        window_psd_data = psd_results[i]
        
        for wl in window_psd_data:
            for det in window_psd_data[wl]:
                freqs, psd = window_psd_data[wl][det]
                
                # Convert frequencies to BPM
                bpms = freqs * 60
                
                # Create mask for physiological BPM range
                mask = (bpms >= bpm_min) & (bpms <= bpm_max)
                
                if np.any(mask):
                    # Filter BPMs and PSDs using the mask
                    valid_bpms = bpms[mask]
                    valid_psd = psd[mask]
                    
                    # Find index of maximum power density
                    peak_idx = np.argmax(valid_psd)
                    
                    # Extract peak BPM
                    peak_bpm = valid_bpms[peak_idx]
                else:
                    peak_bpm = np.nan
                
                # Use a consistent column naming scheme
                col_name = f"{det}_{wl}"
                window_fhr[col_name] = peak_bpm
        
        all_window_fhr.append(window_fhr)
        
        if (i + 1) % 20 == 0 or (i + 1) == num_windows:
             print(f"  Extracted peaks for window {i+1}/{num_windows}")

    # Convert to DataFrame
    fhr_tracks_df = pd.DataFrame(all_window_fhr)
    
    return fhr_tracks_df
