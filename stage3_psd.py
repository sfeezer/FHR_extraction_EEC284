import numpy as np
import pandas as pd
from scipy.linalg import toeplitz, solve
from scipy.signal import freqz
from plotting import plot_stage3_psd_bpm

def yule_walker_psd(data, order, fs, nfft=2048):
    """
    Computes the Power Spectral Density using the Yule-Walker autoregressive method.
    """
    n = len(data)
    
    # Remove mean
    x = data - np.mean(data)
    
    # Biased autocorrelation for lags 0 to order
    r_full = np.correlate(x, x, mode='full')
    r = r_full[n-1 : n + order] / n
    
    # Solve Yule-Walker equations: R * a = -r_tail
    # R is Toeplitz matrix of r[0...order-1]
    # r_tail is r[1...order]
    R = toeplitz(r[:-1])
    rhs = -r[1:]
    
    try:
        a = solve(R, rhs)
    except np.linalg.LinAlgError:
        # If matrix is singular, return zeros
        return np.linspace(0, fs/2, nfft // 2 + 1), np.zeros(nfft // 2 + 1)
    
    # Prediction error variance
    sigma2 = r[0] + np.dot(a, r[1:])
    
    # Frequency response evaluation: H(z) = 1 / (1 + sum_{k=1}^p a_k z^{-k})
    A = np.concatenate(([1], a))
    w, h = freqz([1], A, worN=nfft // 2 + 1)
    
    # PSD = (sigma2 * |H(e^jw)|^2) / fs
    psd = (sigma2 * np.abs(h)**2) / fs
    freqs = w * fs / (2 * np.pi)
    
    return freqs, psd

def process_stage3(anc_df, window_size=60, window_step=30, fs=80, yw_order=100, graph=False, filtered_df=None):
    """
    Segments data and computes PSD for each window using Yule-Walker.
    """
    print(f"Stage 3: PSD Estimation (Yule-Walker, order {yw_order})")
    
    wavelengths = ['WL1', 'WL2']
    detectors = ['ch2', 'ch3', 'ch4', 'ch5']
    
    samples_per_window = int(window_size * fs)
    samples_per_step = int(window_step * fs)
    total_samples = len(anc_df)
    
    # Number of windows
    num_windows = (total_samples - samples_per_window) // samples_per_step + 1
    
    # Results structure: dict mapping window_index -> wavelength -> detector -> (freqs, psd)
    psd_results = {}
    
    for i in range(num_windows):
        start_idx = i * samples_per_step
        end_idx = start_idx + samples_per_window
        window_data = anc_df.iloc[start_idx:end_idx]
        
        window_psd = {}
        for wl in wavelengths:
            wl_psd = {}
            for det in detectors:
                col = f"{det}volts{wl}"
                if col in window_data.columns:
                    signal = window_data[col].values
                    freqs, psd = yule_walker_psd(signal, yw_order, fs)
                    wl_psd[det] = (freqs, psd)
            window_psd[wl] = wl_psd
        
        psd_results[i] = window_psd
        
        if (i + 1) % 10 == 0 or (i + 1) == num_windows:
            print(f"  Processed window {i+1}/{num_windows}")

    if graph and num_windows > 0:
        from plotting import plot_stage3_comparison
        # Plot the first window's PSD for a middle detector as validation
        # ch3voltsWL2 is often used for validation
        val_wl = 'WL2'
        val_det = 'ch3'
        col = f"{val_det}volts{val_wl}"
        
        if val_wl in psd_results[0] and val_det in psd_results[0][val_wl]:
            f_post, p_post = psd_results[0][val_wl][val_det]
            
            if filtered_df is not None and col in filtered_df.columns:
                # Compute PSD for Pre-ANC signal
                pre_signal = filtered_df[col].values[:samples_per_window]
                f_pre, p_pre = yule_walker_psd(pre_signal, yw_order, fs)
                
                plot_stage3_comparison(
                    f_pre, p_pre, 
                    f_post, p_post, 
                    title=f"Stage 3: PSD Comparison (Window 0, {val_det} {val_wl})"
                )
            else:
                plot_stage3_psd_bpm(f_post, p_post, title=f"Stage 3 Validation: PSD for Window 0 ({val_det} {val_wl})")

    return psd_results
