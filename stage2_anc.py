# This helper file implements recursive least square adaptive noise cancelling using ch1wl1 ch2wl2 as reference
import numpy as np
import pandas as pd
from plotting import plot_stage2_spectral_validation

# RLS filter function
def rls_filter(reference, mixed, taps, lam=0.9999, delta=1):
    # data strcture setup/prep
    N = len(mixed)
    w = np.zeros(taps)
    P = np.eye(taps) / delta
    e = np.zeros(N)
    eps = 1e-6
    _warned = False

    x_padded = np.concatenate((np.zeros(taps - 1), reference))

    for n in range(N):
        # examine window
        x_n = x_padded[n : n + taps][::-1]
        
        Px    = P @ x_n
        denom = max(lam + x_n @ Px, eps)
        k     = Px / denom

        # subtract maternal signal from mixed signal
        e[n]  = mixed[n] - w @ x_n

        # update weights based on last e[n]
        w_next = w + k * e[n]
        P_next = (P - np.outer(k, Px)) / lam
        P_next = (P_next + P_next.T) / 2

        # if weights diverge to NaN or inf, reset and continue
        if np.any(np.isnan(w_next)) or np.any(np.isinf(w_next)) or np.any(np.abs(w_next) > 1e6):
            if not _warned:
                print(f"    Warning: RLS weights diverged at sample {n}. Resetting state.")
                _warned = True
            # Reset state; pass through raw sample rather than garbage
            w[:] = 0.0
            P[:] = np.eye(taps) / delta
            e[n] = mixed[n]
        else:
            w = w_next
            P = P_next

    return e

# main management function for stage 2.
def execute_anc(filtered_df, taps=100, lam=0.9999, delta=1.0, fs=80, graph=False):
    print(f"Stage 2: Adaptive Noise Cancellation (RLS, {taps} taps, lambda={lam}, delta={delta})")

    # Create a copy for ANC data
    anc_df = filtered_df.copy()

    wavelengths = ['WL1', 'WL2']
    detectors   = ['ch2', 'ch3', 'ch4', 'ch5']
    ref_prefix  = 'ch1'

    # Store reference signal and name for plotting if graph is True
    val_ref_signal = None
    val_ref_name = None
    val_col = 'ch3voltsWL2'
    val_ref_name = 'ch1voltsWL2'

    # loop through ch/wl combinations
    for wl in wavelengths:
        ref_col = f"{ref_prefix}volts{wl}"
        if ref_col not in filtered_df.columns:
            print(f"  Warning: reference column {ref_col} not found, skipping {wl}.")
            continue

        ref_signal = filtered_df[ref_col].values

        for det in detectors:
            mixed_col = f"{det}volts{wl}"
            if mixed_col not in filtered_df.columns:
                print(f"  Warning: {mixed_col} not found, skipping.")
                continue
            
            # Capture the reference signal and name used for our specific validation channel
            if mixed_col == val_col:
                val_ref_signal = ref_signal
                val_ref_name = ref_col

            print(f"  Processing {mixed_col} using {ref_col} as reference...")

            mixed_signal = filtered_df[mixed_col].values
            # call the ANC RLS
            cleaned_signal = rls_filter(ref_signal, mixed_signal, taps, lam=lam, delta=delta)

            var_mixed   = np.var(mixed_signal)
            var_cleaned = np.var(cleaned_signal)
            reduction   = (1 - var_cleaned / var_mixed) * 100
            print(f"    Variance reduction: {reduction:.2f}%")

            anc_df[mixed_col] = cleaned_signal
    # graph before and after spectrograph for ch3wl2 if user toggled flag.
    if graph:
        if val_col in filtered_df.columns and val_ref_signal is not None:
            plot_stage2_spectral_validation(
                filtered_df[val_col].values,
                anc_df[val_col].values,
                val_ref_signal,
                fs,
                val_col,
                val_ref_name
            )

    return anc_df
