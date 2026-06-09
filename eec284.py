# Created by Shawn Feezer and Anusheel Nand for EEC 284
# Fetal Heart Rate Extraction Tool

import argparse
import pandas as pd
import numpy as np
import os
import sys

# Import stages
from stage1_preprocess import preprocess_data
from stage2_anc import execute_anc
from stage3_psd import process_stage3
from stage5_fusion import fuse_fhr_tracks, calculate_metrics
from plotting import plot_psd, plot_spectrogram, plot_stage5_validation

# .csv loading tool
def load_data(ppg_path, fhr_path):
    """
    Loads PPG data (80Hz) and FHR ground truth (1Hz).
    """
    print(f"Loading PPG data from: {ppg_path}")
    # PPG has a header
    ppg_df = pd.read_csv(ppg_path)
    
    print(f"Loading FHR ground truth from: {fhr_path}")
    # FHR does not have a header
    fhr_df = pd.read_csv(fhr_path, header=None)
    
    return ppg_df, fhr_df


def main():
    # arg parser and default values
    parser = argparse.ArgumentParser(description="EEC284 FHR Extraction Pipeline")
    
    # Required Arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to PPG CSV file")
    
    parser.add_argument("--f_low", type=float, default=0.2, help="Lower corner frequency for Stage 1 (Hz)")
    parser.add_argument("--f_high", type=float, default=15.0, help="Upper corner frequency for Stage 1 (Hz)")
    parser.add_argument("--rls_taps", type=int, default=100, help="Filter length for Stage 2 RLS")
    parser.add_argument("--rls_lambda", type=float, default=0.9999, help="Forgetting factor for Stage 2 RLS")
    parser.add_argument("--rls_delta", type=float, default=1.0, help="Initial regularization for Stage 2 RLS")
    parser.add_argument("--window_size", type=int, default=60, help="Window size in seconds")
    parser.add_argument("--window_step", type=int, default=30, help="Window step in seconds")
    parser.add_argument("--yw_order", type=int, default=100, help="Yule-Walker model order")
    parser.add_argument("--bpm_min", type=int, default=110, help="Min FHR BPM")
    parser.add_argument("--bpm_max", type=int, default=270, help="Max FHR BPM")
    parser.add_argument("--s4_file", type=str, default="base", choices=["base", "alt"], help="Stage 4 implementation to use")
    parser.add_argument("--s4_helper", action="store_true", help="Enable Stage 4 helper (boundary rejection and stateful tracking)")
    parser.add_argument("--mad_thresh", type=float, default=3.0, help="MAD threshold for Stage 5")
    parser.add_argument("--graph", action="store_true", help="Enable visualization plots")
    parser.add_argument("--w2", type=float, default=1.0, help="Weight for Detector 2")
    parser.add_argument("--w3", type=float, default=3.0, help="Weight for Detector 3")
    parser.add_argument("--w4", type=float, default=2.0, help="Weight for Detector 4")
    parser.add_argument("--w5", type=float, default=2.0, help="Weight for Detector 5")

    args = parser.parse_args()

    # construct fhr*.csv path from ppg*.csv
    base_dir = os.path.dirname(args.data_path)
    base_name = os.path.basename(args.data_path)
    fhr_name = base_name.replace("PPG", "FHR")
    fhr_path = os.path.join(base_dir, fhr_name)

    if not os.path.exists(fhr_path):
        print(f"Warning: FHR ground truth not found at {fhr_path}")
        fhr_df = None
    else:
        ppg_df, fhr_df = load_data(args.data_path, fhr_path)
        
        # Call Stage 1, bandpass filter
        filtered_df = preprocess_data(
            ppg_df, 
            args.f_low, 
            args.f_high, 
            fs=80, 
            graph=False # Set to true for validation graph
        )
        
        print("Stage 1 Complete.")

        # Call Stage 2, RLS ANC
        anc_df = execute_anc(
            filtered_df, 
            args.rls_taps, 
            lam=args.rls_lambda,
            delta=args.rls_delta,
            fs=80, 
            graph=False # Set to true for validation graph
        )

        print("Stage 2 Complete.")

        # Call stage 3, PSD estimation
        psd_results = process_stage3(
            anc_df, 
            args.window_size, 
            args.window_step, 
            fs=80, 
            yw_order=args.yw_order,
            graph=False, # Set to true for validation graph
            filtered_df=filtered_df
        )


        print("Stage 3 Complete.")

        # Call Stage 4: peak selection
        if args.s4_file == "alt":
            from stage4_peaks_alt import extract_fhr_tracks
        else:
            from stage4_peaks import extract_fhr_tracks

        fhr_tracks_df = extract_fhr_tracks(
            psd_results, 
            args.bpm_min, 
            args.bpm_max,
            helper_mode=args.s4_helper
        )
        
        print("\nStage 4 Validation: First 5 minutes of FHR tracks (BPM)")
        print(fhr_tracks_df.head(10))

        print("\nStage 4 Complete.")

        # Stage 5 begin.
        # set weights 
        custom_weights = {
            'ch2': args.w2,
            'ch3': args.w3,
            'ch4': args.w4,
            'ch5': args.w5
        }
        # call Stage 5, sensor fusion
        fused_fhr = fuse_fhr_tracks(fhr_tracks_df, args.mad_thresh, custom_weights=custom_weights)
        
        # Error calculations
        if fhr_df is not None:
            # load ground truth
            ref_fhr_1hz = fhr_df.iloc[:, 0].values
            
            num_windows = len(fhr_tracks_df)
            windowed_ref = []
            
            # align 1Hz ref with 30s window data
            for i in range(num_windows):
                start_sec = i * args.window_step
                end_sec = start_sec + args.window_size
                window_ref_segment = ref_fhr_1hz[start_sec : min(end_sec, len(ref_fhr_1hz))]
                if len(window_ref_segment) > 0:
                    windowed_ref.append(np.mean(window_ref_segment))
                else:
                    windowed_ref.append(np.nan)
            
            windowed_ref = np.array(windowed_ref)
            
            # Call error calculation
            mae, rmse = calculate_metrics(fused_fhr.values, windowed_ref)
            
            print("\n" + "="*40)
            print("FINAL VALIDATION RESULTS")
            print("="*40)
            print(f"Mean Absolute Error (MAE): {mae:.4f} BPM")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f} BPM")
            print("="*40)
            
            # Plot if --graph True set in args.
            if args.graph:
                plot_stage5_validation(
                    fhr_tracks_df, 
                    fused_fhr, 
                    ref_fhr_1hz, 
                    args.window_step, 
                    args.window_size
                )

        
        print("\nPipeline Execution Finished.")

if __name__ == "__main__":
    main()
