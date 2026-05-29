import argparse
import pandas as pd
import numpy as np
import os
import sys

# Import stages
from stage1_preprocess import preprocess_data
from stage2_anc import execute_anc
from stage3_psd import process_stage3
from stage4_peaks import extract_fhr_tracks
from stage5_fusion import fuse_fhr_tracks, calculate_metrics
from plotting import plot_psd, plot_spectrogram, plot_stage5_validation

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
    parser = argparse.ArgumentParser(description="EEC284 FHR Extraction Pipeline")
    
    # Required Arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to PPG CSV file")
    
    # Optional Arguments with defaults from GEMINI.md
    parser.add_argument("--f_low", type=float, default=0.2, help="Lower corner frequency for Stage 1 (Hz)")
    parser.add_argument("--f_high", type=float, default=15.0, help="Upper corner frequency for Stage 1 (Hz)")
    parser.add_argument("--rls_taps", type=int, default=100, help="Filter length for Stage 2 RLS")
    parser.add_argument("--window_size", type=int, default=60, help="Window size in seconds")
    parser.add_argument("--window_step", type=int, default=30, help="Window step in seconds")
    parser.add_argument("--yw_order", type=int, default=100, help="Yule-Walker model order")
    parser.add_argument("--bpm_min", type=int, default=110, help="Min FHR BPM")
    parser.add_argument("--bpm_max", type=int, default=270, help="Max FHR BPM")
    parser.add_argument("--mad_thresh", type=float, default=3.0, help="MAD threshold for Stage 5")

    args = parser.parse_args()

    # Determine FHR path based on PPG path (assuming standard naming convention)
    # e.g., data/PPG1.csv -> data/FHR1.csv
    base_dir = os.path.dirname(args.data_path)
    base_name = os.path.basename(args.data_path)
    fhr_name = base_name.replace("PPG", "FHR")
    fhr_path = os.path.join(base_dir, fhr_name)

    if not os.path.exists(fhr_path):
        print(f"Warning: FHR ground truth not found at {fhr_path}")
        fhr_df = None
    else:
        ppg_df, fhr_df = load_data(args.data_path, fhr_path)
        
        # --- STAGE 1: PREPROCESSING ---
        # To enable validation graphing, manually set graph=True below.
        filtered_df = preprocess_data(
            ppg_df, 
            args.f_low, 
            args.f_high, 
            fs=80, 
            graph=False
        )
        
        print("Stage 1 Complete.")

        # --- STAGE 2: ADAPTIVE NOISE CANCELLATION ---
        # To enable validation graphing, manually set graph=True below.
        anc_df = execute_anc(
            filtered_df, 
            args.rls_taps, 
            fs=80, 
            graph=False
        )

        print("Stage 2 Complete.")

        # --- STAGE 3: PSD ESTIMATION ---
        # To enable validation graphing, manually set graph=True below.
        psd_results = process_stage3(
            anc_df, 
            args.window_size, 
            args.window_step, 
            fs=80, 
            yw_order=args.yw_order,
            graph=False,
            filtered_df=filtered_df
        )

        print("Stage 3 Complete.")

        # --- STAGE 4: PEAK PICKING ---
        fhr_tracks_df = extract_fhr_tracks(
            psd_results, 
            args.bpm_min, 
            args.bpm_max
        )
        
        print("\nStage 4 Validation: First 5 minutes of FHR tracks (BPM)")
        print(fhr_tracks_df.head(10))

        print("\nStage 4 Complete.")

        # --- STAGE 5: SENSOR FUSION ---
        fused_fhr = fuse_fhr_tracks(fhr_tracks_df, args.mad_thresh)
        
        # --- VALIDATION AND METRICS ---
        if fhr_df is not None:
            # fhr_df is 1Hz. Align windows.
            # Reference FHR should be windowed Mean of Ground Truth.
            ref_fhr_1hz = fhr_df.iloc[:, 0].values
            
            num_windows = len(fhr_tracks_df)
            windowed_ref = []
            
            for i in range(num_windows):
                start_sec = i * args.window_step
                end_sec = start_sec + args.window_size
                # Since 1Hz, indices are seconds
                window_ref_segment = ref_fhr_1hz[start_sec : min(end_sec, len(ref_fhr_1hz))]
                if len(window_ref_segment) > 0:
                    windowed_ref.append(np.mean(window_ref_segment))
                else:
                    windowed_ref.append(np.nan)
            
            windowed_ref = np.array(windowed_ref)
            
            mae, rmse = calculate_metrics(fused_fhr.values, windowed_ref)
            
            print("\n" + "="*40)
            print("FINAL VALIDATION RESULTS")
            print("="*40)
            print(f"Mean Absolute Error (MAE): {mae:.4f} BPM")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f} BPM")
            print("="*40)
            
            # Plot validation
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
