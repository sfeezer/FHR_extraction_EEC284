# This file calls eec284.py with many window size, overlap, and yule-walker order values
# performs a grid sweep on wnidow size and yw order to identify trends in parameter tuning
# preforms basic sweep on overlap param

import subprocess
import re
import csv
import os
import numpy as np

def run_experiment(params, data_path="data/PPG3.csv"):
    """
    Runs eec284.py with the provided parameter dictionary.
    """
    cmd = ["python", "eec284.py", "--data_path", data_path]
    for key, value in params.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Executing: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = result.stdout
        
        mae_match = re.search(r"Mean Absolute Error \(MAE\): ([\d\.]+)", stdout)
        rmse_match = re.search(r"Root Mean Squared Error \(RMSE\): ([\d\.]+)", stdout)
        
        mae = float(mae_match.group(1)) if mae_match else None
        rmse = float(rmse_match.group(1)) if rmse_match else None
        
        return mae, rmse
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None, None

def main(data_path="data/PPG3.csv"):
    
    # --- BASELINE ---
    default_params = {
        "window_size": 60,
        "window_step": 30,
        "yw_order": 100
    }
    print("Capturing Baseline Results...")
    base_mae, base_rmse = run_experiment(default_params, data_path)
    
    # --- SWEEP 1: OVERLAP ---
    # Overlap % = (size - step) / size * 100
    # Step = size * (1 - overlap/100)
    overlaps = [0, 20, 40, 60, 80, 95] # 100% is invalid (step=0), using 95% as limit
    overlap_file = "sweep_03_overlap.csv"
    
    print(f"\nStarting Overlap Sweep -> {overlap_file}")
    with open(overlap_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["overlap_pct", "window_size", "window_step", "MAE", "RMSE"])
        # Write Baseline
        writer.writerow(["50.0", 60, 30, base_mae, base_rmse])
        
        for ov in overlaps:
            # Skip if it matches baseline 50%
            if ov == 50: continue
            
            size = 60
            step = int(size * (1 - ov/100))
            if step < 1: step = 1 # Minimum step of 1 second
            
            actual_ov = (size - step) / size * 100
            mae, rmse = run_experiment({"window_size": size, "window_step": step}, data_path)
            if mae is not None:
                writer.writerow([f"{actual_ov:.1f}", size, step, mae, rmse])
                f.flush()

    # --- SWEEP 2: WINDOW SIZE & ORDER GRID SEARCH ---
    # 5 orders, 5 sizes
    sizes = [30, 45, 60, 75, 90]
    orders = [50, 75, 100, 125, 150]
    grid_file = "sweep_03_window_order.csv"
    
    print(f"\nStarting Grid Search (Size vs Order) -> {grid_file}")
    with open(grid_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["window_size", "yw_order", "MAE", "RMSE"])
        # Write Baseline (60, 100)
        writer.writerow([60, 100, base_mae, base_rmse])
        
        for s in sizes:
            for o in orders:
                # Skip baseline case
                if s == 60 and o == 100: continue
                
                # Keep step at 50% of window size for consistency
                step = s // 2
                mae, rmse = run_experiment({"window_size": s, "window_step": step, "yw_order": o}, data_path)
                if mae is not None:
                    writer.writerow([s, o, mae, rmse])
                    f.flush()

    print("\nAll Stage 3 Sweeps Complete.")

if __name__ == "__main__":
    main()
