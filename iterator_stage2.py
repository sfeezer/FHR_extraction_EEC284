# This file calls eec284.py with many taps, lambda and delta values
# performs a grid sweep on lambda/delta to identify trends in parameter tuning
# performs basic sweep on tap count

import subprocess
import re
import csv
import os
import numpy as np

def run_experiment(data_path, taps=100, lam=0.9999, delta=1.0):
    cmd = [
        "python", "eec284.py",
        "--data_path", data_path,
        "--rls_taps", str(taps),
        "--rls_lambda", str(lam),
        "--rls_delta", str(delta)
    ]
    
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
        print(f"Error running experiment: {e}")
        return None, None

def sweep_taps(data_path):
    taps_range = [25, 50, 100, 150, 200, 300]
    output_file = "sweep_02_taps.csv"
    
    print(f"\n--- Starting Tap Count Sweep ---")
    results = []
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["taps", "MAE", "RMSE"])
        
        for taps in taps_range:
            mae, rmse = run_experiment(data_path, taps=taps)
            if mae is not None:
                writer.writerow([taps, mae, rmse])
                f.flush()
                print(f"SUCCESS: taps={taps} -> MAE={mae:.4f}, RMSE={rmse:.4f}")
                results.append((taps, mae, rmse))

    if results:
        best = min(results, key=lambda x: x[1])
        print(f"Best Taps: {best[0]} (MAE={best[1]:.4f})")

def sweep_params(data_path):
    lambda_range = [0.99, 0.995, 0.999, 0.9999, 1.0]
    delta_range = [0.01, 0.1, 1.0, 10.0, 100.0]
    output_file = "sweep_02_params.csv"
    
    print(f"\n--- Starting Lambda/Delta Grid Sweep ---")
    results = []
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "delta", "MAE", "RMSE"])
        
        for lam in lambda_range:
            for delta in delta_range:
                mae, rmse = run_experiment(data_path, lam=lam, delta=delta)
                if mae is not None:
                    writer.writerow([lam, delta, mae, rmse])
                    f.flush()
                    print(f"SUCCESS: lambda={lam}, delta={delta} -> MAE={mae:.4f}, RMSE={rmse:.4f}")
                    results.append((lam, delta, mae, rmse))

    if results:
        best = min(results, key=lambda x: x[2])
        print(f"Best Params: lambda={best[0]}, delta={best[1]} (MAE={best[2]:.4f})")

def main(data_path="data/PPG3.csv"):
    sweep_taps(data_path)
    sweep_params(data_path)

if __name__ == "__main__":
    main()
