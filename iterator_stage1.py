import subprocess
import re
import csv
import os

def run_experiment(f_low, f_high, data_path):
    """
    Runs eec284.py with specific f_low and f_high and parses the output for metrics.
    """
    cmd = [
        "python", "eec284.py",
        "--data_path", data_path,
        "--f_low", str(f_low),
        "--f_high", str(f_high)
    ]
    # Ensure no graphs are popped up during batch run
    # Note: args.graph defaults to False in eec284.py now
    
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
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None, None

def main(data_path="data/PPG3.csv"):
    # Configuration
    f_low_range = [0.1, 0.2, 0.4, 1.0, 1.5]
    f_high_range = [5.0, 8.0, 12.0, 15.0, 20.0, 25.0]
    
    output_file = "sweep_01_bandpass.csv"
    
    print(f"Starting parameter sweep on {data_path}...")
    print(f"Results will be saved to {output_file}")
    
    results = []
    
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["f_low", "f_high", "MAE", "RMSE"])
        
        for f_low in f_low_range:
            for f_high in f_high_range:
                if f_low >= f_high:
                    continue
                
                mae, rmse = run_experiment(f_low, f_high, data_path)
                
                if mae is not None:
                    writer.writerow([f_low, f_high, mae, rmse])
                    f.flush() # Ensure it writes to disk immediately
                    print(f"SUCCESS: f_low={f_low}, f_high={f_high} -> MAE={mae:.4f}, RMSE={rmse:.4f}")
                    results.append((f_low, f_high, mae, rmse))
                else:
                    print(f"FAILURE: Could not extract metrics for f_low={f_low}, f_high={f_high}")

    if results:
        # Find best result based on MAE
        best_mae = min(results, key=lambda x: x[2])
        print("\n" + "="*40)
        print("SWEEP COMPLETE")
        print("="*40)
        print(f"Best MAE: {best_mae[2]:.4f} at f_low={best_mae[0]}, f_high={best_mae[1]}")
        print("="*40)

if __name__ == "__main__":
    main()
