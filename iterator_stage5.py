import subprocess
import re
import csv
import numpy as np
import os

def run_experiment(w2, w3, w4, w5, data_path):
    """
    Runs eec284.py with specific weights and parses the output for metrics.
    """
    cmd = [
        "python", "eec284.py",
        "--data_path", data_path,
        "--w2", f"{w2:.4f}",
        "--w3", f"{w3:.4f}",
        "--w4", f"{w4:.4f}",
        "--w5", f"{w5:.4f}"
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

def main(data_path="data/PPG3.csv"):
    # Configuration
    num_iterations = 30
    output_file = "sweep_05_average.csv"
    
    file_exists = os.path.exists(output_file)
    mode = 'a' if file_exists else 'w'
    
    print(f"Starting Stage 5 weight sweep on {data_path}...")
    print(f"Generating {num_iterations} weight combinations...")
    
    # Generate random weights that sum to 1
    weights_list = np.random.dirichlet([1, 1, 1, 1], size=num_iterations)
    
    # If file exists, we only run the random trials. 
    # If new file, we include the original weights for baseline.
    if not file_exists:
        orig_weights = np.array([1, 3, 2, 2])
        orig_norm = orig_weights / np.sum(orig_weights)
        all_weights = np.vstack([orig_norm, weights_list])
    else:
        all_weights = weights_list
    
    results = []
    
    with open(output_file, mode=mode, newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["w2", "w3", "w4", "w5", "MAE", "RMSE"])
        
        for i, (w2, w3, w4, w5) in enumerate(all_weights):
            label = "Baseline" if (not file_exists and i == 0) else f"Trial {i}"
            print(f"\nIteration {i} ({label}):")
            
            mae, rmse = run_experiment(w2, w3, w4, w5, data_path)
            
            if mae is not None:
                writer.writerow([f"{w2:.4f}", f"{w3:.4f}", f"{w4:.4f}", f"{w5:.4f}", mae, rmse])
                f.flush()
                print(f"SUCCESS: MAE={mae:.4f}, RMSE={rmse:.4f}")
                results.append((w2, w3, w4, w5, mae, rmse))
            else:
                print(f"FAILURE: Could not extract metrics.")

    if results:
        best_mae = min(results, key=lambda x: x[4])
        print("\n" + "="*40)
        print("STAGE 5 SWEEP COMPLETE")
        print("="*40)
        print(f"Best MAE: {best_mae[4]:.4f}")
        print(f"Weights: w2={best_mae[0]:.4f}, w3={best_mae[1]:.4f}, w4={best_mae[2]:.4f}, w5={best_mae[3]:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
