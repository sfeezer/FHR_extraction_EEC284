# this file performs a set of six parameter-tuned tests across all datasets
#    in order to find a combination of parameters with strong all-around performance
import subprocess
import re
import csv
import os

def run_experiment():
    datasets = ["data/PPG1.csv", "data/PPG2.csv", "data/PPG3.csv"]

    # Define the test cases based on the robust analysis
    # Overlap percentages are converted to integer window_step values
    test_cases = [
        {
            "name": "Default",
            "args": []
        },
        {
            "name": "Set 1",
            "args": [
                "--f_low", "1.5", "--f_high", "15.0", 
                "--rls_lambda", "0.999", "--rls_delta", "0.01", "--rls_taps", "300", 
                "--window_size", "30", "--window_step", "6", "--yw_order", "150"
            ]
        },
        {
            "name": "Set 2",
            "args": [
                "--f_low", "1.5", "--f_high", "20.0", 
                "--rls_lambda", "1.0", "--rls_delta", "100.0", "--rls_taps", "300", 
                "--window_size", "30", "--window_step", "15", "--yw_order", "150",
                "--w2", "0.01", "--w3", "0.82", "--w4", "0.07", "--w5", "0.10"
            ]
        },
        {
            "name": "Set 3",
            "args": [
                "--f_low", "1.0", "--f_high", "12.0", 
                "--rls_lambda", "0.9999", "--rls_delta", "0.1", "--rls_taps", "200", 
                "--window_size", "45", "--window_step", "2", "--yw_order", "125",
                "--w2", "0.10", "--w3", "0.40", "--w4", "0.20", "--w5", "0.30"
            ]
        },
        {
            "name": "Set 4",
            "args": [
                "--f_low", "0.4", "--f_high", "8.0", 
                "--rls_lambda", "0.995", "--rls_delta", "10.0", "--rls_taps", "150", 
                "--window_size", "60", "--window_step", "30", "--yw_order", "100",
                "--w2", "0.15", "--w3", "0.35", "--w4", "0.25", "--w5", "0.25"
            ]
        },
        {
            "name": "Set 5",
            "args": [
                "--f_low", "1.5", "--f_high", "25.0", 
                "--rls_lambda", "0.999", "--rls_delta", "0.01", "--rls_taps", "300", 
                "--window_size", "30", "--window_step", "7", "--yw_order", "150",
                "--w2", "0.05", "--w3", "0.50", "--w4", "0.40", "--w5", "0.05"
            ]
        }
    ]

    results = []

    for dataset in datasets:
        print(f"\n>>> PROCESSING DATASET: {dataset}")
        for case in test_cases:
            print(f"--- Running Case: {case['name']} ---")
            
            # Construct the command
            cmd = ["python", "eec284.py", "--data_path", dataset] + case['args']
            
            try:
                # Execute the pipeline and capture output
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                output = result.stdout
                
                # Regex to find metrics in the printed output of eec284.py
                mae_match = re.search(r"Mean Absolute Error \(MAE\): ([\d.]+) BPM", output)
                rmse_match = re.search(r"Root Mean Squared Error \(RMSE\): ([\d.]+) BPM", output)
                
                if mae_match and rmse_match:
                    mae = float(mae_match.group(1))
                    rmse = float(rmse_match.group(1))
                    print(f"    SUCCESS | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
                    
                    results.append({
                        "Dataset": dataset,
                        "Set Name": case['name'],
                        "Parameters": " ".join(case['args']) if case['args'] else "Default",
                        "MAE": mae,
                        "RMSE": rmse
                    })
                else:
                    print(f"    WARNING: Could not parse results from output.")
                    print(f"    STDOUT Snippet: {output[-200:]}")
            
            except subprocess.CalledProcessError as e:
                print(f"    ERROR executing case {case['name']}:")
                print(e.stderr)

    # Save all results to a master CSV for comparison
    output_file = "all_around_results.csv"
    with open(output_file, mode='w', newline='') as f:
        fieldnames = ["Dataset", "Set Name", "Parameters", "MAE", "RMSE"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n" + "="*50)
    print(f"All tests complete. Summary saved to: {output_file}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
