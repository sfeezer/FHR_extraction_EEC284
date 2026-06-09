# this master sweep file calls all five iterator_*.py sweep experiment files in succession
# useful for applying full suite to a single PPG set.

import time
import iterator_stage1
import iterator_stage2
import iterator_stage3
import iterator_stage5

import argparse

def main():
    parser = argparse.ArgumentParser(description="Master Parameter Sweep Orchestrator")
    parser.add_argument("--data_path", type=str, default="data/PPG3.csv", help="Path to the source CSV data")
    args = parser.parse_args()

    start_time = time.time()
    print("="*60)
    print(f"STARTING MASTER PARAMETER SWEEP ON: {args.data_path}")
    print("="*60)
    
    stages = [
        ("STAGE 1: Bandpass Filter", lambda: iterator_stage1.main(args.data_path)),
        ("STAGE 2: RLS ANC Parameters", lambda: iterator_stage2.main(args.data_path)),
        ("STAGE 3: Window & AR Model", lambda: iterator_stage3.main(args.data_path)),
        ("STAGE 5: Fusion Weights", lambda: iterator_stage5.main(args.data_path))
    ]
    
    for stage_name, stage_func in stages:
        stage_start = time.time()
        print(f"\n>>> Executing {stage_name}...")
        try:
            stage_func()
            elapsed = time.time() - stage_start
            print(f">>> {stage_name} completed in {elapsed:.2f} seconds.")
        except Exception as e:
            print(f">>> ERROR in {stage_name}: {e}")
            # Continue to next stage even if one fails
            continue
            
    total_elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"MASTER SWEEP COMPLETE in {total_elapsed/60:.2f} minutes")
    print("="*60)

if __name__ == "__main__":
    main()
