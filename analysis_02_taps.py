import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def run_analysis():
    # 1. Load the data
    try:
        df = pd.read_csv('sweep_02_taps.csv')
    except FileNotFoundError:
        print("Error: sweep_02_taps.csv not found.")
        return

    # 2. Prepare labels
    df['label'] = df['taps'].astype(str) + " Taps"
    
    # 3. Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    x_indices = np.arange(len(df))
    # Use a colorful gradient colormap
    colors = cm.get_cmap('plasma')(np.linspace(0.1, 0.8, len(df)))

    # --- MAE Subplot ---
    bars1 = ax1.bar(x_indices, df['MAE'], color=colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('MAE (BPM)', fontweight='bold')
    ax1.set_title('MAE across RLS Filter Tap Counts', fontsize=16, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # --- RMSE Subplot ---
    bars2 = ax2.bar(x_indices, df['RMSE'], color=colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('RMSE (BPM)', fontweight='bold')
    ax2.set_title('RMSE across RLS Filter Tap Counts', fontsize=16, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Final Formatting
    plt.xticks(x_indices, df['label'], fontweight='bold', fontsize=11)
    ax2.set_xlabel('Filter Length (Number of Taps)', fontweight='bold', fontsize=12)

    plt.tight_layout()
    
    # 5. Save and Show
    output_filename = 'analysis_02_taps.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Enhanced analysis visualization (Taps) saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    run_analysis()
