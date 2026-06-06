import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def run_analysis():
    # 1. Load the data
    try:
        df = pd.read_csv('sweep_03_overlap.csv')
    except FileNotFoundError:
        print("Error: sweep_03_overlap.csv not found.")
        return

    # Sort by overlap_pct to ensure line plots are ordered correctly
    df = df.sort_values('overlap_pct').reset_index(drop=True)

    # 2. Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    x_indices = np.arange(len(df))
    # Use a color gradient to represent increasing overlap
    colors = cm.get_cmap('plasma')(np.linspace(0.2, 0.8, len(df)))

    # --- MAE Subplot ---
    bars1 = ax1.bar(x_indices, df['MAE'], color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('MAE (BPM)', fontweight='bold')
    ax1.set_title('MAE across Window Overlap Percentages', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # --- RMSE Subplot ---
    bars2 = ax2.bar(x_indices, df['RMSE'], color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('RMSE (BPM)', fontweight='bold')
    ax2.set_title('RMSE across Window Overlap Percentages', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. Final Formatting
    labels = [f"{row['overlap_pct']}% \n(Step: {int(row['window_step'])}s)" for i, row in df.iterrows()]
    plt.xticks(x_indices, labels, fontweight='bold')
    ax2.set_xlabel('Overlap Percentage and Window Step Size', fontweight='bold')

    plt.tight_layout()
    
    # 5. Save and Show
    output_filename = 'analysis_03_overlap.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis visualization (Bar Style) saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    run_analysis()
