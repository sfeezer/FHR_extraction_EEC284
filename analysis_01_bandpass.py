# this file analyzes and graphs the results of the iterator_stage1.py file
# visualizes effects of adjusting bandpass range
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_analysis():
    # 1. Load the data
    try:
        df = pd.read_csv('sweep_01_bandpass.csv')
    except FileNotFoundError:
        print("Error: sweep_01_bandpass.csv not found.")
        return

    # 2. Prepare labels and grouping
    # Create a descriptive label for each of the 30 configurations
    df['label'] = df['f_low'].astype(str) + " / " + df['f_high'].astype(str)
    
    # Define colors for the 5 distinct groups of f_low (0.1, 0.2, 0.4, 1.0, 1.5)
    # Each group contains 6 rows (different f_high values)
    group_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']
    bar_colors = []
    for i in range(5):
        bar_colors.extend([group_colors[i]] * 6)

    # 3. Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    x_indices = np.arange(len(df))

    # --- MAE Subplot ---
    bars1 = ax1.bar(x_indices, df['MAE'], color=bar_colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('MAE (BPM)', fontweight='bold')
    ax1.set_title('Mean Absolute Error (MAE) across Bandpass Settings', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # --- RMSE Subplot ---
    bars2 = ax2.bar(x_indices, df['RMSE'], color=bar_colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('RMSE (BPM)', fontweight='bold')
    ax2.set_title('Root Mean Square Error (RMSE) across Bandpass Settings', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # 4. Final Formatting
    plt.xticks(x_indices, df['label'], rotation=45, ha='right')
    ax2.set_xlabel('Filter Settings (f_low / f_high) [Hz]', fontweight='bold')
    
    # Add a custom legend to explain the groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=group_colors[0], lw=4, label='f_low = 0.1 Hz'),
        Line2D([0], [0], color=group_colors[1], lw=4, label='f_low = 0.2 Hz'),
        Line2D([0], [0], color=group_colors[2], lw=4, label='f_low = 0.4 Hz'),
        Line2D([0], [0], color=group_colors[3], lw=4, label='f_low = 1.0 Hz'),
        Line2D([0], [0], color=group_colors[4], lw=4, label='f_low = 1.5 Hz')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', title="Grouping by f_low")

    plt.tight_layout()
    
    # 5. Save and Show
    output_filename = 'analysis_01_bandpass.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis visualization saved to {output_filename}")
    plt.show() 

if __name__ == "__main__":
    run_analysis()
