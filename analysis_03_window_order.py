import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def run_analysis():
    # 1. Load the data
    try:
        df = pd.read_csv('sweep_03_window_order.csv')
    except FileNotFoundError:
        print("Error: sweep_03_window_order.csv not found.")
        return

    # Sort to ensure groups are contiguous
    df = df.sort_values(['window_size', 'yw_order']).reset_index(drop=True)

    # 2. Prepare labels and grouping
    df['label'] = df['window_size'].astype(str) + "s / " + df['yw_order'].astype(str)
    
    unique_windows = df['window_size'].unique()
    num_groups = len(unique_windows)
    
    # Define colors using a professional colormap
    colormap = cm.get_cmap('viridis')
    group_colors = [colormap(i) for i in np.linspace(0.1, 0.9, num_groups)]
    
    bar_colors = []
    for i in range(num_groups):
        count = len(df[df['window_size'] == unique_windows[i]])
        bar_colors.extend([group_colors[i]] * count)

    # 3. Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    
    x_indices = np.arange(len(df))

    # --- MAE Subplot ---
    bars1 = ax1.bar(x_indices, df['MAE'], color=bar_colors, edgecolor='black', alpha=0.85)
    ax1.set_ylabel('MAE (BPM)', fontweight='bold')
    ax1.set_title('MAE across Window Size and AR Model Order (Window / Order)', fontsize=16, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # --- RMSE Subplot ---
    bars2 = ax2.bar(x_indices, df['RMSE'], color=bar_colors, edgecolor='black', alpha=0.85)
    ax2.set_ylabel('RMSE (BPM)', fontweight='bold')
    ax2.set_title('RMSE across Window Size and AR Model Order (Window / Order)', fontsize=16, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 4. Final Formatting
    plt.xticks(x_indices, df['label'], rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Configuration (Window Size [s] / AR Model Order)', fontweight='bold', fontsize=12)
    
    # Add a custom legend to explain the groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=group_colors[i], lw=6, label=f'Window = {unique_windows[i]}s')
        for i in range(num_groups)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', title="Grouping by Window Size", 
               title_fontproperties={'weight':'bold'}, fontsize=10, framealpha=0.9)

    plt.tight_layout()
    
    # 5. Save and Show
    output_filename = 'analysis_03_window_order.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis visualization (Window/Order) saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    run_analysis()
