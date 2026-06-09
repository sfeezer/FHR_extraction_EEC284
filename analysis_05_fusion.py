# this file analyzes and graphs the results of the iterator_stage5.py file
# visualizes effects of adjusting weights

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def run_analysis():
    # 1. Load the data
    try:
        df = pd.read_csv('sweep_05_average.csv')
    except FileNotFoundError:
        print("Error: sweep_05_average.csv not found.")
        return

    # Sort by MAE to identify best performers
    df = df.sort_values('MAE').reset_index(drop=True)
    
    print("Top 5 Best Weight Configurations (Lowest MAE):")
    print(df.head(5).to_string(index=False))

    # 2. Visualization Setup
    fig = plt.figure(figsize=(18, 10))
    
    # --- Subplot 1: Parallel Coordinates ---
    # We'll create this manually for maximum control and beauty
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    
    # Columns to plot
    cols = ['w2', 'w3', 'w4', 'w5', 'MAE']
    x = range(len(cols))
    
    # Normalize data for coloring (color by MAE)
    norm = plt.Normalize(df['MAE'].min(), df['MAE'].max())
    colormap = cm.get_cmap('viridis_r') # Reverse viridis: yellow is high error, purple/blue is low
    
    for i, row in df.iterrows():
        color = colormap(norm(row['MAE']))
        # We want the 'best' lines to be on top, so we plot them last
        # But wait, df is sorted by MAE, so the best are at the start.
        # Let's plot in reverse order so best are on top.
        alpha = 0.6 if i < 10 else 0.1
        zorder = 100 - i
        ax1.plot(x, [row[c] for c in cols], color=color, alpha=alpha, zorder=zorder)

    ax1.set_xticks(x)
    ax1.set_xticklabels(cols, fontweight='bold', fontsize=12)
    ax1.set_title("Sensor Fusion Weight Sensitivity (Parallel Coordinates)\n(Purple/Dark lines indicate lower MAE)", 
                 fontsize=14, fontweight='bold')
    ax1.grid(axis='x', linestyle='-', alpha=0.3)
    
    # Add colorbar for MAE
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, ax=ax1, label='MAE (BPM)')

    # --- Subplot 2: Individual Weight Correlations ---
    # We want to see which channel's weight has the strongest trend with MAE
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    weights = ['w2', 'w3', 'w4', 'w5']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    
    for w, color in zip(weights, colors):
        # Calculate a rolling average or just scatter
        # Scatter with low alpha to see density
        ax2.scatter(df[w], df['MAE'], color=color, alpha=0.4, label=f'Channel {w}')
    
    ax2.set_xlabel('Weight Value', fontweight='bold')
    ax2.set_ylabel('MAE (BPM)', fontweight='bold')
    ax2.set_title('MAE vs. Individual Channel Weights', fontweight='bold')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot 3: Correlation Heatmap ---
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    corr = df.corr()
    # We specifically care about correlation with MAE
    mae_corr = corr['MAE'].drop(['MAE', 'RMSE'])
    
    bars = ax3.bar(mae_corr.index, mae_corr.values, color=['#3498db' if v > 0 else '#e74c3c' for v in mae_corr.values])
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('Correlation of Weights with MAE\n(Negative = Weight helps reduce error)', fontweight='bold')
    ax3.set_ylabel('Pearson Correlation Coefficient')
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')

    plt.tight_layout()
    
    # 5. Save and Show
    output_filename = 'analysis_05_fusion.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nAnalysis visualization saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    run_analysis()
