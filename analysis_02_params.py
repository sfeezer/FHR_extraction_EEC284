import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_analysis():
    # 1. Load the data
    try:
        df = pd.read_csv('sweep_02_params.csv')
    except FileNotFoundError:
        print("Error: sweep_02_params.csv not found.")
        return

    # 2. Prepare labels and grouping
    # lambda is a reserved keyword in python, but the column name is 'lambda'
    df['label'] = df['lambda'].astype(str) + " / " + df['delta'].astype(str)
    
    unique_lambdas = df['lambda'].unique()
    num_groups = len(unique_lambdas)
    rows_per_group = len(df) // num_groups
    
    # Define colors for the groups
    group_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e', '#1abc9c']
    bar_colors = []
    for i in range(num_groups):
        # Find how many rows belong to this specific lambda
        count = len(df[df['lambda'] == unique_lambdas[i]])
        bar_colors.extend([group_colors[i % len(group_colors)]] * count)

    # 3. Initialize the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    x_indices = np.arange(len(df))

    # --- MAE Subplot ---
    bars1 = ax1.bar(x_indices, df['MAE'], color=bar_colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('MAE (BPM)', fontweight='bold')
    ax1.set_title('MAE across RLS Parameters (Lambda / Delta)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # --- RMSE Subplot ---
    bars2 = ax2.bar(x_indices, df['RMSE'], color=bar_colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('RMSE (BPM)', fontweight='bold')
    ax2.set_title('RMSE across RLS Parameters (Lambda / Delta)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Annotate bars with values
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # 4. Final Formatting
    plt.xticks(x_indices, df['label'], rotation=45, ha='right')
    ax2.set_xlabel('RLS Parameters (Lambda / Delta)', fontweight='bold')
    
    # Add a custom legend to explain the groups
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=group_colors[i % len(group_colors)], lw=4, label=f'Lambda = {unique_lambdas[i]}')
        for i in range(num_groups)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', title="Grouping by Lambda")

    plt.tight_layout()
    
    # 5. Save and Show
    output_filename = 'analysis_02_params.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Analysis visualization saved to {output_filename}")
    plt.show()

if __name__ == "__main__":
    run_analysis()
