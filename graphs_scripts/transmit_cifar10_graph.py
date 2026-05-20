import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_transmission_comparison(csv_path, output_dir):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Define bins for distance and speed
    dist_bins = np.arange(0, 201, 10)  # 0, 10, 20, ..., 200
    speed_bins = np.arange(0, 101, 10) # 0, 10, 20, ..., 100
    
    # Create binned columns
    df['dist_bin'] = pd.cut(df['distance_values'], bins=dist_bins)
    df['speed_bin'] = pd.cut(df['rel_speed_values'], bins=speed_bins)
    
    # Helper to get bin midpoint for plotting
    def get_bin_midpoint(bin_interval):
        if pd.isna(bin_interval): return np.nan
        return bin_interval.mid

    df['dist_mid'] = df['dist_bin'].apply(get_bin_midpoint)
    df['speed_mid'] = df['speed_bin'].apply(get_bin_midpoint)

    cases = [
        'fuzzy_logic_fixed_snr',
        'fuzzy_logic_dynamic_snr',
        'all_dynamic_snr',
        'all_direct'
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Transmission Performance Comparison (Binned Analysis)', fontsize=16)

    # Subplot 1: PSNR vs Distance
    for i, case in enumerate(cases):
        case_df = df[df['testing_case'] == case]
        if case_df.empty:
            continue
            
        # Group by bin and calculate mean
        stats = case_df.groupby('dist_mid', observed=False)['psnr'].mean().sort_index()
        ax1.plot(stats.index, stats.values, label=case, color=colors[i], marker=markers[i], linestyle='-', linewidth=2)

    ax1.set_xlabel('Distance (m) [Binned per 10m]')
    ax1.set_ylabel('Mean PSNR (dB)')
    ax1.set_title('PSNR vs Distance (0-200m)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Subplot 2: PSNR vs Relative Speed
    for i, case in enumerate(cases):
        case_df = df[df['testing_case'] == case]
        if case_df.empty:
            continue
            
        # Group by bin and calculate mean
        stats = case_df.groupby('speed_mid', observed=False)['psnr'].mean().sort_index()
        ax2.plot(stats.index, stats.values, label=case, color=colors[i], marker=markers[i], linestyle='-', linewidth=2)

    ax2.set_xlabel('Relative Speed (m/s) [Binned per 10m/s]')
    ax2.set_ylabel('Mean PSNR (dB)')
    ax2.set_title('PSNR vs Relative Speed (0-100m/s)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = os.path.join(output_dir, "transmission_comparison_binned.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_path = r"transmit_cifar10\experiments\transmission_results_20260513_120645.csv"
    output_dir = "graphs_scripts"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(csv_path):
        plot_transmission_comparison(csv_path, output_dir)
    else:
        print(f"Error: CSV file not found at {csv_path}")
