import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_composite_snr_analysis(csv_path, output_dir):
    # Load data
    df = pd.read_csv(csv_path)
    
    # Round composite_snr_db to nearest integer for better grouping and visualization
    # since these values are likely continuous.
    df['snr_rounded'] = df['composite_snr_db'].round()

    cases = [
        'fuzzy_logic_fixed_snr',
        'fuzzy_logic_dynamic_snr',
        'all_dynamic_snr',
        'all_direct'
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    plt.figure(figsize=(12, 8))
    
    for i, case in enumerate(cases):
        case_df = df[df['testing_case'] == case]
        if case_df.empty:
            print(f"Warning: No data for case {case}")
            continue
            
        # Group by rounded SNR and calculate mean PSNR
        stats = case_df.groupby('snr_rounded')['psnr'].mean().sort_index(ascending=False)
        
        plt.plot(stats.index, stats.values, label=case, color=colors[i], marker=markers[i], 
                 markersize=4, linestyle='-', linewidth=2)

    plt.title('PSNR vs Composite SNR (Descending X-axis)', fontsize=14)
    plt.xlabel('Composite SNR (dB)', fontsize=12)
    plt.ylabel('Mean PSNR (dB)', fontsize=12)
    
    # Set X-axis to descending order from 40 to -30 as requested
    plt.xlim(40, -30)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "psnr_vs_composite_snr.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    csv_path = r"transmit_cifar10\experiments\transmission_results_20260513_120645.csv"
    output_dir = "graphs_scripts"
    
    if os.path.exists(csv_path):
        plot_composite_snr_analysis(csv_path, output_dir)
    else:
        print(f"Error: CSV file not found at {csv_path}")
