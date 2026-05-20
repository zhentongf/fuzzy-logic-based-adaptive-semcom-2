import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_logs(csv_path1, csv_path2, output_image_path):
    # Load data
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    fig, (ax1_loss, ax2_loss) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Fixed SNR 20
    ax1_psnr = ax1_loss.twinx()
    
    lns1 = ax1_loss.plot(df1['epoch'], df1['train_loss'], 'r-', label='Train Loss')
    lns2 = ax1_loss.plot(df1['epoch'], df1['test_loss'], 'r--', label='Test Loss')
    lns3 = ax1_psnr.plot(df1['epoch'], df1['train_psnr'], 'b-', label='Train PSNR')
    lns4 = ax1_psnr.plot(df1['epoch'], df1['test_psnr'], 'b--', label='Test PSNR')
    
    ax1_loss.set_ylabel('Loss', color='r')
    ax1_psnr.set_ylabel('PSNR (dB)', color='b')
    ax1_loss.set_title('Fixed SNR 20 Training Log')
    
    # Combine legends
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax1_loss.legend(lns, labs, loc='center right')
    ax1_loss.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Dynamic SNR 10-30
    ax2_psnr = ax2_loss.twinx()
    
    lns1 = ax2_loss.plot(df2['epoch'], df2['train_loss'], 'r-', label='Train Loss')
    lns2 = ax2_loss.plot(df2['epoch'], df2['test_loss'], 'r--', label='Test Loss')
    lns3 = ax2_psnr.plot(df2['epoch'], df2['train_psnr'], 'b-', label='Train PSNR')
    lns4 = ax2_psnr.plot(df2['epoch'], df2['test_psnr'], 'b--', label='Test PSNR')
    
    ax2_loss.set_ylabel('Loss', color='r')
    ax2_psnr.set_ylabel('PSNR (dB)', color='b')
    ax2_loss.set_title('Dynamic SNR 10-30 Training Log')
    ax2_loss.set_xlabel('Epoch')
    
    # Combine legends
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax2_loss.legend(lns, labs, loc='center right')
    ax2_loss.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Graph saved to {output_image_path}")
    plt.close()

if __name__ == "__main__":
    csv1 = r"train_encoder_decoder_cifar10\experiments\exp_20260512_184610_fixed_snr_20\training_log.csv"
    csv2 = r"train_encoder_decoder_cifar10\experiments\exp_20260512_202705_dynamic_snr_10to30\training_log.csv"
    
    # Ensure the script can find the files if run from different locations
    # But here we assume it's run from the project root.
    
    output_dir = "graphs_scripts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_img = os.path.join(output_dir, "training_comparison.png")
    
    plot_training_logs(csv1, csv2, output_img)
