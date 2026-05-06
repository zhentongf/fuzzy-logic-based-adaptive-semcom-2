import csv
import os

def log_csv(csv_file, row):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "epoch",
                "train_acc", "train_loss", "train_psnr",
                "test_acc", "test_loss", "test_psnr",
                "epoch_time", "est_total"
            ])

        writer.writerow(row)