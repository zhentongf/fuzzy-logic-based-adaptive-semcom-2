import time

def estimate_total_time(start_time, current_epoch, total_epochs):
    elapsed = time.time() - start_time
    avg = elapsed / (current_epoch + 1)
    remaining = avg * (total_epochs - current_epoch - 1)
    return remaining