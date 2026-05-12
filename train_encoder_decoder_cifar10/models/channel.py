import torch


def add_awgn_noise(x, snr_db):

    signal_power = torch.mean(x ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_power = signal_power / snr_linear

    noise = torch.randn_like(x) * torch.sqrt(noise_power)

    return x + noise