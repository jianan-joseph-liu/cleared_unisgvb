import numpy as np


def get_freq(fs: float, n_time_samples: int, frange=None) -> np.ndarray:
    n = n_time_samples
    dt = 1 / fs
    freq = np.fft.fftfreq(n, d=dt)
    if np.mod(n, 2) == 0:  # the length per chunk is even
        freq = freq[1: int(n / 2)]
    else:  # the length per chunk is odd
        freq = freq[1: int((n - 1) / 2)]

    fmin, fmax = frange
    if frange is not None:
        fmax_idx = np.searchsorted(freq, fmax)
        fmin_idx = np.searchsorted(freq, fmin)
        freq = freq[fmin_idx:fmax_idx+1]
    return freq
