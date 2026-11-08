#!/usr/bin/env python3
"""GW150914 off-source PSD estimation with AR-SGVB, followed by plotting.

Edit the variables below and the SGVB arguments in the estimation block.
"""
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hyperopt import fmin, hp, tpe
from scipy.linalg import solve_toeplitz
from scipy.signal import correlate, welch
from scipy.signal.windows import tukey

from sgvb_univar.psd_estimator import PSDEstimator as SGVBPSDEstimator


detectors = ("H1", "L1")
base_directory = Path(__file__).resolve().parent
data_directory = base_directory / "data_cache"
output_directory = base_directory / "outputs"

segment_length = 4
number_of_blocks = 32
frequency_range = [20, 896]
roll_off = 0.4
learning_rate = 0.012

orders = list(range(300, 601, 5))
max_ar_evaluations = 40
tukey_alpha = 2 * roll_off / segment_length


def apply_tukey_window(data, number_of_chunks):
    """Apply one Tukey window to every non-overlapping block."""
    chunk_length = len(data) // number_of_chunks
    window = tukey(chunk_length, alpha=tukey_alpha)
    chunks = data.reshape(number_of_chunks, chunk_length)
    return (chunks * window).reshape(-1), np.mean(window**2)


def calculate_autocorrelation(data, maximum_order):
    """Calculate the biased autocorrelation up to the largest AR order."""
    data = data - np.mean(data)
    full_correlation = correlate(data, data, mode="full", method="fft")
    zero_lag_index = len(data) - 1
    return (
        full_correlation[zero_lag_index:zero_lag_index + maximum_order + 1]
        / len(data)
    )


def calculate_ar_psd(autocorrelation, order, window_mean_square):
    """Fit the Yule--Walker AR model and calculate its one-sided PSD."""
    variance = float(autocorrelation[0])

    correlation = np.asarray(autocorrelation[:order + 1]) / variance
    ar = solve_toeplitz(correlation[:-1], -correlation[1:])
    noise_variance = variance * (1 + np.dot(ar, correlation[1:]))

    denominator = np.fft.rfft(np.r_[1.0, ar], n=NFFT)
    ar_psd = noise_variance / (fs * np.abs(denominator)**2)
    ar_psd[1:-1] *= 2
    ar_psd /= window_mean_square
    return ar_psd


def find_best_order(data, window_mean_square, welch_psd, frequency_mask):
    """Minimize AR-versus-Welch RIAE using Hyperopt."""
    autocorrelation = calculate_autocorrelation(data, max(orders))
    denominator = np.sum(np.abs(welch_psd))

    def objective(order):
        ar_psd = calculate_ar_psd(
            autocorrelation,
            int(order),
            window_mean_square,
        )[frequency_mask]
        return float(np.sum(np.abs(ar_psd - welch_psd)) / denominator)

    best = fmin(
        fn=objective,
        space=hp.quniform("AR_order", orders[0], orders[-1], orders[1] - orders[0]),
        algo=partial(tpe.suggest, n_startup_jobs=10),
        max_evals=max_ar_evaluations,
        show_progressbar=False,
        rstate=np.random.default_rng(seed),
    )
    best_order = int(best["AR_order"])
    best_riae = objective(best_order)
    best_ar_psd = calculate_ar_psd(
        autocorrelation,
        best_order,
        window_mean_square,
    )
    return best_order, best_riae, best_ar_psd


def whiten_by_ar_psd(data, ar_psd, number_of_chunks):
    """Whiten each block using the complete one-sided AR PSD grid."""
    chunk_length = len(data) // number_of_chunks
    chunks = data.reshape(number_of_chunks, chunk_length)
    fourier_data = np.fft.rfft(chunks, axis=1)
    whitened_fourier_data = fourier_data / np.sqrt(ar_psd)
    return np.fft.irfft(
        whitened_fourier_data,
        n=chunk_length,
        axis=1,
    ).reshape(-1)


output_directory.mkdir(parents=True, exist_ok=True)

for detector in detectors:
    seed = ("H1", "L1").index(detector)
    file_name = f"{detector}_1126259332400_1126259460400.npz"
    data_file = data_directory / file_name

    with np.load(data_file, allow_pickle=False) as saved_data:
        data = np.asarray(saved_data["values"], dtype=np.float64)
        epoch = float(saved_data["epoch"])
        dt = float(saved_data["dt"])

    fs = int(round(1 / dt))
    NFFT = int(round(segment_length * fs))


    # Apply the Tukey window once to each 4 s block.
    window = tukey(NFFT, alpha=tukey_alpha)
    Nbw = NFFT * np.sum(window**2) / np.sum(window)**2
    windowed_data, window_mean_square = apply_tukey_window(
        data,
        number_of_chunks=number_of_blocks,
    )
    frequencies = np.fft.rfftfreq(NFFT, d=dt)
    frequency_mask = (
        (frequencies >= frequency_range[0])
        & (frequencies <= frequency_range[1])
    )
    frequencies_band = frequencies[frequency_mask]

    # Median Welch PSD on the complete 128 s, used for AR order selection.
    _, welch_psd_full = welch(
        data,
        fs=fs,
        window=window,
        nperseg=NFFT,
        noverlap=NFFT // 2,
        detrend="constant",
        scaling="density",
        average="median",
    )
    welch_psd = welch_psd_full[frequency_mask]

    # Select the AR order and prewhiten the 32 blocks.
    ar_start = time.perf_counter()
    best_P, best_ar_riae, ar_psd_full = find_best_order(
        windowed_data,
        window_mean_square,
        welch_psd,
        frequency_mask,
    )
    ar_time = time.perf_counter() - ar_start
    ar_psd_band = ar_psd_full[frequency_mask]
    whitened_data = whiten_by_ar_psd(
        windowed_data,
        ar_psd_full,
        number_of_chunks=number_of_blocks,
    )
    estimator_data = whitened_data[:, np.newaxis]
    print(f"AR order={best_P}", flush=True)

    # AR-SGVB: estimate the residual spectrum and restore the original PSD.
    sgvb_start = time.perf_counter()
    sgvb_estimator = SGVBPSDEstimator(
        x=estimator_data,
        N_theta=4000,
        nchunks=number_of_blocks,
        ntrain_map=15000,
        N_samples=500,
        fs=fs,
        max_hyperparm_eval=1,
        degree_fluctuate=4000,
        n_elbo_maximisation_steps=600,
        frange=frequency_range,
        Nbw=Nbw,
        seed=seed,
    )
    
    sgvb_estimator.run(lr=learning_rate)
    sgvb_time = time.perf_counter() - sgvb_start
    arsgvb_ci = (
        sgvb_estimator.pointwise_ci[:, :, 0, 0].real
        * 2
        / window_mean_square
        * ar_psd_band[np.newaxis, :]
    )
    arsgvb_uniform_ci = (
        sgvb_estimator.uniform_ci[:, :, 0, 0].real
        * 2
        / window_mean_square
        * ar_psd_band[np.newaxis, :]
    )
    arsgvb_lower, arsgvb_median, arsgvb_upper = arsgvb_ci
    losses = np.asarray(sgvb_estimator.inference_runner.kdl_losses)

    # Save the PSD curves and optimization histories.
    np.savez_compressed(
        output_directory / f"gw150914_ar_sgvb_{detector}.npz",
        frequency=frequencies_band,
        welch_psd=welch_psd,
        arsgvb_lower=arsgvb_lower,
        arsgvb_median=arsgvb_median,
        arsgvb_upper=arsgvb_upper,
        arsgvb_uniform_lower=arsgvb_uniform_ci[0],
        arsgvb_uniform_upper=arsgvb_uniform_ci[2],
        negative_elbo=losses,
        map_log_posterior=np.asarray(sgvb_estimator.inference_runner.lp),
    )


# Plot the saved AR-SGVB estimates and the existing comparison PSDs.
plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, axes = plt.subplots(
    len(detectors), 1,
    figsize=(9, 3.8 * len(detectors)),
    sharex=True,
    squeeze=False,
)

for ax, detector in zip(axes[:, 0], detectors):
    with np.load(
        output_directory / f"gw150914_ar_sgvb_{detector}.npz",
        allow_pickle=False,
    ) as saved_data:
        frequency = saved_data["frequency"]
        ax.loglog(frequency, saved_data["welch_psd"],
                  color="#59a14f", linewidth=1, alpha=0.8, label="Welch")
        ax.loglog(frequency, saved_data["arsgvb_median"],
                  color="#d94b4b", linewidth=1.2, label="AR-SGVB")
        ax.fill_between(
            frequency,
            saved_data["arsgvb_lower"],
            saved_data["arsgvb_upper"],
            color="#d94b4b",
            alpha=0.25,
            label="90% pointwise band",
        )

    bayeswave_file = base_directory / "reference_psds" / f"bayeswave_{detector.lower()}_psd.txt"
    bayeswave_data = np.loadtxt(bayeswave_file)
    keep = (
        (bayeswave_data[:, 0] >= frequency[0])
        & (bayeswave_data[:, 0] <= frequency[-1])
        & np.isfinite(bayeswave_data[:, 1])
        & (bayeswave_data[:, 1] > 0)
    )
    ax.loglog(
        bayeswave_data[keep, 0],
        bayeswave_data[keep, 1],
        color="#d8a000",
        linewidth=1,
        alpha=0.85,
        label="BayesWave",
    )

    ax.set_title(f"GW150914 off-source PSD: {detector}")
    ax.set_ylabel(r"PSD [strain$^2$/Hz]")
    ax.set_xlim(frequency[0], frequency[-1])
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=9)

axes[-1, 0].set_xlabel("Frequency [Hz]")
fig.tight_layout()
plot_path = output_directory / "gw150914_ar_sgvb_psd"
fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight")
fig.savefig(plot_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {plot_path}.pdf and .png", flush=True)
