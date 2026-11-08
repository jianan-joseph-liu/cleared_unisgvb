"""Compute posterior spectral-density samples and credible intervals."""
from typing import List, Tuple

import numpy as np
import tensorflow as tf

def compute_psd(
        Xmat_delta:tf.Tensor,
        Xmat_theta:tf.Tensor,
        p_dim:int,
        vi_samples: List[tf.Tensor],
        quantiles=[0.05, 0.5, 0.95],
        psd_scaling=1.0,
        fs=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is used to compute the spectral density given the best surrogate posterior parameters
    :param vi_samples: the surrogate posterior parameters

    Computes:
        1. All posterior PSD draws [n-samples, n-freq, dim, dim].
        2. Pointwise credible intervals [3, n-freq, dim, dim].
        3. A uniform credible band [3, n-freq, dim, dim].

    """
    delta2_all_s = tf.exp(
        tf.matmul(Xmat_delta, tf.transpose(vi_samples[0], [0, 2, 1]))
    )  # (500, #freq, p)


    D_all = tf.map_fn(
        lambda x: tf.linalg.diag(x), delta2_all_s
    ).numpy()  # (500, #freq, p, p)


    psd_all = D_all
    
    pointwise_ci = __get_pointwise_ci(psd_all, quantiles)
    uniform_ci = __get_uniform_ci(psd_all, pointwise_ci, quantiles)
    

    # changing freq from [0, 1/2] to [0, samp_freq/2] (and applying scaling)
    if fs:
        original_fmax = 0.5
        true_fmax = fs / 2
        new_scale = true_fmax / original_fmax
        psd_all = psd_all / new_scale
        pointwise_ci = pointwise_ci / new_scale
        uniform_ci = uniform_ci / new_scale

    return (
        psd_all * psd_scaling ** 2,
        pointwise_ci * psd_scaling ** 2,
        uniform_ci * psd_scaling ** 2,
    )
        

def __get_pointwise_ci(psd_all, quantiles):
    _, num_freq, p_dim, _ = psd_all.shape
    psd_q = np.zeros((len(quantiles), num_freq, p_dim, p_dim), dtype=float)

    diag_indices = np.diag_indices(p_dim)
    psd_q[:, :, diag_indices[0], diag_indices[1]] = np.quantile(
        np.real(psd_all[:, :, diag_indices[0], diag_indices[1]]),
        quantiles,
        axis=0,
    )

    return psd_q


def __get_uniform_ci(psd_all, pointwise_ci, quantiles):
    """Return a simultaneous credible band for a real, diagonal PSD.

    Each posterior draw contributes its largest MAD-standardized deviation over
    all frequencies and diagonal PSD elements.  The central coverage implied by
    ``quantiles`` then selects one common multiplier for the complete band.
    """
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.shape != (3,) or not np.isclose(quantiles[1], 0.5):
        raise ValueError(
            "uniform CI requires [lower, 0.5, upper] quantiles"
        )

    coverage = quantiles[2] - quantiles[0]
    if not 0 < coverage < 1:
        raise ValueError("uniform CI coverage must be between 0 and 1")

    diagonal = np.diagonal(np.real(psd_all), axis1=-2, axis2=-1)
    median = np.diagonal(pointwise_ci[1], axis1=-2, axis2=-1)
    abs_deviation = np.abs(diagonal - median[None, ...])
    mad = np.median(abs_deviation, axis=0)

    # A zero MAD means that all draws agree at that element.  It should add no
    # width to the band and must not create a division-by-zero NaN.
    standardized = np.divide(
        abs_deviation,
        mad[None, ...],
        out=np.zeros_like(abs_deviation),
        where=mad[None, ...] > 0,
    )
    max_deviation = np.max(standardized, axis=(1, 2))
    threshold = np.quantile(max_deviation, coverage)

    lower = median - threshold * mad
    upper = median + threshold * mad
    uniform_ci = np.zeros_like(pointwise_ci, dtype=float)
    for channel in range(psd_all.shape[-1]):
        uniform_ci[0, :, channel, channel] = lower[:, channel]
        uniform_ci[2, :, channel, channel] = upper[:, channel]
    uniform_ci[1] = pointwise_ci[1]
    return uniform_ci




















