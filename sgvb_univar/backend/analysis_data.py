import numpy as np
from typing import Tuple, List
import tensorflow as tf
#from ..logging import logger


class AnalysisData:  # Parent used to create BayesianModel object
    def __init__(
            self,
            x: np.ndarray,
            nchunks: int = 128,
            frange: List[float] = [],
            fs: float = 2048.,
            N_theta: int = 15,
            N_delta: int = 15
    ):
        # x:      N-by-p, multivariate timeseries with N samples and p dimensions
        # y_ft:   fourier transformed time series
        # freq:   frequencies w/ y_ft
        # p:  dimension of x
        # Xmat:   basis matrix
        # Zar:    arry of design matrix Z_k for every freq k
        self.x = x
        self.p = x.shape[1]
        self.nchunks = nchunks
        self.N_theta = N_theta
        self.N_delta = N_delta

        self.fs = fs
        self.frange = frange

        # Compute the required datasets
        self.y_ft, self.freq = compute_chunked_fft(
            self.x,
            self.nchunks,
            self.frange,
            self.fs,
        )
        self.Zar = _compute_chunked_Zmatrix(self.y_ft)
        Xmat_delta, Xmat_theta = _compute_Xmatrices(self.freq, N_delta, N_theta)

        # Setup tensors
        y_ft = tf.convert_to_tensor(self.y_ft, dtype=tf.complex64)
        self.y_re = tf.math.real(y_ft)
        self.y_im = tf.math.imag(y_ft)
        self.n_seg = self.y_re.shape[0]
        periodograms = tf.square(self.y_re) + tf.square(self.y_im)
        self.periodo_mean = tf.reduce_mean(periodograms, axis=0, keepdims=True)
        self.numerator = self.periodo_mean * self.n_seg
        self.Xmat_delta = tf.convert_to_tensor(
            Xmat_delta, dtype=tf.float32
        )
        self.Xmat_theta = tf.convert_to_tensor(
            Xmat_theta, dtype=tf.float32
        )

        Zar = tf.convert_to_tensor(self.Zar, dtype=tf.complex64)
        self.Z_re = tf.math.real(Zar)
        self.Z_im = tf.math.imag(Zar)

        #logger.info(f"Loaded {self}")

    def __repr__(self):
        x = self.x.shape
        y = self.y_ft.shape
        Xd = self.Xmat_delta.shape
        Xt = self.Xmat_theta.shape
        Z = self.Zar.shape
        return f"AnalysisData(x(t)={x}, y(f)={y}, Xmat_delta={Xd}, Xmat_theta={Xt}, Z={Z})"


def _compute_Xmatrices(freq, N_delta: int = 15, N_theta: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the X matrices for delta and theta based on the provided frequencies.

    Parameters:
    freq (np.ndarray): vector of frequencies
    N_delta (int): The number of basis functions to use for delta (default is 15).
    N_theta (int): The number of basis functions to use for theta (default is 15).

    Returns:
    Xd (np.ndarray): The design matrix for Demmler-Reinsch basis functions of delta,
                     the shape is (n, 2 + N_delta).
    Xt (np.ndarray): The design matrix for Demmler-Reinsch basis functions of theta,
                     the shape is (n, 2 + N_theta).

    """
    fstack = np.column_stack([np.repeat(1, freq.shape[0]), freq])
    Xd = np.concatenate([fstack, DR_basis(freq, N=N_delta)], axis=1)
    Xt = np.concatenate([fstack, DR_basis(freq, N=N_theta)], axis=1)
    return Xd, Xt


def _compute_Zmatrix(y_k: np.ndarray) -> np.ndarray:
    """
    Compute the design matrix Z_k for each frequency k.

    Parameters:
    y_k (np.ndarray): Fourier transformed time series data of shape (n, p).

    Returns:
    np.ndarray: Design matrix Z_k of shape (n, p, p*(p-1)/2).
    """
    n, p = y_k.shape
    Z_k = np.zeros((n, p, int(p * (p - 1) / 2)), dtype=np.complex64)

    for j in range(n):
        count = 0
        for i in range(1, p):
            Z_k[j, i, count: count + i] = y_k[j, :i]
            count += i

    return Z_k


def _compute_chunked_Zmatrix(y_ft: np.ndarray) -> np.ndarray:
    """
    Compute the design matrix Z, a 3D array (The design matrix Z_k for every frequency k).

    Parameters:
    y_ft (np.ndarray): Fourier transformed time series data of shape (chunks, n_per_chunk, p).

    Returns:
    np.ndarray: 3D array of design matrices Z_k for each frequency k.
    """
    chunks, n_per_chunk, p = y_ft.shape
    if p == 1:
        return np.zeros((chunks, n_per_chunk, 0), dtype=np.complex64)

    if chunks == 1:
        y_ls = np.squeeze(y_ft, axis=0)
        Z_ = _compute_Zmatrix(y_ls)
    else:
        y_ls = np.squeeze(np.split(y_ft, chunks))
        Z_ = np.array([_compute_Zmatrix(x) for x in y_ls])

    return Z_


def DR_basis(freq: np.ndarray, N=10):
    """
    Return the basis matrix for the Demmler-Reinsch basis
    for linear smoothing splines (Eubank,1999)

            # freq: vector of frequencies
    # N:  amount of basis used
    # return a len(freq)-by-N matrix
    """
    return np.array(
        [
            np.sqrt(2) * np.cos(x * np.pi * freq * 2)
            for x in np.arange(1, N + 1)
        ]
    ).T


def compute_chunked_fft(
        x: np.ndarray,
        nchunks: int,
        frange: List[float],
        fs: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the FFT of input data split into chunks, discard DC component, and retain only
    the positive frequencies within the specified frequency range.

    Parameters:
    ----------
    x : np.ndarray
        Input time-domain data of shape (n_samples, n_dim).
    nchunks : int
        Number of chunks to split the data into.
    frange : List[float]
        Frequency range [fmin, fmax] to retain after FFT (in Hz).
    fs : float
        Sampling frequency (Hz).
    Returns:
    -------
    y_ft : np.ndarray
        Chunked and scaled FFT output of shape (nchunks, n_freq, n_dim).
    ftrue_y : np.ndarray
        Frequencies corresponding to the retained FFT bins (in Hz).
    """
    if x.ndim != 2:
        raise ValueError("Input x must be a 2D array (n_samples, n_dim).")
    if len(frange) != 2 or frange[0] >= frange[1]:
        raise ValueError("frange must be a list of two increasing floats [fmin, fmax].")

    orig_n, p = x.shape
    if orig_n < p:
        raise ValueError(f"Number of samples {orig_n} is less than number of dimensions {p}.")

    # Split x into chunks
    n_per_chunk = orig_n // nchunks
    x = x[:nchunks * n_per_chunk]  # truncate to make evenly divisible
    chunked_x = x.reshape(nchunks, n_per_chunk, p)

    # Remove mean from each chunk
    #chunked_x -= np.mean(chunked_x, axis=1, keepdims=True)

    # FFT along time axis (axis=1)
    y_ft = np.fft.fft(chunked_x, axis=1) / np.sqrt(n_per_chunk)

    # Frequency axes
    Ts = 1  # for VB backend we use Duration of 1.0 (rescale later)
    ftrue_y = np.fft.fftfreq(n_per_chunk, d=1 / fs)
    fq_y = np.fft.fftfreq(np.size(chunked_x, axis=1), Ts)

    # Keep only positive frequencies (excluding DC)
    if n_per_chunk % 2 == 0:
        idx = n_per_chunk // 2
    else:
        idx = (n_per_chunk - 1) // 2

    ftrue_y = ftrue_y[1:idx]
    fq_y = fq_y[1:idx]
    y_ft = y_ft[:, 1:idx, :]

    # Apply frequency mask based on frange
    fmin, fmax = frange
    mask = (ftrue_y >= fmin) & (ftrue_y <= fmax)
    y_ft = y_ft[:, mask, :]
    ftrue_y = ftrue_y[mask]
    fq_y = fq_y[mask]

    return y_ft, fq_y
