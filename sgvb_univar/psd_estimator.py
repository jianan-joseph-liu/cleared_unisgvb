import time
from typing import Tuple

import numpy as np
from hyperopt import fmin, hp, tpe
from .backend import BayesianModel, ViRunner
from .utils.tf_utils import set_seed
from .utils.utils import get_freq

class PSDEstimator:
    """Estimate a univariate power spectral density with SGVB.

    Call :meth:`run` to fit the model and populate ``psd_all``,
    ``pointwise_ci``, and ``uniform_ci``.
    """

    def __init__(
            self,
            x,
            N_theta=30,
            nchunks=1,
            ntrain_map=5000,
            N_samples=500,
            fs=1.0,
            max_hyperparm_eval=100,
            frange=None,
            degree_fluctuate=None,
            seed=None,
            lr_range=(0.002, 0.02),
            n_elbo_maximisation_steps=500,
            init_params=None,
            Nbw=1.0,
    ):
        """Initialize the estimator.

        Parameters
        ----------
        x : numpy.ndarray
            Real time series with shape ``(n_samples, 1)``.
        N_theta : int, default=30
            Number of cosine terms, in addition to the intercept and linear term.
        nchunks : int, default=1
            Number of non-overlapping blocks used by the likelihood.
        ntrain_map : int, default=5000
            Number of MAP optimization steps.
        N_samples : int, default=500
            Number of draws from the variational posterior.
        fs : float, default=1.0
            Sampling frequency in Hz.
        max_hyperparm_eval : int, default=100
            Maximum learning-rate evaluations when ``run(lr=None)``.
        frange : sequence of float or None, default=None
            Analysis range ``[fmin, fmax]`` in Hz.
        degree_fluctuate : float or None, default=None
            Prior smoothness hyperparameter.
        seed : int or None, default=None
            Random seed.
        lr_range : tuple of float, default=(0.002, 0.02)
            Learning-rate search interval.
        n_elbo_maximisation_steps : int, default=500
            Number of ELBO optimization steps.
        init_params : list or None, default=None
            Optional initial model parameters.
        Nbw : float, default=1.0
            Equivalent-noise-bandwidth factor dividing the block likelihood.
        """

        if seed is not None:
            set_seed(seed)

        self.N_theta = N_theta
        self.N_samples = N_samples
        self.nchunks = nchunks
        self.ntrain_map = ntrain_map
        self.n_elbo_maximisation_steps = n_elbo_maximisation_steps
        if not np.isfinite(Nbw) or Nbw <= 0:
            raise ValueError("Nbw must be a positive finite number")
        self.Nbw = float(Nbw)

        self.fs = fs
        self.lr_range = lr_range

        # normalize the data
        self.psd_scaling = np.std(x)
        self.psd_offset = 0.0
        self.x = (x - self.psd_offset) / self.psd_scaling
        self.n, self.p = x.shape
        if frange is None:
            frange = [0, self.fs // 2]

        self.frange = frange

        self.max_hyperparm_eval = max_hyperparm_eval
        self.degree_fluctuate = degree_fluctuate


        # Internal variables
        self.model: BayesianModel = None
        self.samps = None
        self.pointwise_ci = None
        self.uniform_ci = None
        self.psd_all = None
        self.runtimes = {}
        self.inference_runner = ViRunner(
            self.x,
            N_theta=self.N_theta,
            nchunks=self.nchunks,
            frange=self.frange,
            degree_fluctuate=self.degree_fluctuate,
            fs=self.fs,
            init_params=init_params,
            Nbw=self.Nbw,
        )

    def _learning_rate_optimisation_objective(self, lr):
        """
        Objective function for hyperparameter optimization of the learning rate for MAP.

        Args:
            lr (dict): Dictionary containing the learning rate to be optimized.

        Returns:
            float: ELBO log_map_vals.
        """
        vi_losses, _, _, _ = self.inference_runner.run(
            lr_map=lr["lr_map"],
            ntrain_map=self.ntrain_map,
            inference_size=self.N_samples,
            n_elbo_maximisation_steps=self.n_elbo_maximisation_steps,
        )
        return vi_losses[-1].numpy()

    def _find_optimal_learing_rate(self):
        """
        Find the optimal learning rate using hyperopt.

        This method uses the TPE algorithm to optimize the learning rate.
        """
        self.optimal_lr = fmin(
            self._learning_rate_optimisation_objective,
            space={"lr_map": hp.uniform("lr_map", *self.lr_range)},
            algo=tpe.suggest,
            max_evals=self.max_hyperparm_eval,
        )["lr_map"]
        #logger.info(f"Optimal learning rate: {self.optimal_lr:.4e}")

    def _train_model(self):
        """
        Train the model using the optimal learning rate.

        This method runs the variational inference to estimate the posterior PSD.
        """
        _, _, self.model, self.samps = self.inference_runner.run(
            lr_map=self.optimal_lr,
            ntrain_map=self.ntrain_map,
            inference_size=self.N_samples,
            n_elbo_maximisation_steps=self.n_elbo_maximisation_steps,
        )

    def run(self, lr=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the SGVB algorithm to estimate the posterior PSD.

        This method either uses a provided learning rate or finds the optimal one,
        then trains the model and computes the posterior PSD.

        :param lr: Learning rate for MAP. If None, optimal rate is found, defaults to None.
        :type lr: float, optional
        :return: Tuple containing all posterior PSD draws, pointwise CI, and uniform CI.
        :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
        """
        times = {}

        if lr:
            #logger.info(f"Using provided learning rate: {lr}")
            self.optimal_lr = lr
        else:
            #logger.info("Running hyperopt to find optimal learning rate")
            t0 = time.time()
            self._find_optimal_learing_rate()
            times["lr"] = time.time() - t0
            #logger.info(f"Optimal learning rate found in {times['lr']:.2f}s")

        #logger.info("Training model")
        t0 = time.time()
        self._train_model()
        times["train"] = time.time() - t0
        #logger.info(f"Model trained in {times['train']:.2f}s")

        #logger.info("Computing posterior PSDs")
        self.psd_all, self.pointwise_ci, self.uniform_ci = self.model.compute_psd(
            self.samps, psd_scaling=self.psd_scaling, fs=self.fs
        )
        self.runtimes = times
        return self.psd_all, self.pointwise_ci, self.uniform_ci

    @property
    def freq(self) -> np.ndarray:
        """
        Get the frequencies per chunk of the PSD estimate.

        Returns:
            np.ndarray: Array of frequencies.
        """
        if not hasattr(self, "_freq"):
            self._freq = get_freq(
                fs=self.fs,
                n_time_samples=self.nt_per_chunk,
                frange=self.frange
            )
        return self._freq

    @property
    def nt_per_chunk(self) -> int:
        """Return the number of time-points per chunk"""
        return self.n // self.nchunks

    @property
    def nfreq_per_chunk(self) -> int:
        """Return the number of frequencies per chunk"""
        return len(self.freq)

    def sample_posterior(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample the posterior distribution.

        :param n_samples: Number of samples to draw, defaults to 1000
        :type n_samples: int, optional
        :return: Samples from the posterior distribution
        :rtype: np.ndarray
        """
        spline_params = self.inference_runner.surrogate_posterior.sample(n_samples)
        psd, _, _ = self.model.compute_psd(
            spline_params, psd_scaling=self.psd_scaling, fs=self.fs
        )
        return spline_params, psd
