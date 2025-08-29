"""Base classes for representing statistical models.

For the implementations using different optimization libraries, see:

- `attrici.estimation.model_pymc5`
- `attrici.estimation.model_pymc3`
- `attrici.estimation.model_scipy`
"""

from dataclasses import dataclass
from typing import Callable

from func_timeout import FunctionTimedOut, func_timeout
from joblib import Memory
from loguru import logger


class Parameter:
    """Base class for model parameters."""

    pass


class AttriciGLM:
    """
    A class representing the ATTRICI Generalized Linear Model (GLM).

    Attributes
    ----------
    Parameter : dataclass
        A data structure for parameters.
    """

    @dataclass
    class Parameter(Parameter):
        """
        A data structure for parameters that depend on the predictor.

        Attributes
        ----------
        link : Callable
            The link function to be applied.
        dependent : bool
            Whether the parameter is dependent on the predictor.
        """

        link: Callable
        dependent: bool


class Model:
    """
    Base class for a statistical model.
    """

    def __init__(self, distribution, parameters, observed, predictor):
        """
        Initialize the model.

        Parameters
        ----------
        distribution : class
            The distribution class (e.g., `attrici.distributions.Bernoulli`).
        parameters : dict
            A dictionary of parameter names and their respective parameter objects.
        observed : xarray.DataArray
            The observed data to be used for the model.
        predictor : xarray.DataArray
            The predictor data to be used in the model.
        """
        raise NotImplementedError

    def fit(self, **kwargs):
        """
        Fit the statistical model to the data as given to the `__init__` method.

        Returns
        -------
        dict
            A dictionary of the fitted parameters, the "traces".
        """
        raise NotImplementedError

    def estimate_logp(self, trace, **kwargs):
        """
        Estimate the log-probability of the model fit.

        Parameters
        ----------
        trace : dict
            A dictionary of the fitted parameters, the "traces".

        Returns
        -------
        float
            The estimated log-probability.
        """
        raise NotImplementedError

    def estimate_distribution(self, trace, predictor, **kwargs):
        """
        Estimate the distribution of the model given specific predictor values.

        Parameters
        ----------
        trace : dict
            A dictionary of the fitted parameters, the "traces".
        predictor : xarray.DataArray
            The predictor data to be used.

        Returns
        -------
        attrici.distributions.Distribution
            The estimated distribution.
        """
        raise NotImplementedError

    def fit_cached(self, defining_data_inputs, cache_dir=None, timeout=None, **kwargs):
        """
        Fit the model using cached results if available.

        Parameters
        ----------
        defining_data_inputs : dict
            Data inputs that define the model fitting process, see `attrici.detrend`.
        cache_dir : str, pathlib.Path or None
            Directory to store the cache. If None, caching is disabled.
        timeout : float, optional
            Maximum time allowed for fitting. If None, no timeout is applied.
        **kwargs
            Additional keyword arguments passed to the `fit` method.
        """

        memory = Memory(cache_dir, verbose=0)

        def _fit(inputs):
            """
            Fit the model.

            Parameters
            ----------
            inputs : dict
                Data inputs that define the model fitting process. Not used directly in
                this method, but necessary for defining for what inputs the cache is
                valid.

            Returns
            -------
            dict
                A dictionary of the fitted parameters, the "traces".
            """
            return self.fit(**kwargs)

        def _cache_validation_callback(*args):
            """
            Callback function called after cache has been validated. Only used here to
            log the cache status.

            Returns
            -------
            bool
                `True` to indicate that the cache is valid.
            """
            logger.info("Using cached results")
            return True

        def _do_fit_cached():
            """
            Fit the model using cached results if available. Used as the function to be
            called with `func_timeout`.

            Returns
            -------
            dict or None
                A dictionary of the fitted parameters, the "traces", or `None` if
                fitting times out.
            """
            return memory.cache(
                _fit,
                cache_validation_callback=_cache_validation_callback,
            )(defining_data_inputs)

        if timeout is not None:
            try:
                return func_timeout(timeout, _do_fit_cached)
            except FunctionTimedOut:
                return None
        return _do_fit_cached()
