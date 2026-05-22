"""
Statistical distributions used for detrending the climate variables.

This module contains classes for different statistical distributions used in the
fitting process.

See also Table 1 in [Mengel et al. (2021)](https://doi.org/10.5194/gmd-14-5269-2021).
"""

from dataclasses import dataclass

import numpy as np

# ArrayLike might not look nice with pdoc https://github.com/mitmproxy/pdoc/issues/420
from numpy.typing import ArrayLike
from scipy import stats

UPPER_TAIL_QUANTILE = 0.5


class Distribution:
    """Base class for statistical distributions."""

    def cdf(self, y):
        """
        Calculate the cumulative distribution function (CDF) for the distribution.

        Parameters
        ----------
        y : float or ArrayLike
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        float or ArrayLike
            The cumulative probability of the distribution at `y`.
        """
        raise NotImplementedError

    def invcdf(self, quantile):
        """
        Calculate the inverse CDF (quantile function) for the distribution.

        Parameters
        ----------
        quantile : float or ArrayLike
            The quantile value(s) (probability) to compute the inverse CDF for.

        Returns
        -------
        float or ArrayLike
            The corresponding value(s) for the given quantile in the distribution.
        """
        raise NotImplementedError

    def sf(self, y):
        """
        Calculate the survival function for the distribution.

        This is equivalent to ``1 - cdf(y)``, but subclasses should use a native
        implementation where available to preserve upper-tail precision.
        """
        return 1 - self.cdf(y)

    def invsf(self, probability):
        """
        Calculate the inverse survival function for the distribution.

        This is equivalent to ``invcdf(1 - probability)``, but subclasses should
        use a native implementation where available to preserve upper-tail
        precision.
        """
        return self.invcdf(1 - probability)

    def map_quantile_to(self, y, other):
        """
        Map values to the same quantiles in another distribution.

        Lower-tail values use CDF/PPF; upper-tail values use SF/ISF to avoid
        losing precision when the CDF rounds to 1.
        """
        quantile = self.cdf(y)
        mapped = other.invcdf(quantile)
        use_upper_tail = quantile > UPPER_TAIL_QUANTILE
        mapped_upper_tail = other.invsf(self.sf(y))
        return np.where(use_upper_tail, mapped_upper_tail, mapped)

    def expectation(self):
        """
        Calculate the expected value of the distribution.

        Returns
        -------
        float or ArrayLike
            The expected value(s) of the distribution.
        """
        raise NotImplementedError


@dataclass
class Normal(Distribution):
    """
    Normal distribution.

    Attributes
    ----------
    mu : float or ArrayLike
        Mean value of the distribution.
    sigma : float or ArrayLike
        Standard deviation of the distribution.
    """

    mu: float | ArrayLike
    sigma: float | ArrayLike

    # docstr-coverage:inherited
    def cdf(self, y):
        return stats.norm.cdf(y, loc=self.mu, scale=self.sigma)

    # docstr-coverage:inherited
    def invcdf(self, quantile):
        return stats.norm.ppf(quantile, loc=self.mu, scale=self.sigma)

    # docstr-coverage:inherited
    def sf(self, y):
        return stats.norm.sf(y, loc=self.mu, scale=self.sigma)

    # docstr-coverage:inherited
    def invsf(self, probability):
        return stats.norm.isf(probability, loc=self.mu, scale=self.sigma)

    # docstr-coverage:inherited
    def expectation(self):
        return self.mu


@dataclass
class BernoulliGamma(Distribution):
    """
    Bernoulli-Gamma distribution, a mixture of a Bernoulli and a Gamma distribution.

    Attributes
    ----------
    p : float or ArrayLike
        Probability of the Bernoulli distribution, i.e. of a value according to the
        Gamma distribution part.
    mu : float or ArrayLike
        Mean value of the Gamma distribution.
    nu : float or ArrayLike
        Shape parameter of the Gamma distribution.
    """

    p: float | ArrayLike
    mu: float | ArrayLike
    nu: float | ArrayLike

    # docstr-coverage:inherited
    def cdf(self, y):
        return self.p + (1 - self.p) * stats.gamma.cdf(
            y, self.nu**2.0, scale=self.mu / self.nu**2.0
        )

    # docstr-coverage:inherited
    def invcdf(self, quantile):
        return stats.gamma.ppf(
            (quantile - self.p) / (1 - self.p),
            self.nu**2.0,
            scale=self.mu / self.nu**2.0,
        )

    # docstr-coverage:inherited
    def sf(self, y):
        return (1 - self.p) * stats.gamma.sf(
            y, self.nu**2.0, scale=self.mu / self.nu**2.0
        )

    # docstr-coverage:inherited
    def invsf(self, probability):
        return stats.gamma.isf(
            probability / (1 - self.p),
            self.nu**2.0,
            scale=self.mu / self.nu**2.0,
        )

    # docstr-coverage:inherited
    def expectation(self):
        return (1 - self.p) * self.mu


@dataclass
class Gamma(Distribution):
    """
    Gamma distribution.

    Attributes
    ----------
    mu : float or ArrayLike
        Mean value of the distribution.
    nu : float or ArrayLike
        Shape parameter of the distribution.
    """

    mu: float | ArrayLike
    nu: float | ArrayLike

    # docstr-coverage:inherited
    def cdf(self, y):
        return stats.gamma.cdf(y, self.nu**2.0, scale=self.mu / self.nu**2.0)

    # docstr-coverage:inherited
    def invcdf(self, quantile):
        return stats.gamma.ppf(quantile, self.nu**2.0, scale=self.mu / self.nu**2.0)

    # docstr-coverage:inherited
    def sf(self, y):
        return stats.gamma.sf(y, self.nu**2.0, scale=self.mu / self.nu**2.0)

    # docstr-coverage:inherited
    def invsf(self, probability):
        return stats.gamma.isf(probability, self.nu**2.0, scale=self.mu / self.nu**2.0)

    # docstr-coverage:inherited
    def expectation(self):
        return self.mu


@dataclass
class Beta(Distribution):
    """
    Beta distribution.

    Attributes
    ----------
    mu : float or ArrayLike
        Mean value of the distribution.
    phi : float or ArrayLike
        Shape parameter of the distribution.
    """

    mu: float | ArrayLike
    phi: float | ArrayLike

    # docstr-coverage:inherited
    def cdf(self, y):
        return stats.beta.cdf(y, self.mu * self.phi, (1 - self.mu) * self.phi)

    # docstr-coverage:inherited
    def invcdf(self, quantile):
        return stats.beta.ppf(quantile, self.mu * self.phi, (1 - self.mu) * self.phi)

    # docstr-coverage:inherited
    def sf(self, y):
        return stats.beta.sf(y, self.mu * self.phi, (1 - self.mu) * self.phi)

    # docstr-coverage:inherited
    def invsf(self, probability):
        return stats.beta.isf(probability, self.mu * self.phi, (1 - self.mu) * self.phi)

    # docstr-coverage:inherited
    def expectation(self):
        return self.mu


@dataclass
class Weibull(Distribution):
    """
    Weibull distribution.

    Attributes
    ----------
    alpha : float or ArrayLike
        Shape parameter of the distribution.
    beta : float or ArrayLike
        Scale parameter of the distribution.
    """

    alpha: float | ArrayLike
    beta: float | ArrayLike

    # docstr-coverage:inherited
    def cdf(self, y):
        return stats.weibull_min.cdf(y, self.alpha, scale=self.beta)

    # docstr-coverage:inherited
    def invcdf(self, quantile):
        return stats.weibull_min.ppf(quantile, self.alpha, scale=self.beta)

    # docstr-coverage:inherited
    def sf(self, y):
        return stats.weibull_min.sf(y, self.alpha, scale=self.beta)

    # docstr-coverage:inherited
    def invsf(self, probability):
        return stats.weibull_min.isf(probability, self.alpha, scale=self.beta)

    # docstr-coverage:inherited
    def expectation(self):
        return stats.weibull_min.mean(self.alpha, scale=self.beta)
