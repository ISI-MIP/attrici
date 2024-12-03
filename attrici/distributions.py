from dataclasses import dataclass
from typing import Any

from scipy import stats


class Distribution:
    pass


@dataclass
class Normal(Distribution):
    mu: Any
    sigma: Any

    def cdf(self, y):
        return stats.norm.cdf(y, loc=self.mu, scale=self.sigma)

    def invcdf(self, quantile):
        return stats.norm.ppf(quantile, loc=self.mu, scale=self.sigma)


@dataclass
class BernoulliGamma(Distribution):
    p: Any
    mu: Any
    nu: Any

    def cdf(self, y):
        return self.p + (1 - self.p) * stats.gamma.cdf(
            y, self.nu**2.0, scale=self.mu / self.nu**2.0
        )

    def invcdf(self, quantile):
        return stats.gamma.ppf(
            (quantile - self.p) / (1 - self.p),
            self.nu**2.0,
            scale=self.mu / self.nu**2.0,
        )


@dataclass
class Bernoulli(Distribution):
    p: Any

    def cdf(self, y):
        return self.p + (1 - self.p) * y

    def invcdf(self, quantile):
        return (quantile - self.p) / (1 - self.p)


@dataclass
class Gamma(Distribution):
    mu: Any
    nu: Any

    def cdf(self, y):
        return stats.gamma.cdf(y, self.nu**2.0, scale=self.mu / self.nu**2.0)

    def invcdf(self, quantile):
        return stats.gamma.ppf(quantile, self.nu**2.0, scale=self.mu / self.nu**2.0)


@dataclass
class Beta(Distribution):
    mu: Any
    phi: Any

    def cdf(self, y):
        return stats.beta.cdf(y, self.mu * self.phi, (1 - self.mu) * self.phi)

    def invcdf(self, quantile):
        return stats.beta.ppf(quantile, self.mu * self.phi, (1 - self.mu) * self.phi)


@dataclass
class Weibull(Distribution):
    alpha: Any
    beta: Any

    def cdf(self, y):
        return stats.weibull_min.cdf(y, self.alpha, scale=self.beta)

    def invcdf(self, quantile):
        return stats.weibull_min.ppf(quantile, self.alpha, scale=self.beta)
