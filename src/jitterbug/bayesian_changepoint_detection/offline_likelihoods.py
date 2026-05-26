from abc import ABC, abstractmethod
import functools

import numpy as np
from scipy.special import gammaln


def _dynamic_programming(f, *args, **kwargs):
    if f.data is None:
        f.data = args[1]

    if not np.array_equal(f.data, args[1]):
        f.cache = {}
        f.data = args[1]

    try:
        f.cache[args[2:4]]
    except KeyError:
        f.cache[args[2:4]] = f(*args, **kwargs)
    return f.cache[args[2:4]]


def dynamic_programming(f):
    f.cache = {}
    f.data = None
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return _dynamic_programming(f, *args, **kwargs)
    return wrapper


class BaseLikelihood(ABC):
    @abstractmethod
    def pdf(self, data: np.array, t: int, s: int):
        raise NotImplementedError(
            "PDF is not defined. Please define in separate class and override this function."
        )


class StudentT(BaseLikelihood):
    @dynamic_programming
    def pdf(self, data: np.ndarray, t: int, s: int):
        """StudentT predictive likelihood for offline Bayesian changepoint detection.

        Uses the update approach from https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
        (page 8, equation 89).
        """
        s += 1
        n = s - t

        mean = data[t:s].sum(0) / n
        muT = (n * mean) / (1 + n)
        nuT = 1 + n
        alphaT = 1 + n / 2

        betaT = (
            1
            + 0.5 * ((data[t:s] - mean) ** 2).sum(0)
            + ((n) / (1 + n)) * (mean ** 2 / 2)
        )
        scale = (betaT * (nuT + 1)) / (alphaT * nuT)

        prob = np.sum(np.log(1 + (data[t:s] - muT) ** 2 / (nuT * scale)))
        lgA = (
            gammaln((nuT + 1) / 2)
            - np.log(np.sqrt(np.pi * nuT * scale))
            - gammaln(nuT / 2)
        )

        return np.sum(n * lgA - (nuT + 1) / 2 * prob)
