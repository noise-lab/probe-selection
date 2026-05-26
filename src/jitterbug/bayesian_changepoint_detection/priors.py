import numpy as np


def const_prior(t, p: float = 0.25):
    """Constant log-prior for every changepoint candidate."""
    return np.log(p)
