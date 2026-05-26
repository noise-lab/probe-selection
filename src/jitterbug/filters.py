import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def moving_average(x, w):
    """
    Compute the moving average of a given data array.

    Parameters
    ----------
    x : numpy array
        The input data array to compute the moving average on.
    w : int
        The window size for the moving average.

    Returns
    -------
    numpy array
        The moving average of the input data array over the specified window size.

    Notes
    -----
    This function computes the moving average using a convolution approach, which
    can be more efficient than a straightforward implementation for large datasets.
    The 'valid' mode in `np.convolve` ensures that the returned moving average
    only contains values where the window is fully within the input array.
    """
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_iqr_filter_symmetric(x, k):
    """
    Calculate the interquartile range (IQR) of the data, using a symmetric window.

    Parameters
    ----------
    x : numpy array
        The input data array to compute the IQR on.
    k : int
        The half window size. The total window size will be `2*k+1`, centered around each point.

    Returns
    -------
    numpy array
        The IQR values of the input data array, computed over a window of size `2*k+1` for each point.
    """
    window_len = 2 * k + 1
    if len(x) < window_len:
        return np.empty(0)
    windows = sliding_window_view(x, window_len)        # shape: (len(x)-2k, 2k+1)
    q1, q3 = np.percentile(windows, [25, 75], axis=1)  # single vectorised C call
    return q3 - q1
