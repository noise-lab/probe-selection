from .filters import *

def calculate_jitter(epoch, rtt):
    """
    Calculate the jitter from the given epoch and round-trip time (RTT) measurements.

    Parameters
    ----------
    epoch : list or numpy array
        Timestamps of the RTT measurements.
    rtt : list or numpy array
        Corresponding RTT values.

    Returns
    -------
    tuple
        A tuple containing two elements: (1) the time differences between consecutive epochs, and
        (2) the jitter values calculated as the difference between consecutive RTT measurements.

    Raises
    ------
    Exception
        If the epoch and RTT arrays have different lengths or contain fewer than two samples.
    """
    if len(epoch) != len(rtt):
        raise Exception("Epochs and RTTs have different lengths.")

    if len(rtt) <= 1:
        raise Exception("Insufficient samples (fewer than 2) to compute jitter.")

    t = epoch[1:]
    x = rtt[1:] - rtt[:-1]
    
    return t, x

def calculate_jitter_dispersion(epoch, mins, iqr_order, ma_order, jitter_dispersion_threshold=0.25):
    """
    Calculate the jitter dispersion using moving average and interquartile range filtering on minimum RTT data.

    Parameters
    ----------
    epoch : list or numpy array
        Timestamps of the minimum RTT measurements.
    mins : list or numpy array
        Minimum RTT values for each timestamp.
    iqr_order : int
        Order of the interquartile range filter.
    ma_order : int
        Order of the moving average filter.

    Returns
    -------
    tuple
        A tuple containing two elements: (1) the modified timestamps after applying the filters, and
        (2) the jitter dispersion values.

    Raises
    ------
    Exception
        If the epoch and mins arrays have different lengths or if there's an issue with the filter orders being odd.
    """
    if len(epoch) != len(mins):
        raise Exception("Epochs and mins have different lengths.")

    kl = iqr_order + int(ma_order / 2)

    jmin_t, jmin_vals = calculate_jitter(epoch, mins)
    
    t = jmin_t[kl:-(kl - 1)]
    x = moving_average(moving_iqr_filter_symmetric(jmin_vals, iqr_order), ma_order)

    return t, x