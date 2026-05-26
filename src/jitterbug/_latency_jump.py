import numpy as np
from .preprocessing import calculate_jitter

class LatencyJumps:
    """
    A class to identify significant jumps in latency over time within a given dataset.

    Attributes
    ----------
    epoch : numpy.ndarray
        Array of epoch timestamps for the round-trip time (RTT) measurements.
    rtt : numpy.ndarray
        Array of round-trip time (RTT) measurements.
    change_points : numpy.ndarray
        Array of timestamps indicating potential change points in the latency data.

    Methods
    -------
    fit(threshold=0.5):
        Analyzes the RTT data to find significant jumps in latency.
    getLatencyJumps():
        Returns the significant latency jumps identified.
    """

    def __init__(self, epoch, rtt, change_points):
        self.epoch = np.array(epoch)
        self.rtt = np.array(rtt)
        self.change_points = np.array(change_points)
        self.jumps = []

    def fit(self, threshold=0.5):
        if len(self.rtt) == 0:
            return
        last_jump_index = 0
        last_dip_index = -1
        # Faster mode equivalent: most frequent value (numpy, no pd.Series overhead)
        vals, counts = np.unique(self.rtt, return_counts=True)
        baseline = float(vals[np.argmax(counts)])
        rtt_std = np.std(self.rtt)   # precompute — used inside loop

        # Precompute segment boundaries once via searchsorted (epoch is sorted)
        bounds = np.searchsorted(self.epoch, self.change_points, side='right')

        for i in range(1, len(self.change_points) - 1):
            rtt1 = self.rtt[bounds[i - 1]:bounds[i]]
            rtt2 = self.rtt[bounds[i]:bounds[i + 1]]

            if len(rtt1) == 0 or len(rtt2) == 0:
                continue

            rtt1_mean = np.mean(rtt1)
            rtt2_mean = np.mean(rtt2)
            mean_diff = rtt2_mean - rtt1_mean

            is_dip_recovery = (last_dip_index == i)
            jump = (mean_diff > threshold) and (rtt2_mean > baseline) and not is_dip_recovery

            jump_start = self.change_points[i]
            jump_end = self.change_points[i + 1]

            if not jump and np.abs(np.max(rtt2) - np.max(rtt1)) <= 1.5 * rtt_std and last_jump_index == i:
                jump = True

            if jump:
                last_jump_index = i + 1
            elif rtt2_mean < baseline:
                last_dip_index = i + 1

            duration = jump_end - jump_start
            amplitude = np.max(rtt2) - baseline
            self.jumps.append((jump_start, jump_end, jump, duration, amplitude))


    def getLatencyJumps(self):
        """
        Returns the significant latency jumps identified after analyzing the RTT data.

        Returns
        -------
        numpy.ndarray
            An array of tuples (jump_start, jump_end, jump, duration).
        """
        return np.array(self.jumps)