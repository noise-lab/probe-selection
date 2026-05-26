import os
import pandas as pd
from ..jitterbug.bcp import BCP
from ..jitterbug._jitter import compute_jitter_dispersion, compute_ks_test
from detecta import detect_cusum
from ..jitterbug._latency_jump import LatencyJumps
from ..jitterbug.cong_inference import CongestionInference
import numpy as np
import ruptures as rpt
from ruptures.exceptions import BadSegmentationParameters

# Constants
DEFAULT_MOVING_IQR_ORDER = 4
DEFAULT_MOVING_AVERAGE_ORDER = 6
DEFAULT_CPD_THRESHOLD = 0.25
DEFAULT_JITTER_DISPERSION_THRESHOLD = 0.25
DEFAULT_LATENCY_JUMP_THRESHOLD = 0.5
DEFAULT_PELT_PENALTY = 0.5
DEFAULT_CUSUM_THRESHOLD = 1
DEFAULT_CUSUM_DRIFT = 0.001
CONGESTION_INFERENCE_METHODS = ["ks", "jd", "lj_only"]
CDP_ALGORITHMS = ["bcp", "cusum", "pelt"]

class JitterbugAnalyzer:
    def __init__(self,
                 inference_method="jd",
                 cdp_algorithm="pelt",
                 latency_jump_threshold=DEFAULT_LATENCY_JUMP_THRESHOLD,
                 jitter_dispersion_threshold=DEFAULT_JITTER_DISPERSION_THRESHOLD,
                 moving_average_order=DEFAULT_MOVING_AVERAGE_ORDER,
                 moving_iqr_order=DEFAULT_MOVING_IQR_ORDER,
                 cpd_threshold=DEFAULT_CPD_THRESHOLD,
                 pelt_penalty=DEFAULT_PELT_PENALTY,
                 cusum_threshold=DEFAULT_CUSUM_THRESHOLD,
                 cusum_drift=DEFAULT_CUSUM_DRIFT):
        if inference_method not in CONGESTION_INFERENCE_METHODS:
            raise ValueError(f"Inference method must be one of {CONGESTION_INFERENCE_METHODS}")
        if cdp_algorithm not in CDP_ALGORITHMS:
            raise ValueError(f"CDP algorithm must be one of {CDP_ALGORITHMS}")

        self.inference_method = inference_method
        self.cdp_algorithm = cdp_algorithm
        self.latency_jump_threshold = latency_jump_threshold
        self.jitter_dispersion_threshold = jitter_dispersion_threshold
        self.moving_average_order = self._check_even_value(moving_average_order)
        self.moving_iqr_order = self._check_positive_value(moving_iqr_order)
        self.cpd_threshold = cpd_threshold
        self.pelt_penalty = pelt_penalty
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift

    def _check_positive_value(self, x):
        x = int(x)
        if x < 0:
            raise ValueError("The order must be a positive integer.")
        return x

    def _check_even_value(self, x):
        x = self._check_positive_value(x)
        if x % 2 != 0:
            raise ValueError("Moving average order must be a positive even integer.")
        return x

    def load_rtts(self, data, time_col="time", rtt_col="latency_min", mdev_col="latency_mdev"):
        # Compute derived columns without copying the entire input DataFrame
        datetime_series = pd.to_datetime(data[time_col], utc=True)
        epoch_series = (datetime_series - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()

        # Build full-resolution RTT DataFrame
        rtts_df = pd.DataFrame({
            "epoch":    epoch_series.values,
            "values":   data[rtt_col].values,
            "datetime": datetime_series,
        })

        # Aggregate min RTT every 15 minutes; drop empty bins (NaN) to avoid propagating
        # missing-bin placeholders into PELT and downstream numpy reductions.
        mins_df = (
            rtts_df.groupby(pd.Grouper(key="datetime", freq="15min"))[["epoch", "values"]]
            .min()
            .reset_index()
            .dropna(subset=["values"])
        )

        # Build RTT-stdev DataFrame: use mdev_col if available, else approximate from bin std-dev
        if mdev_col is not None and mdev_col in data.columns:
            mdev_df = pd.DataFrame({
                "epoch":    epoch_series.values,
                "values":   data[mdev_col].values,
                "datetime": datetime_series,
            })
        else:
            mdev_agg = (
                rtts_df.set_index("datetime")
                .resample("15min")["values"]
                .std()
                .fillna(0)
                .reset_index()
            )
            epoch_map = mins_df.set_index("datetime")["epoch"]
            mdev_agg["epoch"] = mdev_agg["datetime"].map(epoch_map)
            mdev_df = mdev_agg[["epoch", "values", "datetime"]]

        return rtts_df, mins_df, mdev_df

    def _detect_bcp_change_points(self, epoch, values):
        """
        Detect change points in the given DataFrame using the BCP algorithm.

        Parameters
        ----------
        epoch : list or numpy array
            Timestamps of the RTT measurements.
        values : list or numpy array
            Corresponding RTT values.

        Returns
        -------
        list
            List of detected change points.
        """

        # Initialize and fit the BCP model
        cp_detector = BCP(epoch, values, cpd_threshold=self.cpd_threshold)
        cp_detector.fit()

        # Get the detected change points
        change_points = cp_detector.getChangePoints()
        return change_points

    def _detect_cusum_change_points(self, epoch, values):
        """
        Detect change points in the given DataFrame using the CUSUM algorithm.

        Parameters
        ----------
        epoch : list or numpy array
            Timestamps of the RTT measurements.
        values : list or numpy array
            Corresponding RTT values.

        Returns
        -------
        list
            List of detected change points.
        """
        sdev = np.std(values)
        ta, tai, taf, amp = detect_cusum(values, threshold=self.cusum_threshold * sdev, drift=self.cusum_drift * sdev, ending=True, show=False, ax=None)

        cp_boundaries = sorted(set(list(tai) + list(taf)))
        change_points = []
        for i in range(len(cp_boundaries)):
            if cp_boundaries[i] >= 0:
                change_points.append(epoch[cp_boundaries[i]])
        return change_points

    def _detect_pelt_change_points(self, epoch, values):
        """
        Detect change points in the given DataFrame using the PELT algorithm.

        Parameters
        ----------
        epoch : list or numpy array
            Timestamps of the RTT measurements.
        values : list or numpy array
            Corresponding RTT values.

        Returns
        -------
        list
            List of detected change points.
        """
        algo = rpt.Pelt(model="l2").fit(values)
        change_indices = algo.predict(pen=self.pelt_penalty)
        change_indices = [i for i in change_indices if i < len(values)]
        change_points = [epoch[i] for i in change_indices]
        return change_points

    def _detect_jumps(self, epoch, values, changepoints):
        """
        Detect jumps in the given DataFrame using the LatencyJumps algorithm.

        Parameters
        ----------
        epoch : list or numpy array
            Timestamps of the RTT measurements.
        values : list or numpy array
            Corresponding RTT values.
        changepoints : list
            List of detected change points.

        Returns
        -------
        list
            List of detected jumps.
        """
        latency_jumps_detector = LatencyJumps(epoch, values, changepoints)
        latency_jumps_detector.fit(self.latency_jump_threshold)
        return latency_jumps_detector.getLatencyJumps()

    def analyze(self, rtts_df, mins_df, mdev_df):
        epoch_rtt = rtts_df["epoch"].values
        rtt = rtts_df["values"].values
        epoch_mins = mins_df["epoch"].values
        mins = mins_df["values"].values
        epoch_mdev = mdev_df["epoch"].values
        mdev = mdev_df["values"].values

        # ---- CHOOSE CHANGEPOINT DETECTION METHOD ----
        if self.cdp_algorithm == "bcp":
            min_change_points = self._detect_bcp_change_points(epoch_mins, mins)

        elif self.cdp_algorithm == "cusum":
            min_change_points = self._detect_cusum_change_points(epoch_mins, mins)

        elif self.cdp_algorithm == "pelt":
            try:
                min_change_points = self._detect_pelt_change_points(epoch_mins, mins)
            except BadSegmentationParameters as e:
                min_change_points = []
        else:
            raise NotImplementedError(f"CDP algorithm {self.cdp_algorithm} is not yet supported.")

        if len(min_change_points) == 0:
            return pd.DataFrame(columns=["starts", "ends", "spike_type", "duration", "amplitude"])

        min_jumps = np.array([])

        if min_change_points:
            min_jumps = self._detect_jumps(epoch_mins, mins, min_change_points)

        try:
            if self.inference_method == "lj_only":
                # Jitter dispersion only distinguishes "Congestion" from "Path Change" —
                # both map to label=1, so skip it and classify directly from amplitude.
                if len(min_jumps) == 0:
                    results = pd.DataFrame(columns=["starts", "ends", "spike_type", "duration", "amplitude"])
                else:
                    jumps = np.array(min_jumps)
                    spike_types = np.where(jumps[:, 4].astype(float) > 0.5, "Path Change", "Normal")
                    df_jumps = pd.DataFrame({
                        "starts":    pd.to_datetime(jumps[:, 0].astype(float) * 1e9),
                        "ends":      pd.to_datetime(jumps[:, 1].astype(float) * 1e9),
                        "spike_type": spike_types,
                        "duration":  jumps[:, 3].astype(float),
                        "amplitude": jumps[:, 4].astype(float),
                    })
                    results = df_jumps
            else:
                if self.inference_method == "jd":
                    jitter_analysis = compute_jitter_dispersion(min_change_points, epoch_mins, mins,
                                                                self.moving_iqr_order, self.moving_average_order,
                                                                self.jitter_dispersion_threshold)
                elif self.inference_method == "ks":
                    jitter_analysis = compute_ks_test(min_change_points, epoch_rtt, rtt)

                congestion_inference = CongestionInference(min_jumps, jitter_analysis)
                congestion_inference.fit()
                results = congestion_inference.getInferences()

        except Exception:
            results = pd.DataFrame(columns=["starts", "ends", "spike_type", "duration", "amplitude"])

        return results