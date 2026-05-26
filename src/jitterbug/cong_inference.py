import numpy as np
import pandas as pd

class CongestionInference:
    """
    A class to infer network congestion events based on latency jumps and jitter analysis.

    Attributes
    ----------
    latency_jumps : numpy array
        Array containing the latency jump analysis results.
    jitter_analysis : numpy array
        Array containing the jitter analysis results.
    congestion : bool
        Indicates whether a congestion event is currently inferred.
    congestion_inferences : list
        List of tuples representing the inferred congestion states over time.

    Methods
    -------
    fit():
        Processes the latency jumps and jitter analysis to infer congestion events.
    getInferences():
        Returns the inferred congestion events as an array.
    """

    def __init__(self, latency_jumps, jitter_analysis):
        """
        Initializes the CongestionInference class with latency jump and jitter analysis data.

        Parameters
        ----------
        latency_jumps : numpy array
            Array containing the latency jump analysis results.
        jitter_analysis : numpy array
            Array containing the jitter analysis results.
        """
        self.latency_jumps = latency_jumps
        self.jitter_analysis = jitter_analysis
        self.congestion = False
        self.congestion_inferences = []

    def fit(self):
        """
        Processes the latency jumps and jitter analysis to infer congestion events.
        """
        self.congestion_inferences = []
        for jump, jitter in zip(self.latency_jumps, self.jitter_analysis):
            if jump[-1] > 0.5 and jitter[2]:
                state = "Congestion"
            elif jump[-1] > 0.5:
                state = "Path Change"
            else:
                state = "Normal"
 
            self.congestion_inferences.append((jump[0], jump[1], state, jump[3], jump[4]))

    def getInferences(self):
        """
        Returns the inferred congestion events.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the inferred congestion events with start and end times.
        -----------
        """
        if len(self.congestion_inferences) == 0:
            return pd.DataFrame(columns=["starts", "ends", "spike_type", "duration", "amplitude"])
        df = pd.DataFrame(self.congestion_inferences, columns=["starts", "ends", "spike_type", "duration", "amplitude"])
        df["starts"] = pd.to_datetime(df["starts"] * 1e9)
        df["ends"] = pd.to_datetime(df["ends"] * 1e9)
        return df
