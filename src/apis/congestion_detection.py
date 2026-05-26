import duckdb
import os
import pandas as pd
from tqdm import tqdm
from .device_processor import DeviceProcessor
from .jitterbug_analyzer import *

class CongestionDetector:
    def __init__(self,
                 db_connection,
                 inference_method="jd",
                 cdp_algorithm="pelt",
                 latency_jump_threshold=DEFAULT_LATENCY_JUMP_THRESHOLD,
                 jitter_dispersion_threshold=DEFAULT_JITTER_DISPERSION_THRESHOLD,
                 moving_average_order=DEFAULT_MOVING_AVERAGE_ORDER,
                 moving_iqr_order=DEFAULT_MOVING_IQR_ORDER,
                 cpd_threshold=DEFAULT_CPD_THRESHOLD,
                 pelt_penalty=DEFAULT_PELT_PENALTY,
                 cusum_threshold=DEFAULT_CUSUM_THRESHOLD,
                 cusum_drift=DEFAULT_CUSUM_DRIFT,
                 window_size=48*60*60,
                 window_step_size=48*60*60):

        self.jitterbug_analyzer = JitterbugAnalyzer(
            inference_method=inference_method,
            cdp_algorithm=cdp_algorithm,
            latency_jump_threshold=latency_jump_threshold,
            jitter_dispersion_threshold=jitter_dispersion_threshold,
            moving_average_order=moving_average_order,
            moving_iqr_order=moving_iqr_order,
            cpd_threshold=cpd_threshold,
            pelt_penalty=pelt_penalty,
            cusum_threshold=cusum_threshold,
            cusum_drift=cusum_drift
        )
        self.device_list = None
        self.db_connection = db_connection
        self.window_size = window_size
        self.window_step_size = window_step_size
        self.common_columns = [
            'starts', 'ends', 'spike_type', 'device_id', 'isp_whois', 'latency_target',
            'window_start', 'window_end', 'window_size', 'window_step_size', 'duration', 'amplitude', 'jump_threshold'
        ]
        self.param_list = {"pelt": ["penalty"], "cusum": ["drift", "threshold"], "bcp": ["bcp_threshold"]}
        self.param_to_attr = {"penalty": "pelt_penalty", "drift": "cusum_drift", "threshold": "cusum_threshold", "bcp_threshold": "cdp_threshold"}

    def _find_deployment_period(self, device_id):
        query = f"SELECT MIN(time), MAX(time) FROM google WHERE device_id = '{device_id}'"
        result = self.db_connection.execute(query).fetchone()
        if not result:
            raise ValueError(f"No data found for device {device_id}.")
        return result[0], result[1]

    def _run_congestion_detection(self, device_id, latency_target, start_time, end_time):
        query = f"SELECT * FROM {latency_target} WHERE device_id = '{device_id}' and time BETWEEN '{start_time}' AND '{end_time}' AND isp_whois is NOT NULL"
        data = self.db_connection.execute(query).fetchdf()

        cpd_data = []

        for isp in data['isp_whois'].unique():
            device_data = data[data['isp_whois'] == isp]

            if device_data.empty:
                return pd.DataFrame({})

            processor = DeviceProcessor(device_data, self.jitterbug_analyzer)
            changepoints = processor.detect_congestion()
            changepoints['device_id'] = device_id
            changepoints['isp_whois'] = isp
            changepoints['latency_target'] = latency_target
            changepoints['window_start'] = start_time
            changepoints['window_end'] = end_time
            changepoints['window_size'] = self.window_size
            changepoints['window_step_size'] = self.window_step_size
            changepoints['jump_threshold'] = self.jitterbug_analyzer.latency_jump_threshold
            for col in self.param_list[self.jitterbug_analyzer.cdp_algorithm]:
                changepoints[col] = getattr(self.jitterbug_analyzer, self.param_to_attr[col])
            cpd_data.append(changepoints)

        if len(cpd_data) == 0:
            raise ValueError(f"No data found for device {device_id}.")
        
        cp = pd.concat(cpd_data, ignore_index=True)
        return cp


    def _deduplicate_changepoints(self, cp, merge_gap_seconds=60):
        """
        Deduplicate and merge adjacent changepoints for a device's congestion intervals.
        
        Parameters
        ----------
        cp : pd.DataFrame
            DataFrame of changepoints.
        merge_gap_seconds : int
            Maximum allowed time (in seconds) between adjacent changepoints to merge them.
            
        Returns
        -------
        pd.DataFrame
            Cleaned and merged changepoint DataFrame.
        """
        if cp.empty:
            return cp

        cp = cp.copy()
        cp['duration'] = cp['ends'] - cp['starts']
        cp = cp.sort_values(['device_id', 'latency_target', 'isp_whois', 'starts']).reset_index(drop=True)

        # Separate out 'Normal' spikes
        cp_normal = cp[cp['spike_type'] == 'Normal']
        cp_spike = cp[cp['spike_type'] != 'Normal']

        # Deduplicate overlapping spikes
        deduped = []
        last_row = None
        for _, row in cp_spike.iterrows():
            if last_row is None:
                last_row = row
                continue

            # Same device, target, ISP, spike_type
            same_context = (
                row['device_id'] == last_row['device_id'] and
                row['latency_target'] == last_row['latency_target'] and
                row['isp_whois'] == last_row['isp_whois']
            )

            time_gap = (row['starts'] - last_row['ends']).total_seconds()

            if same_context and time_gap <= merge_gap_seconds:
                # Extend the current spike
                last_row['ends'] = row['ends']
                last_row['duration'] = last_row['ends'] - last_row['starts']
                last_row['amplitude'] = max(last_row['amplitude'], row['amplitude'])
            else:
                deduped.append(last_row)
                last_row = row

        if last_row is not None:
            deduped.append(last_row)

        merged_cp_spike = pd.DataFrame(deduped)
        merged_cp = pd.concat([cp_normal, merged_cp_spike], ignore_index=True)
        return merged_cp


    def slide_window_and_detect_congestion(self, device_id, latency_target):
        """
        Slide a window over the data and detect congestion for a given device and target.
        :param device_id: The ID of the device to process.
        :param latency_target: The target latency to analyze.
        :return: A DataFrame with the detected changepoints.
        """
        deployment_start, deployment_end = map(pd.to_datetime, self._find_deployment_period(device_id))
        start_time = deployment_start
        end_time = deployment_start + pd.Timedelta(seconds=self.window_size)
        all_changepoints = []

        while end_time <= deployment_end:
            try:
                changepoints = self._run_congestion_detection(device_id, latency_target, start_time, end_time)
            except ValueError:
                changepoints = pd.DataFrame()
            if not changepoints.empty:
                all_changepoints.append(changepoints)
            start_time += pd.Timedelta(seconds=self.window_step_size)
            end_time += pd.Timedelta(seconds=self.window_step_size)

        if not all_changepoints:
            return pd.DataFrame()

        cp = pd.concat(all_changepoints, ignore_index=True)
        # Deduplicate overlaps
        cp = cp[cp['amplitude'] > cp['amplitude'].mean()]
        cp = self._deduplicate_changepoints(cp)

        return cp