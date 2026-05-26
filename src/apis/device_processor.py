import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .jitterbug_analyzer import JitterbugAnalyzer, DEFAULT_MOVING_IQR_ORDER, DEFAULT_MOVING_AVERAGE_ORDER
from ..jitterbug import preprocessing as preproc

class DeviceProcessor:
    def __init__(self, data, jitterbug_analyzer):
        """
        Initialize the LatencyProcessor with the given data.
        :param data: The data to be processed for a single device. It should be in the same format as the latency tables.
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame.")

        if data['device_id'].nunique() == 0:
            raise ValueError("Data must contain at least one device ID.")
        elif data['device_id'].nunique() > 1:
            raise ValueError("Data must contain measurements from a single device.")

        self.data = data
        self.processed_data = data.copy()
        self.jitterbug_analyzer = jitterbug_analyzer

    def remove_unlabelled_measurements(self):
        """
            Remove measurements with NaN ISP or other labels.
        """
        self.processed_data = self.processed_data.dropna(subset=["isp", "zipcode"])

    def remove_noisy_measurements(self, isp_significance_threshold=0.05):
        """
            Remove measurements that were conducted in-lab before devices were shipped.
        """
        isp_proportion = self.processed_data['isp_whois'].value_counts(normalize=True).reset_index()
        isp_proportion.columns = ['isp', 'proportion']
        significant_isps = isp_proportion[isp_proportion['proportion'] > isp_significance_threshold]['isp'].tolist()
        self.processed_data = self.processed_data[self.processed_data['isp_whois'].isin(significant_isps)]

    def remove_isolated_spikes(self):
        """
            Remove isolated spikes in the data. We define an isolated spike as one that's at least 5 ms more the previous and next value.
            :return: A DataFrame with the cleaned data.
        """
        self.processed_data['latency_diff_prev'] = self.processed_data['latency_min'].diff()
        self.processed_data['latency_diff_next'] = self.processed_data['latency_min'].diff(-1)
        self.processed_data['is_isolated_spike'] = (
            (self.processed_data['latency_diff_prev'] > 5) &
            (self.processed_data['latency_diff_next'] > 5)
        )
        self.processed_data = self.processed_data[~self.processed_data['is_isolated_spike']]
        self.processed_data = self.processed_data.drop(columns=['latency_diff_prev', 'latency_diff_next', 'is_isolated_spike'])

    def clean_data(self):
        """
            Clean the data by removing unlabelled and noisy measurements.
        """
        self.remove_unlabelled_measurements()
        self.remove_noisy_measurements()
        # self.remove_isolated_spikes()


    def detect_congestion(self):
        """
            Detect changepoints in the latency data using the Jitterbug library.
            :return: A DataFrame with the detected changepoints.
        """
        self.clean_data()

        if self.processed_data.empty:
            raise ValueError("No data to process.")

        # Load the RTTs
        rtts_df, mins_df, mdev_df = self.jitterbug_analyzer.load_rtts(self.processed_data)
        # Detect changepoints
        changepoints = self.jitterbug_analyzer.analyze(rtts_df, mins_df, mdev_df)

        return changepoints

    def plot_time_series(self, metric: str, start=None, end=None, figsize=(20, 3), binsize=None, agg_fn=None):
        """
            Plot the time series of latency data for the device.
            :param metric: The aggregation metric (only 'latency_avg', 'latency_max', 'latency_min', and 'latency_mdev' are available).
            :param start: The start time for the query.
            :param end: The end time for the query.
        """

        self.clean_data()

        if start:
            self.processed_data = self.processed_data[self.processed_data['time'] >= start]

        if end:
            self.processed_data = self.processed_data[self.processed_data['time'] <= end]

        if self.processed_data.empty:
            raise ValueError("No data found for the given time range.")

        if metric not in ["latency_avg", "latency_max", "latency_min", "latency_mdev"]:
            raise ValueError("Invalid metric. Only 'latency_avg', 'latency_max', 'latency_min', and 'latency_mdev' are available.")

        # Plot the data
        fig, ax = plt.subplots(figsize=figsize)
        ts_data = self.processed_data.copy()
        if binsize is not None and agg_fn is not None:
            ts_data = ts_data.set_index('time')
            ts_data = ts_data.resample(binsize)[metric].agg(agg_fn).reset_index()

        sns.lineplot(data=ts_data, x='time', y=metric, color='blue', ax=ax)
        ax.set_title(f"{metric} for {self.processed_data['device_short_id'].iloc[0]}\nAggregated by {agg_fn if agg_fn else 'raw values'}")
        ax.set_xlabel(f"Time [{'5min' if binsize is None else binsize} bins]")
        ax.set_ylabel('Latency [ms]')
        plt.xticks(rotation=45)
        return fig, ax, ts_data

    def plot_congestion_detection_results(self, changepoints, start=None, end=None, figsize=(20, 9), min_only=True):
        if not hasattr(self, "processed_data") or self.processed_data is None:
            raise ValueError("processed_data is not initialized.")

        # Filter data
        df = self.processed_data.copy()
        if start:
            df = df[df['time'] >= pd.to_datetime(start)]
        if end:
            df = df[df['time'] <= pd.to_datetime(end)]
        if df.empty:
            raise ValueError("No data found for the given time range.")

        # Setup subplots
        if min_only:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        else:
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            ax0, ax1, ax2, ax3 = axes  # unpack axes

        # Plot aggregated RTT
        min_data = df.set_index('time').resample('15min')['latency_min'].min().reset_index()
        sns.lineplot(data=min_data, x='time', y='latency_min', color='blue', ax=ax1)
        sns.lineplot(data=min_data, x='time', y='latency_min', ax=ax1)
        ax1.set_ylabel('Aggregated min-RTT [ms]')
        ax1.set_xlabel('Time')
        ax1.grid()

        if not min_only:
            # Plot raw RTT
            sns.lineplot(data=df, x='time', y='latency_min', color='blue', ax=ax0)
            ax0.set_ylabel('Raw RTT [ms]')
            ax0.grid()

            # Plot jitter
            jitter_t, jitter = preproc.calculate_jitter(df['time'].astype(int) / 1e6, df['latency_min'].values)
            jitter_df = pd.DataFrame({'epoch': jitter_t, 'jitter': jitter})
            jitter_df['epoch'] = pd.to_datetime(jitter_df['epoch'], unit='s')
            sns.lineplot(data=jitter_df, x='epoch', y='jitter', color='blue', ax=ax2)
            ax2.set_ylabel('Jitter [ms]')
            ax2.grid()

            # Plot jitter dispersion
            epoch = min_data['time'].astype(int) / 1e6
            jd_t, jd = preproc.calculate_jitter_dispersion(epoch, min_data['latency_min'].values, DEFAULT_MOVING_IQR_ORDER, DEFAULT_MOVING_AVERAGE_ORDER)
            jd_df = pd.DataFrame({'epoch': jd_t, 'jitter_dispersion': jd})
            jd_df['epoch'] = pd.to_datetime(jd_df['epoch'], unit='s')
            sns.lineplot(data=jd_df, x='epoch', y='jitter_dispersion', color='blue', ax=ax3)
            ax3.set_ylabel('Jitter Dispersion [ms]')
            ax3.grid()

        # Highlight changepoints
        for _, cp in changepoints.iterrows():
            if pd.isnull(cp['starts']) or pd.isnull(cp['ends']):
                continue
            color = 'red' if cp['spike_type'] == "Path Change" else ('red' if cp['spike_type'] == "Congestion" else 'white')
            start, end = cp['starts'], cp['ends']
            width = end - start
            if not min_only:
                ax0.add_patch(plt.Rectangle((start, 0), width, ax0.get_ylim()[1], color=color, alpha=0.2))
                ax2.add_patch(plt.Rectangle((start, 0), width, ax2.get_ylim()[1], color=color, alpha=0.2))
            ax1.add_patch(plt.Rectangle((start, 0), width, ax1.get_ylim()[1], color=color, alpha=0.2))

        plt.tight_layout()
        plt.show()
