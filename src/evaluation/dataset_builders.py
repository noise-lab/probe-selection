import numpy as np
import pandas as pd

from ..optimization.baselines import assign_spike_ids, greedy_probe_selection


def get_greedy_devices(
    all_spikes_with_ids: pd.DataFrame,
    coverage_fraction: float,
) -> set[str]:
    """Run greedy probe selection per latency target and return the union of
    selected device_ids across all targets."""
    selected = set()
    for target in all_spikes_with_ids['latency_target'].unique():
        target_spikes = all_spikes_with_ids[
            all_spikes_with_ids['latency_target'] == target
        ]
        probes, _ = greedy_probe_selection(target_spikes, coverage_fraction=coverage_fraction)
        selected.update(probes)
    return selected


def get_random_devices(
    all_device_ids: np.ndarray,
    n_devices: int,
    rng: np.random.Generator,
) -> set[str]:
    """Select n_devices uniformly at random (without replacement)."""
    chosen = rng.choice(all_device_ids, size=min(n_devices, len(all_device_ids)), replace=False)
    return set(chosen)


def label_anomalous_rows(
    timeseries: pd.DataFrame,
    all_spikes: pd.DataFrame,
) -> pd.Series:
    """Return a boolean Series (aligned with timeseries index) that is True
    where the row falls within any spike window for that device.

    Uses per-device vectorized interval containment via searchsorted.
    """
    mask = pd.Series(False, index=timeseries.index)

    for device_id, device_spikes in all_spikes.groupby('device_id'):
        device_ts = timeseries[timeseries['device_id'] == device_id]
        if device_ts.empty:
            continue

        starts = device_spikes['starts'].values
        ends = device_spikes['ends'].values

        order = np.argsort(starts)
        starts = starts[order]
        ends = ends[order]

        merged_starts = [starts[0]]
        merged_ends = [ends[0]]
        for s, e in zip(starts[1:], ends[1:]):
            if s <= merged_ends[-1]:
                merged_ends[-1] = max(merged_ends[-1], e)
            else:
                merged_starts.append(s)
                merged_ends.append(e)
        merged_starts = np.array(merged_starts)
        merged_ends = np.array(merged_ends)

        times = device_ts['time'].values
        idx = np.searchsorted(merged_starts, times, side='right') - 1
        in_range = (idx >= 0) & (times <= merged_ends[np.clip(idx, 0, len(merged_ends) - 1)])
        mask.loc[device_ts.index] = in_range

    return mask


def build_augmented_timeseries(
    timeseries: pd.DataFrame,
    is_anomalous: pd.Series,
    anomaly_frac: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a dataset where anomaly_frac of the output rows are anomalous.

    All anomalous rows are kept. Baseline rows are subsampled so that
    n_anomalous / (n_anomalous + n_baseline_sampled) = anomaly_frac.
    """
    anomalous = timeseries[is_anomalous]
    baseline = timeseries[~is_anomalous]

    n_anomalous = len(anomalous)
    n_baseline_target = int(n_anomalous * (1 - anomaly_frac) / anomaly_frac)
    n_baseline_target = min(n_baseline_target, len(baseline))

    baseline_sample = baseline.sample(n=n_baseline_target, random_state=rng.integers(2**31))
    return pd.concat([anomalous, baseline_sample], ignore_index=True)


def build_anomaly_only_timeseries(
    timeseries: pd.DataFrame,
    is_anomalous: pd.Series,
) -> pd.DataFrame:
    """Keep only timeseries rows within detected spike windows."""
    return timeseries[is_anomalous].reset_index(drop=True)
