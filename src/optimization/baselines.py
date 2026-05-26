import uuid
import numpy as np
import pandas as pd
from collections import defaultdict


def greedy_probe_selection(all_spikes, coverage_fraction=0.95):
    spike_impact = all_spikes.drop_duplicates(subset='spike_id').set_index('spike_id')['log_impact'].to_dict()
    probe_to_spikes = defaultdict(set)
    for _, row in all_spikes.iterrows():
        probe_to_spikes[row['device_id']].add((row['spike_id'], row['log_impact']))

    total_impact = sum(spike_impact.values())
    covered_spikes = set()
    covered_impact = 0
    selected_probes = []

    while covered_impact < coverage_fraction * total_impact:
        best_probe = None
        best_gain = 0
        best_new_spikes = set()

        for probe_id, spikes in probe_to_spikes.items():
            new_spikes = {s for s in spikes if s[0] not in covered_spikes}
            gain = sum(spike_impact[s[0]] for s in new_spikes)

            if gain > best_gain:
                best_gain = gain
                best_probe = probe_id
                best_new_spikes = new_spikes

        if best_probe is None:
            break

        selected_probes.append(best_probe)
        for spike_id, _ in best_new_spikes:
            covered_spikes.add(spike_id)
            covered_impact += spike_impact[spike_id]
        del probe_to_spikes[best_probe]

    return selected_probes, len(covered_spikes)


def random_probe_selection(all_spikes, target_coverage=0.95, n_trials=100):
    unique_spikes = all_spikes.drop_duplicates(subset='spike_id')
    spike_impact = unique_spikes.set_index('spike_id')['log_impact'].to_dict()
    total_impact = unique_spikes['log_impact'].sum()
    device_ids = all_spikes['device_id'].unique()

    probe_counts = []
    covered_spike_counts = []
    for _ in range(n_trials):
        np.random.shuffle(device_ids)
        selected = []
        covered_spikes = set()
        covered_impact = 0

        for dev in device_ids:
            new_spikes = set(all_spikes[all_spikes['device_id'] == dev]['spike_id']) - covered_spikes
            gain = sum(spike_impact[s] for s in new_spikes)
            covered_spikes.update(new_spikes)
            covered_impact += gain
            selected.append(dev)
            if covered_impact >= target_coverage * total_impact:
                break

        probe_counts.append(len(selected))
        covered_spike_counts.append(len(covered_spikes))

    return np.mean(probe_counts), np.std(probe_counts), np.mean(covered_spike_counts), np.std(covered_spike_counts)


def sort_by_impact_probe_selection(all_spikes, target_coverage=0.95):
    unique_spikes = all_spikes.drop_duplicates(subset='spike_id')
    spike_impact = unique_spikes.set_index('spike_id')['log_impact'].to_dict()
    total_impact = unique_spikes['log_impact'].sum()
    device_ids = all_spikes['device_id'].unique()
    probe_unique_impact = {
        dev: sum(spike_impact[s] for s in all_spikes[all_spikes['device_id'] == dev]['spike_id'].unique())
        for dev in device_ids
    }
    sorted_device_ids = sorted(device_ids, key=lambda x: probe_unique_impact[x], reverse=True)
    selected = []
    covered_spikes = set()
    covered_impact = 0
    for dev in sorted_device_ids:
        new_spikes = set(all_spikes[all_spikes['device_id'] == dev]['spike_id']) - covered_spikes
        gain = sum(spike_impact[s] for s in new_spikes)
        covered_spikes.update(new_spikes)
        covered_impact += gain
        selected.append(dev)
        if covered_impact >= target_coverage * total_impact:
            break
    return selected, len(covered_spikes)


def assign_spike_ids(
    all_spikes: pd.DataFrame,
    shared_spikes: pd.DataFrame,
    iou_threshold: float = 0.6,
    amplitude_similarity_threshold: float = 0.6,
) -> pd.DataFrame:
    """Assign a shared spike_id to overlapping spikes and compute impact columns.

    Spikes from different probes that overlap (IoU >= iou_threshold) with
    similar amplitude (amplitude_sim >= amplitude_similarity_threshold) are
    merged to the same spike_id so probe selection treats them as one event.
    Adds ``impact`` (amplitude × duration_hours) and ``log_impact`` columns.
    """
    all_spikes = all_spikes.copy()
    all_spikes['spike_id'] = [str(uuid.uuid4()) for _ in range(len(all_spikes))]
    all_spikes['impact'] = all_spikes['amplitude'] * all_spikes['duration'].dt.total_seconds() / 3600
    all_spikes['log_impact'] = np.log(all_spikes['impact'])
    df = all_spikes.copy()

    filtered_shared = shared_spikes[
        (shared_spikes['IoU'] >= iou_threshold) &
        (shared_spikes['amplitude_sim'] >= amplitude_similarity_threshold)
    ]

    spike_lookup = {
        (row.device_id, row.starts, row.ends, row.latency_target): row.spike_id
        for row in all_spikes.itertuples(index=False)
    }

    update_map = []
    for row in filtered_shared.itertuples(index=False):
        key1 = (row.device_1, row.start_time_1, row.end_time_1, row.latency_target)
        key2 = (row.device_2, row.start_time_2, row.end_time_2, row.latency_target)

        spike_id_1 = spike_lookup.get(key1)
        spike_id_2 = spike_lookup.get(key2)

        if spike_id_1 is None or spike_id_2 is None:
            continue

        min_id = min(spike_id_1, spike_id_2)
        update_map.append((row.device_1, row.start_time_1, row.end_time_1, row.latency_target, min_id))
        update_map.append((row.device_2, row.start_time_2, row.end_time_2, row.latency_target, min_id))

    updates_df = pd.DataFrame(update_map, columns=['device_id', 'starts', 'ends', 'latency_target', 'spike_id'])
    df = df.merge(updates_df, on=['device_id', 'starts', 'ends', 'latency_target'], how='left', suffixes=('', '_new'))
    df['spike_id'] = df['spike_id_new'].combine_first(df['spike_id'])
    return df.drop(columns=['spike_id_new'])


def produce_reduced_dataset(
    all_spikes: pd.DataFrame,
    shared_spikes: pd.DataFrame,
    coverage_fraction: float = 0.95,
    iou_threshold: float = 0.9,
    amplitude_similarity_threshold: float = 0.9,
) -> pd.DataFrame:
    """Return all_spikes filtered to the greedy-selected probes.

    Runs greedy probe selection independently per latency target and takes the
    union of selected device_ids across all targets, then filters all_spikes to
    those devices.
    """
    all_spikes_with_ids = assign_spike_ids(
        all_spikes, shared_spikes,
        iou_threshold=iou_threshold,
        amplitude_similarity_threshold=amplitude_similarity_threshold,
    )
    parts = []
    for target in all_spikes_with_ids['latency_target'].unique():
        target_spikes = all_spikes_with_ids[all_spikes_with_ids['latency_target'] == target]
        probes, _ = greedy_probe_selection(target_spikes, coverage_fraction=coverage_fraction)
        parts.append(
            all_spikes[
                (all_spikes['latency_target'] == target) &
                (all_spikes['device_id'].isin(probes))
            ]
        )
    return pd.concat(parts, ignore_index=True) if parts else all_spikes.iloc[:0].copy()
