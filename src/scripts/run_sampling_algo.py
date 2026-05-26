import os

import duckdb
import pandas as pd

from ..config import DB_PATH
from ..optimization.baselines import produce_reduced_dataset

def _fetch_chicago_devices(con: duckdb.DuckDBPyConnection) -> list:
    devices = con.execute(
        "SELECT distinct device_id FROM chicago"
        " where isp_whois is NOT NULL AND zipcode is NOT NULL"
        " AND topic IN ('chicago', 'beta') AND time < '2023-01-01'"
    ).fetchall()
    return [device[0] for device in devices]

def _fetch_zipcode_mapping(con: duckdb.DuckDBPyConnection) -> dict:
    zipcode_df = con.execute("SELECT * FROM zipcode_map").fetchdf()
    return {row['device_id']: row['zipcode'] for _, row in zipcode_df.iterrows() if row['zipcode'] is not None}

def _fetch_all_spikes(con: duckdb.DuckDBPyConnection, device_list: list) -> pd.DataFrame:
    all_spikes = con.execute(f"""
        SELECT *
        FROM changepoints_pelt
        WHERE device_id IN {tuple(device_list)}
        AND latency_target != 'cloudflare'
        AND starts < '2023-01-01'
        AND ends < '2023-01-01'
    """).fetchdf()
    return all_spikes[
        all_spikes['latency_target'].isin(
            ['denver', 'atlanta', 'seattle', 'chicago', 'johannesburg',
             'stockholm', 'sydney', 'tunis', 'paris']
        )
    ]

def _fetch_raw_timeseries(con: duckdb.DuckDBPyConnection, device_list: list) -> pd.DataFrame:
    return con.execute(f"""
        SELECT *
        FROM chicago
        WHERE device_id IN {tuple(device_list)}
        AND topic IN ('chicago', 'beta')
        AND time < '2023-01-01'
    """).fetchdf()

def _fetch_shared_spikes(con: duckdb.DuckDBPyConnection, device_list: list) -> pd.DataFrame:
    shared_spikes = con.execute(f"""
        SELECT *
        FROM event_overlaps
        WHERE device_1 IN {tuple(device_list)}
        AND device_2 IN {tuple(device_list)}
        AND start_time_1 < '2023-01-01'
        AND end_time_1 < '2023-01-01'
        AND latency_target != 'cloudflare'
    """).fetchdf()
    zipcode_dict = _fetch_zipcode_mapping(con)
    shared_spikes['device_short_1'] = shared_spikes['device_1'].str[-4:]
    shared_spikes['device_short_2'] = shared_spikes['device_2'].str[-4:]
    shared_spikes['amplitude_diff'] = abs(shared_spikes['amplitude_1'] - shared_spikes['amplitude_2'])
    shared_spikes['duration_diff'] = abs(
        shared_spikes['spike_duration_1'].dt.total_seconds() -
        shared_spikes['spike_duration_2'].dt.total_seconds()
    )
    shared_spikes['zipcode_1'] = shared_spikes['device_1'].map(zipcode_dict)
    shared_spikes['zipcode_2'] = shared_spikes['device_2'].map(zipcode_dict)
    shared_spikes = shared_spikes[
        shared_spikes['latency_target'].isin(
            ['denver', 'atlanta', 'seattle', 'chicago', 'johannesburg',
             'stockholm', 'sydney', 'tunis', 'paris']
        )
    ]
    shared_spikes['duration_hours_1'] = shared_spikes['spike_duration_1'].dt.total_seconds() / 3600
    shared_spikes['duration_hours_2'] = shared_spikes['spike_duration_2'].dt.total_seconds() / 3600
    shared_spikes['impact_1'] = shared_spikes['amplitude_1'] * shared_spikes['duration_hours_1']
    shared_spikes['impact_2'] = shared_spikes['amplitude_2'] * shared_spikes['duration_hours_2']
    shared_spikes['impact_sim'] = (
        shared_spikes[['impact_1', 'impact_2']].min(axis=1) /
        shared_spikes[['impact_1', 'impact_2']].max(axis=1)
    )
    shared_spikes['amplitude_sim'] = (
        shared_spikes[['amplitude_1', 'amplitude_2']].min(axis=1) /
        shared_spikes[['amplitude_1', 'amplitude_2']].max(axis=1)
    )
    return shared_spikes


def run():
    OUTPUT_DIR = os.environ["OUTPUT_DIR"]
    netrics_all_spikes_path = f"{OUTPUT_DIR}/netrics_all_spikes_cache.parquet"
    netrics_shared_spikes_path = f"{OUTPUT_DIR}/netrics_shared_spikes_cache.parquet"
    netrics_raw_timeseries_path = f"{OUTPUT_DIR}/netrics_raw_timeseries_cache.parquet"

    if (os.path.exists(netrics_all_spikes_path) and
            os.path.exists(netrics_shared_spikes_path) and
            os.path.exists(netrics_raw_timeseries_path)):
        print("Loading cached Netrics spikes and time series...")
        netrics_all_spikes = pd.read_parquet(netrics_all_spikes_path)
        netrics_shared_spikes = pd.read_parquet(netrics_shared_spikes_path)
        netrics_raw_timeseries = pd.read_parquet(netrics_raw_timeseries_path)
    else:
        print("Fetching Netrics spikes and time series from database...")
        con = duckdb.connect(database=str(DB_PATH), read_only=True)
        device_list = _fetch_chicago_devices(con)
        netrics_all_spikes = _fetch_all_spikes(con, device_list)
        netrics_shared_spikes = _fetch_shared_spikes(con, device_list)
        netrics_raw_timeseries = _fetch_raw_timeseries(con, device_list)
        con.close()

        netrics_all_spikes['duration'] = pd.to_timedelta(netrics_all_spikes['duration'])
        netrics_all_spikes.to_parquet(netrics_all_spikes_path)
        netrics_shared_spikes.to_parquet(netrics_shared_spikes_path)
        netrics_raw_timeseries.to_parquet(netrics_raw_timeseries_path)

    reduced = produce_reduced_dataset(netrics_all_spikes, netrics_shared_spikes)
    reduced.to_parquet(f"{OUTPUT_DIR}/netrics_reduced_dataset.parquet")
    print(f"Saved reduced dataset ({len(reduced)} rows) to {OUTPUT_DIR}/netrics_reduced_dataset.parquet")

    reduced_device_ids = set(reduced['device_id'].unique())

    netrics_raw_timeseries.to_parquet(f"{OUTPUT_DIR}/netrics_timeseries_full.parquet")
    print(f"Saved full time series ({len(netrics_raw_timeseries)} rows)")

    reduced_timeseries = netrics_raw_timeseries[netrics_raw_timeseries['device_id'].isin(reduced_device_ids)]
    reduced_timeseries.to_parquet(f"{OUTPUT_DIR}/netrics_timeseries_reduced.parquet")
    print(f"Saved reduced time series ({len(reduced_timeseries)} rows, {len(reduced_device_ids)} devices)")


if __name__ == "__main__":
    run()
