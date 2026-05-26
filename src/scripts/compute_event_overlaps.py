import duckdb
import pandas as pd
from tqdm import tqdm

from ..config import DB_PATH

def select_best_overlaps(df):
    df = df.sort_values(by='IoU', ascending=False)
    selected = []

    for _, row in df.iterrows():
        if not selected:
            selected.append(row)
            continue

        sel_df = pd.DataFrame(selected)
        overlaps_1 = (
            (sel_df['end_time_1'] > row['start_time_1']) &
            (sel_df['start_time_1'] < row['end_time_1'])
        ).any()

        overlaps_2 = (
            (sel_df['end_time_2'] > row['start_time_2']) &
            (sel_df['start_time_2'] < row['end_time_2'])
        ).any()

        if overlaps_1 or overlaps_2:
            continue

        selected.append(row)

    return pd.DataFrame(selected)

def run():
    db_connection = duckdb.connect(database=str(DB_PATH), read_only=False)

    device_list = db_connection.execute(
        "SELECT DISTINCT device_id FROM uchicago WHERE device_id IS NOT NULL"
    ).fetchdf()['device_id'].tolist()

    target_list = ['atlanta', 'chicago', 'denver', 'johannesburg', 'paris', 'seattle', 'stockholm', 'tunis']

    db_connection.execute("""
        CREATE OR REPLACE TABLE event_overlaps (
            device_1 TEXT,
            isp_whois_1 TEXT,
            latency_target TEXT,
            start_time_1 TIMESTAMP,
            end_time_1 TIMESTAMP,
            spike_type_1 TEXT,
            spike_duration_1 INTERVAL,
            jump_threshold_1 DOUBLE,
            amplitude_1 DOUBLE,
            device_2 TEXT,
            isp_whois_2 TEXT,
            start_time_2 TIMESTAMP,
            end_time_2 TIMESTAMP,
            spike_type_2 TEXT,
            spike_duration_2 INTERVAL,
            jump_threshold_2 DOUBLE,
            amplitude_2 DOUBLE,
            union_start TIMESTAMP,
            union_end TIMESTAMP,
            intersection_start TIMESTAMP,
            intersection_end TIMESTAMP,
            IoU DOUBLE
        )
    """)

    for target in tqdm(target_list, desc="Processing targets"):
        for device in tqdm(device_list, desc="Processing devices", leave=False):
            device_spikes = db_connection.execute(f"""
                SELECT * FROM changepoints_pelt 
                WHERE device_id = '{device}' 
                AND latency_target = '{target}' 
                AND starts IS NOT NULL AND ends IS NOT NULL
            """).fetchdf()

            if device_spikes.empty:
                continue

            db_connection.register("device_spikes", device_spikes)

            overlap_query = f"""
                SELECT
                    '{device}' AS device_1,
                    ds.isp_whois AS isp_whois_1,
                    ds.latency_target,
                    ds.starts AS start_time_1,
                    ds.ends AS end_time_1,
                    ds.spike_type AS spike_type_1,
                    ds.duration AS spike_duration_1,
                    ds.jump_threshold AS jump_threshold_1,
                    ds.amplitude AS amplitude_1,

                    other.device_id AS device_2,
                    other.isp_whois AS isp_whois_2,
                    other.starts AS start_time_2,
                    other.ends AS end_time_2,
                    other.spike_type AS spike_type_2,
                    other.duration AS spike_duration_2,
                    other.jump_threshold AS jump_threshold_2,
                    other.amplitude AS amplitude_2,
                FROM device_spikes ds
                JOIN changepoints_pelt other
                ON ds.latency_target = other.latency_target
                AND ds.starts <= other.ends
                AND ds.ends >= other.starts
                WHERE other.device_id != '{device}' 
                AND other.latency_target = '{target}'
                AND other.starts IS NOT NULL AND other.ends IS NOT NULL
            """

            overlap_data = db_connection.execute(overlap_query).fetchdf()

            if overlap_data.empty:
                continue

            # Compute IoU
            overlap_data['union_start'] = overlap_data[['start_time_1', 'start_time_2']].min(axis=1)
            overlap_data['union_end'] = overlap_data[['end_time_1', 'end_time_2']].max(axis=1)
            overlap_data['intersection_start'] = overlap_data[['start_time_1', 'start_time_2']].max(axis=1)
            overlap_data['intersection_end'] = overlap_data[['end_time_1', 'end_time_2']].min(axis=1)

            overlap_data['IoU'] = (
                (overlap_data['intersection_end'] - overlap_data['intersection_start']).dt.total_seconds()
                / (overlap_data['union_end'] - overlap_data['union_start']).dt.total_seconds()
            )
                
            overlap_data['device_pair'] = overlap_data.apply(
                lambda row: f"{row['device_1']}_{row['device_2']}" if row['device_1'] < row['device_2'] else f"{row['device_2']}_{row['device_1']}",
                axis=1
            )

            overlap_data['low_start'] = overlap_data.apply(lambda row: row['start_time_1'] if row['device_1'] < row['device_2'] else row['start_time_2'], axis=1)
            overlap_data['high_start'] = overlap_data.apply(lambda row: row['start_time_2'] if row['device_1'] < row['device_2'] else row['start_time_1'], axis=1)
            overlap_data['low_end'] = overlap_data.apply(lambda row: row['end_time_1'] if row['device_1'] < row['device_2'] else row['end_time_2'], axis=1)
            overlap_data['high_end'] = overlap_data.apply(lambda row: row['end_time_2'] if row['device_1'] < row['device_2'] else row['end_time_1'], axis=1)

            overlap_data = overlap_data.drop_duplicates(subset=['device_pair', 'latency_target', 'low_start', 'high_start', 'low_end', 'high_end'])

            # Select best overlaps
            grouped = overlap_data.groupby(['device_pair', 'latency_target'])
            selected = []

            for _, group_df in grouped:
                selected.append(select_best_overlaps(group_df))

            if not selected:
                continue
            
            filtered = pd.concat(selected, ignore_index=True)
            filtered = filtered.drop(columns=['device_pair', 'low_start', 'high_start', 'low_end', 'high_end'])

            db_connection.register("filtered_chunk", filtered)
            db_connection.execute("INSERT INTO event_overlaps SELECT * FROM filtered_chunk")

    db_connection.close()