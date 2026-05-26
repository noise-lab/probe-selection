"""
Detect latency jumps using the Jitterbug pipeline on Netrics / FLOTO DuckDB data.

Usage:
    uv run detect-jumps --output /data/device_placement/pelt_changepoints.parquet
"""

import argparse
import os
import uuid
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ..config import DB_PATH

NETRICS_TARGETS = [
    'atlanta', 'chicago', 'denver', 'johannesburg', 'paris',
    'seattle', 'stockholm', 'tunis'
]


def _acquire_device_list(db_path: str) -> list[str]:
    import duckdb
    conn = duckdb.connect(database=db_path, read_only=True)
    rows = conn.execute(
        "SELECT DISTINCT device_id FROM google WHERE device_id IS NOT NULL"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _run_netrics_single(penalty, jump_threshold, db_path, device_id, latency_target, parquet_dir):
    import duckdb
    from ..apis.congestion_detection import CongestionDetector
    temp_path = f"{parquet_dir}/changepoints_temp_{uuid.uuid4().hex}.parquet"
    try:
        conn = duckdb.connect(database=db_path, read_only=True)
        detector = CongestionDetector(
            db_connection=conn,
            inference_method="jd",
            cdp_algorithm="pelt",
            pelt_penalty=penalty,
            latency_jump_threshold=jump_threshold,
            window_size=48 * 60 * 60,
            window_step_size=24 * 60 * 60,
        )
        try:
            cp = detector.slide_window_and_detect_congestion(device_id, latency_target)
        except ValueError:
            cp = pd.DataFrame()

        if not cp.empty:
            anomaly = cp[cp["spike_type"] != "Normal"]
            if not anomaly.empty:
                cols = detector.common_columns + detector.param_list[detector.jitterbug_analyzer.cdp_algorithm]
                anomaly[cols].to_parquet(temp_path, index=False)

        conn.close()
        return (penalty, jump_threshold, device_id, latency_target,
                len(cp), temp_path if not cp.empty else None)
    except Exception:
        return (penalty, jump_threshold, device_id, latency_target, None, None)


def main():
    parser = argparse.ArgumentParser(description="Detect latency jumps via Jitterbug.")
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"Path to netrics.db DuckDB file (default: {DB_PATH}).")
    parser.add_argument("--output", required=True,
                        help="Output parquet path for detected changepoints.")
    parser.add_argument("--pelt-penalty", type=float, default=0.001,
                        help="PELT penalty (default: 0.001).")
    parser.add_argument("--jump-threshold", type=float, default=0.5,
                        help="Latency jump threshold in ms (default: 0.5).")

    args = parser.parse_args()

    parquet_dir = str(Path(args.output).parent)
    Path(parquet_dir).mkdir(parents=True, exist_ok=True)

    device_list = _acquire_device_list(args.db)

    param_grid = [
        (args.pelt_penalty, args.jump_threshold, args.db, dev, tgt)
        for dev in device_list
        for tgt in NETRICS_TARGETS
    ]

    results = []
    for params in tqdm(param_grid, desc="netrics devices × targets"):
        results.append(_run_netrics_single(*params, parquet_dir=parquet_dir))

    temp_files = [r[-1] for r in results if r[-1] is not None]
    if temp_files:
        frames = [pd.read_parquet(f) for f in temp_files if os.path.exists(f)]
        final = pd.concat(frames, ignore_index=True)
        final.to_parquet(args.output, index=False)
        print(f"Saved {len(final):,} changepoints → {args.output}")

        import duckdb
        conn = duckdb.connect(database=args.db)
        conn.register("final_df", final)
        conn.execute("CREATE OR REPLACE TABLE changepoints_pelt AS SELECT * FROM final_df")
        conn.close()

        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
    else:
        print("No changepoints detected.")


if __name__ == "__main__":
    main()
