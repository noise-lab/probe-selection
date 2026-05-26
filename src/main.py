from .scripts import (
    detect_jumps,
    compute_event_overlaps,
    run_sampling_algo,
)

if __name__ == '__main__':
    # Step 1: Detect latency jumps (run via CLI: uv run detect-jumps ...)
    # detect_jumps.run()

    # Step 2: Detect overlaps between devices
    # compute_event_overlaps.run()

    # Step 3: Run sampling algorithms to select probes
    # run_sampling_algo.run()
