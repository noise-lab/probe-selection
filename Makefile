# OUTPUT_DIR ?= /data/device_placement
# CHANGEPOINTS ?= $(OUTPUT_DIR)/pelt_changepoints.parquet

OUTPUT_DIR ?= /Users/taveesh/Documents/Repositories/probe-selection/data
CHANGEPOINTS ?= $(OUTPUT_DIR)/pelt_changepoints.parquet

.PHONY: install lint detect-jumps overlaps select-probes pipeline

install:
	uv sync --dev

lint:
	uv run ruff check src/

detect-jumps:
	uv run detect-jumps --output "$(CHANGEPOINTS)"

overlaps:
	uv run python -c "from src.scripts.compute_event_overlaps import run; run()"

select-probes:
	OUTPUT_DIR="$(OUTPUT_DIR)" uv run python -c "from src.scripts.run_sampling_algo import run; run()"

pipeline: detect-jumps overlaps select-probes
