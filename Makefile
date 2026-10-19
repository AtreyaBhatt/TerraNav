.PHONY: setup test test-all lint format-check eval eval-scan benchmark latency smoke-pv

setup:
	uv sync

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

# Table evaluations, strictly one after another (each uses all cores). Refuse a dirty tree.
eval:
	uv run python scripts/eval_map.py
	uv run python scripts/eval_trav.py
	uv run python scripts/eval_plan.py

# Scan-strategy comparison (slow; only when the survey strategy changes). Refuses a dirty tree.
eval-scan:
	uv run python scripts/eval_scan.py

# Pre-registered held-out benchmark (docs/benchmark_protocol.md). Refuses a dirty tree; writes
# results/<git>_<config-hash>/.
benchmark:
	uv run python scripts/benchmark.py

# Serial latency measurements. Run only when perception or planning code changed.
latency:
	uv run python scripts/eval_map.py --latency
	uv run python scripts/eval_trav.py --latency
	uv run python scripts/eval_plan.py --latency

lint:
	uv run ruff check src/ tests/ scripts/

format-check:
	uv run ruff format --check src/ tests/ scripts/

smoke-pv:
	uv run python -c "import pyvista as pv; pv.OFF_SCREEN = True; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('out/sphere_check.png'); pl.close(); print('Saved out/sphere_check.png')"

smoke-pv-xvfb:
	xvfb-run -a uv run python -c "import pyvista as pv; pv.OFF_SCREEN = True; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('out/sphere_check.png'); pl.close(); print('Saved out/sphere_check.png')"
