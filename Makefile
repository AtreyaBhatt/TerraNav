.PHONY: setup test test-all lint format-check eval smoke-pv

setup:
	uv sync

test:
	uv run pytest -m "not slow"

test-all:
	uv run pytest

# Evaluations, strictly one after another (each uses all cores). Refuses a dirty tree.
eval:
	uv run python scripts/eval_map.py
	uv run python scripts/eval_map.py --latency
	uv run python scripts/eval_trav.py
	uv run python scripts/eval_trav.py --latency
	uv run python scripts/eval_plan.py
	uv run python scripts/eval_plan.py --latency

lint:
	uv run ruff check src/ tests/ scripts/

format-check:
	uv run ruff format --check src/ tests/ scripts/

smoke-pv:
	uv run python -c "import pyvista as pv; pv.OFF_SCREEN = True; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('out/sphere_check.png'); pl.close(); print('Saved out/sphere_check.png')"

smoke-pv-xvfb:
	xvfb-run -a uv run python -c "import pyvista as pv; pv.OFF_SCREEN = True; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('out/sphere_check.png'); pl.close(); print('Saved out/sphere_check.png')"
