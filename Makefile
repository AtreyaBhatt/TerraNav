.PHONY: setup test lint format-check smoke-pv

setup:
	uv sync

test:
	uv run pytest -v

lint:
	uv run ruff check src/ tests/ scripts/

format-check:
	uv run ruff format --check src/ tests/ scripts/

smoke-pv:
	uv run python -c "import pyvista as pv; pv.OFF_SCREEN = True; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('out/sphere_check.png'); pl.close(); print('Saved out/sphere_check.png')"

smoke-pv-xvfb:
	xvfb-run -a uv run python -c "import pyvista as pv; pv.OFF_SCREEN = True; pl = pv.Plotter(off_screen=True); pl.add_mesh(pv.Sphere()); pl.screenshot('out/sphere_check.png'); pl.close(); print('Saved out/sphere_check.png')"
