"""Architecture rule: perception never imports the simulator (sensor boundary, CLAUDE.md)."""

from __future__ import annotations

import ast
from pathlib import Path

PERCEPTION = Path(__file__).resolve().parents[1] / "src" / "terranav" / "perception"


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against terranav.perception
                names.append("terranav.perception" + ("." + node.module if node.module else ""))
            elif node.module:
                names.append(node.module)
    return names


def test_perception_does_not_import_sim():
    files = sorted(PERCEPTION.glob("*.py"))
    assert len(files) >= 3  # backproject, elevation_map, hole_fill (sanity: glob works)
    offending = [
        f"{f.name}: {m}"
        for f in files
        for m in imported_modules(f)
        if m == "terranav.sim" or m.startswith("terranav.sim.")
    ]
    assert not offending, "perception imports the simulator: " + ", ".join(offending)
