"""Architecture rule: perception and planning never import the simulator (CLAUDE.md)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGES = Path(__file__).resolve().parents[1] / "src" / "terranav"


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: resolve against the file's own package
                pkg = "terranav." + path.parent.name
                names.append(pkg + ("." + node.module if node.module else ""))
            elif node.module:
                names.append(node.module)
    return names


@pytest.mark.parametrize(("package", "min_files"), [("perception", 3), ("planning", 3)])
def test_package_does_not_import_sim(package, min_files):
    files = sorted((PACKAGES / package).glob("*.py"))
    assert len(files) >= min_files  # sanity: the glob found the package's modules
    offending = [
        f"{f.name}: {m}"
        for f in files
        for m in imported_modules(f)
        if m == "terranav.sim" or m.startswith("terranav.sim.")
    ]
    assert not offending, f"{package} imports the simulator: " + ", ".join(offending)
