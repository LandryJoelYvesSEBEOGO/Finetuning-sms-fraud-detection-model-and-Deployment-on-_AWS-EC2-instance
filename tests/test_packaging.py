"""Vercel may read dependencies from pyproject.toml or requirements.txt: keep them identical."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _requirements() -> set[str]:
    lines = (ROOT / "requirements.txt").read_text().splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_runtime_dependencies_are_declared_identically() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert set(pyproject["project"]["dependencies"]) == _requirements()


def test_vercel_entrypoint_points_to_the_app() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    module, _, attribute = pyproject["tool"]["vercel"]["entrypoint"].partition(":")
    assert (ROOT / (module.replace(".", "/") + ".py")).is_file()
    assert attribute == "app"
