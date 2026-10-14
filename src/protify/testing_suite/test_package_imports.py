"""Protify imports as a package: every module loads, the package root loads nothing, and the version agrees.

Each check runs in a fresh interpreter, because this process has already imported half the package. A module
that names a vendored foundry module the projection does not carry fails here in a GitHub clone.
"""

import json
import os
import subprocess
import sys
import tomllib

import protify

from pathlib import Path

from protify.version import PROTIFY_VERSION


PACKAGE = Path(protify.__file__).resolve().parent


def _module_names() -> list[str]:
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
        if "testing_suite" in parts or (len(parts) > 1 and parts[1] == "fastplms"):
            continue
        names.append(".".join(parts).removesuffix(".__init__"))
    return names


def _run(code: str) -> dict:
    # The parent's sys.path carries the import roots that conftest and the runner set up.
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=environment, timeout=600, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_every_module_imports() -> None:
    names = _module_names()
    code = (
        "import importlib, json\n"
        f"names = {names!r}\n"
        "failed = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as error:  # noqa: BLE001 - every failure is reported, not only the first\n"
        "        failed[name] = f'{type(error).__name__}: {error}'\n"
        "print(json.dumps(failed))\n"
    )
    assert _run(code) == {}
    assert "protify.main" in names and "protify.probes.get_probe" in names


def test_importing_the_package_loads_nothing_heavy_and_leaves_sys_path_alone() -> None:
    code = (
        "import json, sys\n"
        "before = list(sys.path)\n"
        "import protify, protify.probes, protify.base_models\n"
        "heavy = sorted(name for name in ('torch', 'transformers', 'datasets', 'sklearn') if name in sys.modules)\n"
        "print(json.dumps({'heavy': heavy, 'path_changed': sys.path != before, 'version': protify.__version__}))\n"
    )
    assert _run(code) == {"heavy": [], "path_changed": False, "version": PROTIFY_VERSION}


def test_pyproject_declares_the_package_version() -> None:
    pyproject = tomllib.loads((PACKAGE.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["tool"]["poetry"]["version"] == PROTIFY_VERSION
