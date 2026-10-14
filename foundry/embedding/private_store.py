"""The FastPLMs feature store, loaded under a module name of its own.

A model published as a Hugging Face artifact (`trust_remote_code=True`) embeds a FastPLMs runtime and
installs it as `fastplms`. That runtime refuses to load once a different `fastplms` is imported ("Load
the Hub artifact in a separate Python process"), and the runtimes published so far ship no `features`
package, so after one has loaded, `import fastplms.features` fails. A process that both loads such a
model and uses the store can therefore never `import fastplms.features`, in either order.

The store imports nothing outside its own package, so `load_private_feature_store` loads
`fastplms/features` from the copy on `sys.path` under the module name `PRIVATE_STORE_NAME`. The
classes are that copy's own and only the module name differs. The copy is found by a path search that
imports nothing, so an artifact runtime that already holds the name `fastplms` does not hide it, and a
project vendors `fastplms.features` the way it always did.

One process, one copy: every caller gets the same module, so a `StoredFeature` one of them builds is
the class another checks with `isinstance`. A process that also imports `fastplms.features` holds a
second copy, and the two share rows and descriptors only as plain data. Use this loader in a process
that loads Hub artifacts and nowhere else: everywhere else `import fastplms.features`, so that
FastPLMs' own embedding functions and the store agree on the classes.
"""

from __future__ import annotations

import importlib.util
import sys
import threading

from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType


PRIVATE_STORE_NAME = "foundry_private_feature_store"

_LOADING = threading.Lock()


def _store_directory() -> Path:
    """The `features` package inside the `fastplms` directory that `sys.path` reaches first."""

    found = PathFinder.find_spec("fastplms")
    roots = [] if found is None else list(found.submodule_search_locations or ())
    for root in roots:
        directory = Path(root) / "features"
        if (directory / "__init__.py").is_file():
            return directory
    raise ImportError("fastplms/features is not on sys.path; vendor fastplms.features with this project.")


def load_private_feature_store() -> ModuleType:
    """The feature store package (`FeatureStore`, `FeatureReader`, `StoredFeature`, `write_rows`, ...).

    Loaded once from the `fastplms/features` directory on `sys.path` as `PRIVATE_STORE_NAME`, with its
    submodules beneath that name, and never as `fastplms.features`. Raises `ImportError` when no such
    directory is on the path, and leaves nothing half loaded when the package fails to import.
    """

    with _LOADING:
        loaded = sys.modules.get(PRIVATE_STORE_NAME)
        if loaded is not None:
            return loaded
        directory = _store_directory()
        spec = importlib.util.spec_from_file_location(
            PRIVATE_STORE_NAME, directory / "__init__.py", submodule_search_locations=[str(directory)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load the feature store from {directory}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[PRIVATE_STORE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for name in [name for name in sys.modules if name == PRIVATE_STORE_NAME or name.startswith(f"{PRIVATE_STORE_NAME}.")]:
                del sys.modules[name]
            raise
        return module
