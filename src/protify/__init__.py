"""Protify: train and benchmark probes on protein and chemical language model embeddings.

Importing the package loads nothing heavy and changes no import path; each name below loads its
module on first use, so `import protify.feature_cache` costs only that module and its imports.
"""

from importlib import import_module
from typing import Any

from .version import PROTIFY_VERSION


__version__ = PROTIFY_VERSION

# Each public name, by the module that defines it.
_EXPORTS = {
    "Embedder": ".embedder",
    "EmbeddingArguments": ".embedder",
    "Pooler": ".pooler",
    "DataArguments": ".data.data_mixin",
    "DataMixin": ".data.data_mixin",
    "torch_load": ".legacy_cache",
    "print_message": ".console_output",
    "set_global_seed": ".seed_utils",
    "get_global_seed": ".seed_utils",
    "seed_worker": ".seed_utils",
    "dataloader_generator": ".seed_utils",
    "set_determinism": ".seed_utils",
}
_SUBPACKAGES = ("probes", "data", "base_models")

__all__ = ["__version__", *_SUBPACKAGES, *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES:
        return import_module(f".{name}", __name__)
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
