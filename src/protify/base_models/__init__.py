"""The base-model registry and factories.

Each name loads its module on first use, so importing one model wrapper or the registry does not
load every wrapper's dependencies.
"""

from importlib import import_module
from typing import Any


# Each public name, by the module that defines it.
_EXPORTS = {
    "BaseModelArguments": ".get_base_models",
    "experimental_models": ".get_base_models",
    "get_base_model": ".get_base_models",
    "get_base_model_for_training": ".get_base_models",
    "get_tokenizer": ".get_base_models",
    "model_descriptions": "..resource_info",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
