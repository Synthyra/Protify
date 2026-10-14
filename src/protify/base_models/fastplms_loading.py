import importlib.util
import os
import sys
import torch
import torch.nn as nn

from typing import Any


def ensure_fastplms_submodule_on_path() -> str:
    fastplms_repository = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'fastplms',
    )
    fastplms_root = os.path.join(fastplms_repository, 'src')
    fastplms_package = os.path.join(fastplms_root, 'fastplms')
    if not os.path.isfile(os.path.join(fastplms_package, '__init__.py')):
        # The research foundry keeps one FastPLMs, at models/fastplms, importable as
        # `fastplms`. A GitHub checkout keeps it as this submodule, which wins when present.
        importable = importlib.util.find_spec("fastplms")
        if importable is not None and importable.origin is not None:
            return os.path.dirname(os.path.dirname(importable.origin))
        raise ImportError(
            "FastPLMs 1.0 source package is missing. Initialize the submodule with "
            "`git submodule update --init --recursive`."
        )

    normalized_root = os.path.normcase(os.path.abspath(fastplms_root))
    sys.path[:] = [
        path
        for path in sys.path
        if os.path.normcase(os.path.abspath(path)) != normalized_root
    ]
    sys.path.insert(0, fastplms_root)

    if "fastplms" not in sys.modules:
        return fastplms_root

    fastplms_module = sys.modules["fastplms"]
    module_locations = []
    if (
        "__file__" in fastplms_module.__dict__
        and fastplms_module.__dict__["__file__"] is not None
    ):
        module_locations.append(fastplms_module.__dict__["__file__"])
    if "__path__" in fastplms_module.__dict__:
        module_locations.extend(list(fastplms_module.__dict__["__path__"]))

    fastplms_root_abs = os.path.normcase(os.path.abspath(fastplms_package))
    loaded_from_submodule = any(
        os.path.normcase(os.path.abspath(str(location))).startswith(fastplms_root_abs)
        for location in module_locations
    )
    if loaded_from_submodule:
        return fastplms_root

    for module_name in list(sys.modules):
        if module_name == "fastplms" or module_name.startswith("fastplms."):
            del sys.modules[module_name]
    return fastplms_root


# SDPA is the one backend every family reaching this loader advertises, and FastPLMs
# documents it as ESMC's highest-fidelity path. Flex Attention is documented as
# numerically divergent on ESMC and pays a compilation cost on every new sequence
# shape, which variable-length protein batches hit constantly. The two FlashAttention
# backends are BF16-and-CUDA only and are recorded as unavailable on the current
# kernel lock for ESMC. Callers that want to measure another backend pass it
# explicitly; FastPLMs raises rather than substituting one it cannot honor.
DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"


def load_fastplms_model(
    model_class: type[Any],
    model_path: str,
    *,
    dtype: torch.dtype | None = None,
    attn_implementation: str = DEFAULT_ATTENTION_IMPLEMENTATION,
    **kwargs: Any,
) -> nn.Module:
    """Load a FastPLMs 1.0 model through the Transformers attention contract."""
    if dtype not in (None, torch.float32, torch.bfloat16):
        raise ValueError(
            "FastPLMs 1.0 supports model_dtype values float32 and bfloat16; "
            f"received {dtype}."
        )

    load_kwargs = dict(kwargs)
    if dtype is not None:
        module_name = model_class.__module__
        fp32_parameter_families = (
            "fastplms.models.esm2.",
            "fastplms.models.dplm.",
            "fastplms.models.dplm2.",
        )
        load_kwargs["dtype"] = (
            torch.float32
            if dtype is torch.bfloat16
            and module_name.startswith(fp32_parameter_families)
            else dtype
        )
    load_kwargs["attn_implementation"] = attn_implementation
    return model_class.from_pretrained(model_path, **load_kwargs)
