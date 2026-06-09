import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, LoraModel


def ensure_fastplms_submodule_on_path() -> str:
    fastplms_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'fastplms',
    )
    if fastplms_root in sys.path:
        sys.path.remove(fastplms_root)
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

    fastplms_root_abs = os.path.abspath(os.path.join(fastplms_root, 'fastplms'))
    loaded_from_submodule = any(
        os.path.abspath(str(location)).startswith(fastplms_root_abs)
        for location in module_locations
    )
    if loaded_from_submodule:
        return fastplms_root

    for module_name in list(sys.modules):
        if module_name == "fastplms" or module_name.startswith("fastplms."):
            del sys.modules[module_name]
    return fastplms_root


def select_hidden_state(
    last_hidden_state: torch.Tensor,
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    hidden_state_index: int,
) -> torch.Tensor:
    assert isinstance(hidden_state_index, int), "hidden_state_index must be an integer."
    if hidden_state_index == -1:
        return last_hidden_state
    assert hidden_states is not None, "hidden_state_index selection requires output_hidden_states=True."
    assert len(hidden_states) > 0, "Model returned no hidden states."
    return hidden_states[hidden_state_index]


def wrap_lora(module: nn.Module, r: int, lora_alpha: float, lora_dropout: float) -> nn.Module:
    # these modules handle ESM++ and ESM2 attention types, as well as any additional transformer blocks from Syndev
    target_modules=["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    module = LoraModel(module, lora_config, 'default')
    for name, param in module.named_parameters():
        if 'classifier' in name.lower():
            param.requires_grad = True
    return module
