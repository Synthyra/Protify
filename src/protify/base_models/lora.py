import torch.nn as nn

from peft import LoraConfig, LoraModel


# these modules handle ESM++ and ESM2 attention types, as well as any additional transformer blocks from Syndev
LORA_TARGET_MODULES = ["layernorm_qkv.1", "out_proj", "query", "key", "value", "dense"]

# Llama-style projection names, which is what CARBON's backbone and head expose. PEFT
# matches target names by suffix, so adding these unconditionally would also attach
# adapters inside any other Llama-derived checkpoint loaded through --model_types custom,
# silently changing that model's trainable parameters. Opt in by model instead.
LLAMA_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# LoRA freezes the backbone, so the task head has to be unfrozen by name. 'classifier' is
# the ESM and BERT head; 'score' is the Llama sequence-classification head.
TRAINABLE_HEAD_NAME_FRAGMENTS = ("classifier", "score")


def wrap_lora(
    module: nn.Module,
    r: int,
    lora_alpha: float,
    lora_dropout: float,
    model_name: str = "",
) -> nn.Module:
    target_modules = list(LORA_TARGET_MODULES)
    if 'carbon' in model_name.lower():
        target_modules += LLAMA_LORA_TARGET_MODULES

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    module = LoraModel(module, lora_config, 'default')
    for name, param in module.named_parameters():
        if any(fragment in name.lower() for fragment in TRAINABLE_HEAD_NAME_FRAGMENTS):
            param.requires_grad = True
    return module
