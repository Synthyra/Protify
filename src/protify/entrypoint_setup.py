"""Process settings Protify's entry points apply before Hugging Face or torch loads.

Hugging Face reads its environment settings when it is imported, so `configure_process` runs at
the top of `main.py`, `gui.py`, and `embedder.py`, before their other imports, rather than as a
side effect of importing the package.
"""

import os


def configure_process() -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only error/warning messages
    os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "true"
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Imported here, after the environment above, which torch reads when it loads.
    import torch
    import torch._dynamo as dynamo
    import torch._inductor.config as inductor_config

    # Use TensorFloat32 tensor cores for float32 matmul on Ampere and newer GPUs.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # The cuDNN autotuner benefits stable input sizes but adds first-use overhead.
    torch.backends.cudnn.benchmark = True
    inductor_config.max_autotune_gemm_backends = "ATEN,CUTLASS,FBGEMM"
    dynamo.config.capture_scalar_outputs = True
    dynamo.config.recompile_limit = 16
