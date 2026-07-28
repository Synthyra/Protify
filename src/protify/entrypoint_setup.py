import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only error/warning messages
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "true"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import TensorFlow under a targeted deprecation-warning filter.
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=(
            "The name tf.losses.sparse_softmax_cross_entropy is deprecated. "
            "Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead."
        ),
        category=FutureWarning,
        module=".*tf_keras\\.src\\.losses.*",
    )
    try:
        import tensorflow as tf
    except ImportError:
        pass

import torch
import torch._inductor.config as inductor_config
import torch._dynamo as dynamo

# Use TensorFloat32 tensor cores for float32 matmul on Ampere and newer GPUs.
torch.set_float32_matmul_precision("high")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# The cuDNN autotuner benefits stable input sizes but adds first-use overhead.
torch.backends.cudnn.benchmark = True
inductor_config.max_autotune_gemm_backends = "ATEN,CUTLASS,FBGEMM"

dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.recompile_limit = 16

try:
    import wandb

    os.environ["WANDB_AVAILABLE"] = "true"
except ImportError:
    os.environ["WANDB_AVAILABLE"] = "false"
