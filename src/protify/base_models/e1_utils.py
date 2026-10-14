"""E1 batch preparation as Protify configures it.

The preparer is FastPLMs', which owns the multi-sequence encoding. Protify adds where its tokenizer comes
from, the bundled file and then any path a host registered, and where the boundary token ids live.
"""

import os
import torch
import torch.distributed as dist

from typing import Any
from tokenizers import Tokenizer

from .fastplms_loading import ensure_fastplms_submodule_on_path


ensure_fastplms_submodule_on_path()

from fastplms.models.e1.modeling_e1 import PAD_TOKEN_ID, DataPrepConfig, get_context
from fastplms.models.e1.modeling_e1 import E1BatchPreparer as FastE1BatchPreparer


__all__ = [
    "PAD_TOKEN_ID",
    "DataPrepConfig",
    "E1BatchPreparer",
    "barrier",
    "destroy_process_group",
    "get_context",
    "get_device",
    "get_local_rank",
    "get_rank",
    "get_tokenizer",
    "get_world_size",
    "is_dist_initialized",
    "register_tokenizer_path",
    "setup_dist",
]

_E1_TOKENIZER_FILENAME = "e1_tokenizer.json"
_E1_HF_REPO = "Synthyra/Profluent-E1-150M"
_E1_HF_FILENAME = "tokenizer.json"

_extra_tokenizer_paths: list[str] = []


def register_tokenizer_path(path: str) -> None:
    """Register an additional search path for the E1 tokenizer.

    Allows host projects (e.g. synth) to inject environment-specific paths
    (Modal volumes, repo data dirs) without coupling this module to them.
    """
    if path not in _extra_tokenizer_paths:
        _extra_tokenizer_paths.append(path)


def get_tokenizer() -> Tokenizer:
    candidates = [
        os.path.join(os.path.dirname(__file__), _E1_TOKENIZER_FILENAME),
        *_extra_tokenizer_paths,
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            tokenizer = Tokenizer.from_file(path)
            assert tokenizer.padding["pad_id"] == PAD_TOKEN_ID, (
                f"Padding token id must be {PAD_TOKEN_ID}, but got {tokenizer.padding['pad_id']}"
            )
            return tokenizer

    print(f"[E1] Tokenizer not found locally, downloading from HuggingFace ({_E1_HF_REPO})")
    from huggingface_hub import hf_hub_download
    fname = hf_hub_download(repo_id=_E1_HF_REPO, filename=_E1_HF_FILENAME)
    tokenizer = Tokenizer.from_file(fname)
    assert tokenizer.padding["pad_id"] == PAD_TOKEN_ID, (
        f"Padding token id must be {PAD_TOKEN_ID}, but got {tokenizer.padding['pad_id']}"
    )
    return tokenizer


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not is_dist_initialized():
        return 1
    return dist.get_world_size(group=group)


def get_rank(group: Any = None) -> int:
    if os.environ.get("RANK", -1) == -1 or not is_dist_initialized():
        return 0
    return dist.get_rank(group=group)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0)) if is_dist_initialized() else 0


def setup_dist() -> None:
    rank = int(os.environ.get("RANK", -1))
    if dist.is_available() and torch.cuda.is_available() and rank != -1:
        torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))


def destroy_process_group() -> None:
    if is_dist_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


class E1BatchPreparer(FastE1BatchPreparer):
    """FastPLMs' E1 batch preparer on Protify's tokenizer, with its boundary token ids on ``device``."""

    def __init__(
        self,
        data_prep_config: DataPrepConfig | None = None,
        tokenizer: Tokenizer | None = None,
        preserve_context_labels: bool = False,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            data_prep_config=data_prep_config,
            tokenizer=tokenizer or get_tokenizer(),
            preserve_context_labels=preserve_context_labels,
        )
        self.boundary_token_ids = self.boundary_token_ids.to(device or get_device())  # (5,) <bos>, <eos>, 1, 2, <pad>
