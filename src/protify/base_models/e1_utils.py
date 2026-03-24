import os
import torch
import torch.distributed as dist
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


PAD_TOKEN_ID = 0

_E1_TOKENIZER_FILENAME = "e1_tokenizer.json"
_E1_HF_REPO = "Synthyra/Profluent-E1-150M"
_E1_HF_FILENAME = "tokenizer.json"

_extra_tokenizer_paths: List[str] = []


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


@dataclass
class DataPrepConfig:
    max_num_sequences: int = 512
    max_num_positions_within_seq: int = 8192
    remove_X_tokens: bool = False


def get_context(sequence: str) -> Optional[str]:
    if "," in sequence:
        return sequence.rsplit(",", 1)[0]
    return None


class E1BatchPreparer:
    def __init__(
        self,
        data_prep_config: Optional[DataPrepConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        preserve_context_labels: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.tokenizer = tokenizer or get_tokenizer()
        self.data_prep_config = data_prep_config or DataPrepConfig()
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.preserve_context_labels = preserve_context_labels
        self.boundary_token_ids = torch.tensor(
            [self.tokenizer.token_to_id(token) for token in ["<bos>", "<eos>", "1", "2", "<pad>"]], 
            device=(device or get_device())
        ).long()
        self.mask_token = "?"  # nosec
        self.mask_token_id = self.tokenizer.token_to_id(self.mask_token)
        self.X_token_id = self.tokenizer.token_to_id("X")
        self.vocab = self.tokenizer.get_vocab()

    def get_batch_kwargs(  # type: ignore[override]
        self, sequences: List[str], device: torch.device = torch.device("cpu"), non_blocking: bool = False
    ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        sequence_encodings = [self.prepare_multiseq(sequence) for sequence in sequences]
        return self.pad_encodings(sequence_encodings, device, non_blocking)

    def pad_encodings(
        self,
        sequence_encodings: List[Dict[str, torch.Tensor]],
        device: torch.device = torch.device("cpu"),
        non_blocking: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[str], List[int]]]:
        non_blocking = non_blocking and device.type == "cuda"
        padded_encodings = {}
        # Note: We use -1 as the padding value for sequence and position ids because the 0 value
        # is a valid value for sequence and position ids. -1 is then used to distinguish valid
        # tokens from padding tokens, for example, when doing padding/unpadding for flash attention.
        for key, padding_value in {
            "input_ids": self.pad_token_id,
            "sequence_ids": -1,
            "within_seq_position_ids": -1,
            "global_position_ids": -1,
            "labels": self.pad_token_id,
        }.items():
            padded_encodings[key] = pad_sequence(
                [enc[key] for enc in sequence_encodings], batch_first=True, padding_value=padding_value
            ).to(device=device, dtype=torch.long, non_blocking=non_blocking)

        padded_encodings["context"] = [enc["context"] for enc in sequence_encodings]
        padded_encodings["context_len"] = [enc["context_len"] for enc in sequence_encodings]

        return padded_encodings

    def prepare_multiseq(self, sequence: str) -> Dict[str, Union[torch.Tensor, str, int]]:
        single_sequences = sequence.split(",")
        if len(single_sequences) > self.data_prep_config.max_num_sequences:
            raise ValueError(
                f"Number of sequences {len(single_sequences)} exceeds max number of sequences {self.data_prep_config.max_num_sequences}"
                " in the provided multi-sequence instance. Please remove some homologous sequences before trying again."
            )

        single_sequence_encodings = [self.prepare_singleseq(sequence) for sequence in single_sequences]

        num_tokens = [len(x["input_ids"]) for x in single_sequence_encodings]
        input_ids = torch.cat([x["input_ids"] for x in single_sequence_encodings])
        labels = torch.cat([x["labels"] for x in single_sequence_encodings])

        within_seq_position_ids = torch.cat([encoding["position_ids"] for encoding in single_sequence_encodings])
        global_position_ids, ctx_len = [], 0
        for encoding in single_sequence_encodings:
            global_position_ids.append(encoding["position_ids"] + ctx_len)
            ctx_len = max(ctx_len, encoding["position_ids"].max().item() + ctx_len + 1)
        global_position_ids = torch.cat(global_position_ids)

        sequence_ids = torch.repeat_interleave(torch.tensor(num_tokens))

        # Get multi-seq context & mask out all but last sequence in multi-seq instance if desired
        context_len = sum(num_tokens[:-1])
        context = self.tokenizer.decode(input_ids[:context_len].tolist(), skip_special_tokens=False)
        if not self.preserve_context_labels:
            labels[:context_len] = self.pad_token_id

        assert (
            input_ids.shape
            == sequence_ids.shape
            == within_seq_position_ids.shape
            == global_position_ids.shape
            == labels.shape
        ), "Input ids, sequence ids, within seq position ids, global position ids, and labels must have the same shape"

        assert input_ids.shape[0] >= context_len, "Input ids must have at least as many tokens as the context length"

        return {
            "input_ids": input_ids,
            "sequence_ids": sequence_ids,
            "within_seq_position_ids": within_seq_position_ids,
            "global_position_ids": global_position_ids,
            "labels": labels,
            "context": context,
            "context_len": context_len,
        }

    def prepare_singleseq(self, sequence: str) -> Dict[str, torch.Tensor]:
        if not self.validate_sequence(sequence):
            raise ValueError(f"Invalid sequence: {sequence}; Input sequence should contain [A-Z] or ? characters only")

        if len(sequence) > self.data_prep_config.max_num_positions_within_seq:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds max length {self.data_prep_config.max_num_positions_within_seq}"
            )

        # Can also use `tokens = torch.tensor(self.tokenizer.encode(f"<bos>1{sequence}2<eos>").ids)`
        # but following is faster since our vocabulary is simple.
        tokens = torch.tensor([self.vocab[token] for token in ["<bos>", "1", *sequence, "2", "<eos>"]])
        position_ids = torch.arange(len(tokens))

        if self.data_prep_config.remove_X_tokens:
            X_positions = torch.where(tokens != self.X_token_id)[0]
            tokens = tokens[X_positions]
            position_ids = position_ids[X_positions]

        return {"input_ids": tokens, "labels": tokens, "position_ids": position_ids}

    def get_boundary_token_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return torch.isin(tokens, self.boundary_token_ids.to(tokens.device))

    def get_mask_positions_mask(self, tokens: torch.Tensor) -> torch.BoolTensor:
        return tokens == self.mask_token_id

    def validate_sequence(self, sequence: str) -> bool:
        assert isinstance(sequence, str), "Sequence must be a string"
        sequence = sequence.replace(self.mask_token, "")
        return sequence.isalpha() and sequence.isupper()
