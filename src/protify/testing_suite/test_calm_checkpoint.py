"""`multimolecule/calm`, at the revision Protify pins, loads whole and gives its published outputs.

The model card at that revision publishes, for 13 coding sequences with one codon masked, the five
codons the checkpoint's masked-LM head ranks highest there and their probabilities, rounded to six
decimals. Protify loads the backbone alone, so these tests apply the head the checkpoint holds the
way multimolecule's `MaskedLMHead` does: a dense layer, GELU, and layer norm, then the codon
embeddings as the decoder, plus its bias.

Two faults each changed every published example, moving a probability by up to 0.48: the RNA
tokenizer read each T as U, so each codon holding a T became `<unk>`, and the rotary embedding
turned channel halves instead of adjacent pairs. With both fixed, the largest difference from a
published probability is 4.9e-7, inside the card's rounding of 5e-7; the bound, 1e-6, leaves twice
that.

These tests read only the Hugging Face cache, refuse every Hub request, and fail on what is not
cached, because the `checkpoint` marker runs them only when they are asked for. Allocations made while the backbone loads hold NaN, so a buffer the load leaves unfilled
fails deterministically.
"""

import huggingface_hub.constants
import pytest
import torch
import yaml

from collections.abc import Iterator
from pathlib import Path
from huggingface_hub import try_to_load_from_cache
from safetensors import safe_open
from torch import Tensor
from torch.nn import functional as F

from foundry.checkpoint_loading import assert_unsaved_buffers_match, uninitialized_memory_as_nan
from protify.base_models.calm import DEFAULT_REVISIONS, CaLmModel, build_calm_model, presets
from protify.base_models.calm_utils import DnaTokenizer


pytestmark = pytest.mark.checkpoint

PRESET = "CaLM"
REPOSITORY = presets[PRESET]
REVISION = DEFAULT_REVISIONS[PRESET]
PUBLISHED_EXAMPLES = 13
TOP = 5  # codons the card publishes per example
PROBABILITY_TOLERANCE = 1e-6  # see the module docstring
HEAD_TENSORS = (
    "lm_head.transform.dense.weight",
    "lm_head.transform.dense.bias",
    "lm_head.transform.layer_norm.weight",
    "lm_head.transform.layer_norm.bias",
    "lm_head.bias",
)


def _cached(filename: str) -> Path:
    """The local path of a cached file of the pinned revision; a file that is not cached fails the test."""
    path = try_to_load_from_cache(REPOSITORY, filename, revision=REVISION)
    assert isinstance(path, str), (
        f"{REPOSITORY} {filename} at {REVISION} is not in the local Hugging Face cache; "
        f"fetch it with `hf download {REPOSITORY} --revision {REVISION}` before running checkpoint tests"
    )
    return Path(path)


@pytest.fixture(scope="module", autouse=True)
def hub_offline() -> Iterator[None]:
    """Refuse every Hugging Face Hub request, so nothing missing from the cache downloads."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
        yield


@pytest.fixture(scope="module")
def checkpoint_file() -> Path:
    for filename in ("config.json", "vocab.txt", "tokenizer_config.json", "README.md"):
        _cached(filename)
    return _cached("model.safetensors")


@pytest.fixture(scope="module")
def calm(checkpoint_file: Path) -> tuple[CaLmModel, DnaTokenizer]:
    """The backbone and codon tokenizer, loaded the way Protify's CaLM preset loads them."""
    with uninitialized_memory_as_nan():
        model, tokenizer = build_calm_model(PRESET)
    return model.calm, tokenizer.tokenizer


@pytest.fixture(scope="module")
def masked_lm_head(checkpoint_file: Path) -> dict[str, Tensor]:
    with safe_open(str(checkpoint_file), framework="pt") as checkpoint:
        return {name: checkpoint.get_tensor(name) for name in HEAD_TENSORS}  # head tensors, such as lm_head.bias (v,)


@pytest.fixture(scope="module")
def published_examples(checkpoint_file: Path) -> list[dict]:
    """The card's fill-mask examples: a masked coding sequence, then five codons and probabilities."""
    card = _cached("README.md").read_text(encoding="utf-8")
    examples = yaml.safe_load(card.split("---\n", 2)[1])["widget"]
    assert len(examples) == PUBLISHED_EXAMPLES
    return examples


def _codon_logits(backbone: CaLmModel, head: dict[str, Tensor], input_ids: Tensor, attention_mask: Tensor) -> Tensor:
    """Logits over the codon vocabulary from the checkpoint's masked-LM head, as multimolecule applies it."""
    # input_ids, attention_mask: (1, l)
    # head: (v,) or (d, d) or (d,), by tensor
    hidden_states = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (1, l, d)
    hidden_states = F.linear(
        hidden_states, head["lm_head.transform.dense.weight"], head["lm_head.transform.dense.bias"]
    )  # (1, l, d)
    hidden_states = F.gelu(hidden_states)  # (1, l, d)
    hidden_states = F.layer_norm(
        hidden_states,
        hidden_states.shape[-1:],
        head["lm_head.transform.layer_norm.weight"],
        head["lm_head.transform.layer_norm.bias"],
        eps=backbone.config.layer_norm_eps,
    )  # (1, l, d)
    # The decoder is tied to the codon embeddings, so the checkpoint holds only its bias.
    return F.linear(hidden_states, backbone.embeddings.word_embeddings.weight, head["lm_head.bias"])  # (1, l, v)


def test_every_backbone_tensor_loads_unchanged(calm, checkpoint_file):
    backbone, _ = calm
    loaded = backbone.state_dict()
    with safe_open(str(checkpoint_file), framework="pt") as checkpoint:
        saved = {
            key.removeprefix("model."): checkpoint.get_tensor(key)
            for key in checkpoint.keys()  # noqa: dict-idiom  a safetensors file handle has no __iter__
            if not key.startswith("lm_head.")
        }

    assert set(saved) == {name for name, _ in backbone.named_parameters()}
    for name, tensor in saved.items():
        torch.testing.assert_close(loaded[name], tensor, rtol=0, atol=0, msg=name)


def test_buffers_the_checkpoint_lacks_equal_a_fresh_backbone(calm, checkpoint_file):
    backbone, _ = calm
    with safe_open(str(checkpoint_file), framework="pt") as checkpoint:
        checkpoint_keys = list(checkpoint.keys())

    unsaved = assert_unsaved_buffers_match(
        backbone, checkpoint_keys, lambda: CaLmModel(backbone.config, add_pooling_layer=False)
    )

    assert unsaved > 0, "the rotary frequencies and position ids are the buffers under test"


@pytest.mark.parametrize("index", range(PUBLISHED_EXAMPLES))
def test_the_masked_codon_predictions_match_the_published_ones(calm, masked_lm_head, published_examples, index):
    backbone, tokenizer = calm
    example = published_examples[index]
    encoded = tokenizer(example["text"], return_tensors="pt")
    input_ids = encoded["input_ids"]  # (1, l)
    masked = (input_ids[0] == tokenizer.mask_token_id).nonzero().flatten()  # (1,)
    published_codons = [entry["label"] for entry in example["output"]]
    published_probabilities = torch.tensor([entry["score"] for entry in example["output"]])  # (5,)

    with torch.inference_mode():
        logits = _codon_logits(backbone, masked_lm_head, input_ids, encoded["attention_mask"])  # (1, l, v)
    probabilities = logits[0, masked[0]].softmax(dim=-1)  # (v,)
    top_probabilities, top_ids = probabilities.topk(TOP)  # (5,); (5,)

    assert not (input_ids == tokenizer.unk_token_id).any(), "a codon the vocabulary lacks"
    assert masked.numel() == 1
    assert tokenizer.convert_ids_to_tokens(top_ids.tolist()) == published_codons
    torch.testing.assert_close(top_probabilities, published_probabilities, rtol=0, atol=PROBABILITY_TOLERANCE)
