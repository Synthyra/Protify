"""A tiny canonical embedding store for CPU tests: a random ESM++ with a random SAE, every stream l + 2 rows.

The embedding engine writes real stores (`foundry.embedding.job`). This builds one from a tiny random model
with the engine's own taps, contract and manifest, so a reader, a probe and a research run can be tested
without a checkpoint or a GPU. It follows the engine's own test fixture (`tests/tier1_unit/test_embedding_canonical.py`).

Symbols: `d` hidden width, `c` SAE codebook, `k` codes kept per token, `l` residues of a protein.
"""

import torch

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastplms.embeddings import embed_into_features
from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel
from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import ESMplusplusSAELayer, ESMplusplusSAEParams
from foundry.embedding import FeatureExtractionContract, canonical_base, sequence_inventory
from foundry.embedding.canonical import canonical_taps, stream_manifest, write_manifest


SOURCE = {"repository": "fixture/backbone", "revision": "1" * 40}
HIDDEN_SIZE = 32  # d
N_LAYERS = 24  # hidden states 0..24, so the depth layer 23 of ESMC-300 exists
SAE_LAYER = 23
CODEBOOK = 8192  # c, the width of the toy suite's SAE branch spec
K = 64  # k
MAX_RESIDUES = 62  # the crop; the toy suite's max_length of 64 counts CLS and EOS


def _tiny_model(seed: int) -> ESMplusplusModel:
    torch.manual_seed(seed)
    model = ESMplusplusModel(ESMplusplusConfig(
        hidden_size=HIDDEN_SIZE, num_attention_heads=4, num_hidden_layers=N_LAYERS, attn_backend="sdpa",
    )).eval().requires_grad_(False)
    model.config._name_or_path, model.config._commit_hash = SOURCE["repository"], SOURCE["revision"]
    return model


def _tiny_sae() -> ESMplusplusSAELayer:
    generator = torch.Generator().manual_seed(1)
    layer = ESMplusplusSAELayer(ESMplusplusSAEParams(d_model=HIDDEN_SIZE, codebook_dim=CODEBOOK, k=K, layer=SAE_LAYER))
    torch.nn.init.normal_(layer.W_enc, generator=generator)
    torch.nn.init.normal_(layer.W_dec, generator=generator)
    torch.nn.init.normal_(layer.b_dec, std=0.1, generator=generator)
    return layer.eval().requires_grad_(False)


def _structural_weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2)
    return {
        "layer_logits": torch.randn(N_LAYERS + 1, generator=generator),  # (n,)
        "projection.0.weight": torch.randn(HIDDEN_SIZE, generator=generator),  # (d,)
        "projection.0.bias": torch.randn(HIDDEN_SIZE, generator=generator),  # (d,)
        "projection.1.weight": torch.randn(256, HIDDEN_SIZE, generator=generator),  # (256, d)
    }


def write_canonical_store(root: Path, sequences: Sequence[str], *, base_id: str = "esmc_small", seed: int = 5) -> Path:
    """Embed `sequences` into a canonical store of the base `base_id` under `root`, which is returned.

    A trained base (`esmc_small`) gets every stream; the random-init control (`esmc_small_random`) gets its three.
    The width, SAE layer and crop are the tiny model's, and the manifest records them as the engine would.
    """
    base = replace(canonical_base(base_id), hidden_width=HIDDEN_SIZE, sae_layer=SAE_LAYER)
    model = _tiny_model(seed)
    sae, weights = (_tiny_sae(), _structural_weights()) if base.has_sae else (None, None)
    taps = canonical_taps(base, sae=sae, weights=weights)
    names = [tap.name for tap in taps]
    options: dict[str, Any] = dict(
        model_source=SOURCE, tokenizer_source=SOURCE, engine_sha256="2" * 64, reducer_sha256={name: "3" * 64 for name in names},
        normalization_version="protein_ascii_v1", max_length=MAX_RESIDUES, truncate=True, batch_size=8, batch_window_size=32,
        max_tokens_per_batch=512, dtype=None, keep_special_tokens=True, model_initialization=base.model_initialization,
    )
    if base.has_sae:
        options.update(
            sae_sources={"sae_codes": SOURCE, "sae_max": SOURCE}, sae_layers={"sae_codes": sae, "sae_max": sae},
            structural_sources={"structural": {"repository": "Synthyra/fixture", "revision": "4" * 40}},
            structural_weights={"structural": weights},
        )
    contract = FeatureExtractionContract(model, taps, sequence_inventory(list(sequences)), **options)
    embed_into_features(
        model, list(sequences), root, contract.stored_features, taps=taps, contract=contract, keep_special_tokens=True,
        **contract.embedding_options,
    )
    write_manifest(root, stream_manifest(base, contract))
    return root
