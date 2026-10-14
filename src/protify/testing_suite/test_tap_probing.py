"""Tap probing on a tiny random ESMC: every hooked state against a manual forward, the stores, the probe bank's
gradients and optimizer, the scores, and both trainers end to end on small synthetic tasks."""

import json
import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from dataclasses import replace
from pathlib import Path
from safetensors.torch import load_file

try:
    from src.protify.base_models.esmc import get_esmc_tokenizer
    from src.protify.interpretability.capture import capture_states, model_dims
    from src.protify.interpretability import figures
    from src.protify.interpretability.cli import STORE_VARIANTS, main, store_path, train_tasks, write_stores
    from src.protify.interpretability.concepts import ConceptData, ConceptSplit, ConceptTask, load_concept, prot_param_values, scramble, scramble_seed, shuffled_labels
    from src.protify.interpretability.interventions import ACTIVE, INACTIVE, UNKNOWN, PhosphoProtein, fit_logistic, paired_tests, phosphoproteins, pooled_means, run_phospho_interventions, scanning_mutants
    from src.protify.interpretability.latent import SCRAMBLED, label_sets, principal_components, project_layers
    from src.protify.interpretability.pooled_store import PooledTapStore, embed_pooled, masked_mean_var
    from src.protify.interpretability.probe_bank import ProbeBank, ProbeSettings
    from src.protify.interpretability.protein_probing import head_view, score_bank, validation_loss
    from src.protify.interpretability.residue_probing import untrained_copy
    from src.protify.interpretability.scores import ResamplePlan, ScoreAccumulator, binary_mcc, multiclass_mcc
    from src.protify.interpretability.selection import choose_setting, rank_settings, read_sweeps
    from src.protify.interpretability.sites import ALL_SITES, ModelDims, TapSpec, parse_sites, site_layouts
    from src.protify.interpretability.token_batches import SPECIAL_TOKENS, token_batches
    from src.protify.interpretability.token_store import TokenTapStore, embed_tokens
    from src.protify.interpretability.training import CONTROLS, PLM_STATES, PROBE_STATES, TrainingSettings, make_bank
except ImportError:
    from protify.base_models.esmc import get_esmc_tokenizer
    from protify.interpretability.capture import capture_states, model_dims
    from protify.interpretability import figures
    from protify.interpretability.cli import STORE_VARIANTS, main, store_path, train_tasks, write_stores
    from protify.interpretability.concepts import ConceptData, ConceptSplit, ConceptTask, load_concept, prot_param_values, scramble, scramble_seed, shuffled_labels
    from protify.interpretability.interventions import ACTIVE, INACTIVE, UNKNOWN, PhosphoProtein, fit_logistic, paired_tests, phosphoproteins, pooled_means, run_phospho_interventions, scanning_mutants
    from protify.interpretability.latent import SCRAMBLED, label_sets, principal_components, project_layers
    from protify.interpretability.pooled_store import PooledTapStore, embed_pooled, masked_mean_var
    from protify.interpretability.probe_bank import ProbeBank, ProbeSettings
    from protify.interpretability.protein_probing import head_view, score_bank, validation_loss
    from protify.interpretability.residue_probing import untrained_copy
    from protify.interpretability.scores import ResamplePlan, ScoreAccumulator, binary_mcc, multiclass_mcc
    from protify.interpretability.selection import choose_setting, rank_settings, read_sweeps
    from protify.interpretability.sites import ALL_SITES, ModelDims, TapSpec, parse_sites, site_layouts
    from protify.interpretability.token_batches import SPECIAL_TOKENS, token_batches
    from protify.interpretability.token_store import TokenTapStore, embed_tokens
    from protify.interpretability.training import CONTROLS, PLM_STATES, PROBE_STATES, TrainingSettings, make_bank

from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny_model() -> ESMplusplusModel:
    torch.manual_seed(0)
    config = ESMplusplusConfig(hidden_size=64, num_attention_heads=4, num_hidden_layers=3, attn_backend="sdpa")
    return ESMplusplusModel(config).eval()


@pytest.fixture(scope="module")
def tokenizer() -> object:
    return get_esmc_tokenizer("ESMC-300")


def synthetic_proteins(count: int, seed: int) -> tuple[list[str], list[int]]:
    """Random sequences whose class (0, 1, 2) sets which residue (A, L, K) is enriched."""
    rng = np.random.default_rng(seed)
    sequences, classes = [], []
    for index in range(count):
        favored = index % 3
        weights = np.ones(len(AMINO_ACIDS))
        weights[AMINO_ACIDS.index("ALK"[favored])] = 12.0
        length = int(rng.integers(12, 40))
        body = rng.choice(list(AMINO_ACIDS), size=length - 1, p=weights / weights.sum())
        sequences.append("M" + "".join(body))
        classes.append(favored)
    return sequences, classes


def residue_classes(sequence: str) -> list[int]:
    return [0 if residue in "AVILMFWC" else 1 if residue in "DEKRH" else 2 for residue in sequence]


def write_config(root: Path, name: str, level: str, kind: str, label_names: list[str], splits: dict[str, tuple[list[str], list]]) -> None:
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    for split, (sequences, labels) in splits.items():
        frame = pd.DataFrame({"id": [f"{name}_{split}_{index}" for index in range(len(sequences))], "sequence": sequences, "label": labels})
        frame.to_parquet(folder / f"{split}.parquet", index=False)
    metadata = {
        "config": name,
        "task_level": level,
        "task_type": kind,
        "num_labels": 1 if kind == "binary" else len(label_names),
        "label_names": label_names,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def synthetic_dataset(root: Path) -> None:
    """Four protein tasks (multiclass, regression, multilabel, binary) and one residue task over the same proteins."""
    sequences, classes = synthetic_proteins(150, seed=7)
    cuts = {"train": slice(0, 90), "validation": slice(90, 120), "test": slice(120, 150)}
    parts = {split: (sequences[cut], classes[cut]) for split, cut in cuts.items()}
    composition = ["percent_a", "percent_l", "percent_k"]

    def task(labels_of: object) -> dict[str, tuple[list[str], list]]:
        return {split: (seqs, [labels_of(seq, cls) for seq, cls in zip(seqs, cls_list)]) for split, (seqs, cls_list) in parts.items()}

    write_config(root, "localization", "protein", "multiclass", ["class_0", "class_1", "class_2"], task(lambda seq, cls: cls))
    write_config(root, "prot_param", "protein", "regression", composition,
                 task(lambda seq, cls: prot_param_values([seq], composition)[0].tolist()))
    write_config(root, "go_cc", "protein", "multilabel", ["a_rich", "l_rich"], task(lambda seq, cls: [int(cls == 0), int(cls == 1)]))
    write_config(root, "metal_ion_binding", "protein", "binary", ["k_rich"], task(lambda seq, cls: int(cls == 2)))
    write_config(root, "ssp_q3", "residue", "multiclass", ["hydrophobic", "charged", "other"], task(lambda seq, cls: residue_classes(seq)))


# -- sites and hooks ---------------------------------------------------------------------------------------------


def test_site_catalog_and_layouts() -> None:
    dims = ModelDims(d=64, f=256, heads=4, blocks=3)
    layouts = site_layouts(TapSpec(), dims)
    assert tuple(layouts) == ALL_SITES
    assert layouts["embedding"].layers == (0,) and layouts["final_norm"].layers == (0,)
    assert layouts["resid_post"].layers == (0, 1, 2)
    assert layouts["mlp_act"].width == 256 and layouts["q_rot"].width == 64
    assert layouts["q_rot"].heads == 4 and layouts["resid_post"].heads == 0
    assert layouts["attn_context"].head_names()[:2] == ["attn_context:0:h0", "attn_context:0:h1"]
    assert TapSpec(blocks=(-1,)).block_indices(dims) == (2,)
    assert site_layouts(TapSpec(per_head=False), dims)["q"].heads == 0
    assert parse_sites("residual,q") == ("embedding", "resid_post", "final_norm", "q")
    with pytest.raises(ValueError):
        TapSpec(sites=("nonsense",))


def test_taps_match_a_manual_forward(tiny_model: ESMplusplusModel, tokenizer: object) -> None:
    encoded = tokenizer(["MKTAYIAKQRQISFVKSHFSRQ", "MSEQ", "MLLAVLYCLAVF"], padding=True, return_tensors="pt")
    ids, mask = encoded["input_ids"], encoded["attention_mask"]  # (b, l)
    states = capture_states(tiny_model, ids, mask, TapSpec())
    dims = model_dims(tiny_model)
    assert dims == ModelDims(d=64, f=256, heads=4, blocks=3)
    assert len(states) == 2 + 15 * dims.blocks
    with torch.no_grad():
        output = tiny_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        valid = mask.bool()  # (b, l)
        torch.testing.assert_close(states[("embedding", 0)], tiny_model.embed(ids))
        for layer, block in enumerate(tiny_model.transformer.blocks):
            x = output.hidden_states[layer]  # the input to this block, (b, l, d)
            previous = states[("resid_post", layer - 1)] if layer else states[("embedding", 0)]
            torch.testing.assert_close(previous, x)
            attention = block.attn
            normed = attention.layernorm_qkv[0](x)
            torch.testing.assert_close(states[("attn_ln", layer)], normed)
            q, k, v = attention.layernorm_qkv[1](normed).chunk(3, dim=-1)
            torch.testing.assert_close(states[("q", layer)], attention.q_ln(q))
            torch.testing.assert_close(states[("k", layer)], attention.k_ln(k))
            torch.testing.assert_close(states[("v", layer)], v)
            q_rotated, k_rotated = attention._apply_rotary(attention.q_ln(q), attention.k_ln(k))
            torch.testing.assert_close(states[("q_rot", layer)], q_rotated)
            torch.testing.assert_close(states[("k_rot", layer)], k_rotated)

            def heads(values: torch.Tensor) -> torch.Tensor:
                return values.unflatten(-1, (dims.heads, dims.head_width)).transpose(1, 2)  # (b, h, l, d_h)

            logits = heads(q_rotated) @ heads(k_rotated).transpose(-1, -2) / math.sqrt(dims.head_width)  # (b, h, l, l)
            logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
            context = (logits.softmax(dim=-1) @ heads(v)).transpose(1, 2).flatten(-2)  # (b, l, d)
            torch.testing.assert_close(states[("attn_context", layer)][valid], context[valid], rtol=1e-4, atol=1e-5)
            attended = attention.out_proj(states[("attn_context", layer)])
            torch.testing.assert_close(states[("attn_out", layer)], attended)
            middle = x + attended / block.scaling_factor
            torch.testing.assert_close(states[("resid_mid", layer)], middle)
            mlp_in = block.ffn[0](middle)
            torch.testing.assert_close(states[("mlp_ln", layer)], mlp_in)
            gate, up = block.ffn[1](mlp_in).chunk(2, dim=-1)
            torch.testing.assert_close(states[("mlp_gate", layer)], gate)
            torch.testing.assert_close(states[("mlp_up", layer)], up)
            activated = F.silu(gate) * up
            torch.testing.assert_close(states[("mlp_act", layer)], activated)
            torch.testing.assert_close(states[("mlp_out", layer)], block.ffn[3](activated))
            torch.testing.assert_close(states[("resid_post", layer)], middle + block.ffn[3](activated) / block.scaling_factor)
        torch.testing.assert_close(states[("final_norm", 0)], output.last_hidden_state)
        torch.testing.assert_close(states[("final_norm", 0)], output.hidden_states[-1])


def test_hooks_are_removed(tiny_model: ESMplusplusModel, tokenizer: object) -> None:
    encoded = tokenizer(["MKTAYIAK"], padding=True, return_tensors="pt")
    capture_states(tiny_model, encoded["input_ids"], encoded["attention_mask"], TapSpec())
    hooked = [module for module in tiny_model.modules() if module._forward_hooks or module._forward_pre_hooks]
    assert hooked == []


# -- batching and stores -------------------------------------------------------------------------------------------


def test_token_batches_cover_every_sequence_once() -> None:
    sequences, _ = synthetic_proteins(60, seed=1)
    for shuffle in (False, True):
        batches = token_batches(sequences, max_tokens=200, shuffle=shuffle, seed=3)
        indices = [index for batch in batches for index in batch.indices]
        assert sorted(indices) == list(range(len(sequences)))
        for batch in batches:
            assert [sequences[index] for index in batch.indices] == list(batch.sequences)
            assert batch.padded_length == max(len(sequence) for sequence in batch.sequences) + SPECIAL_TOKENS
            assert len(batch.indices) * batch.padded_length <= 200 or len(batch.indices) == 1


def test_masked_mean_var_ignores_padding() -> None:
    generator = torch.Generator().manual_seed(0)
    X = torch.randn(2, 5, 3, generator=generator)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    pooled = masked_mean_var(X, mask)  # (2, 6)
    torch.testing.assert_close(pooled[0, :3], X[0, :3].mean(dim=0))
    torch.testing.assert_close(pooled[0, 3:], X[0, :3].var(dim=0, unbiased=False))
    torch.testing.assert_close(pooled[1, 3:], X[1].var(dim=0, unbiased=False))


def test_pooled_store_matches_pooling_each_sequence_alone(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    sequences, _ = synthetic_proteins(12, seed=2)
    spec = TapSpec(sites=("embedding", "k_rot", "attn_context", "mlp_act", "resid_post", "final_norm"))
    roots = {False: tmp_path / "raw", True: tmp_path / "normalized"}
    embed_pooled(tiny_model, tokenizer, sequences, spec, roots, "tiny", "trained", max_tokens=120, device=CPU, progress=False)
    raw, normalized = PooledTapStore(roots[False]), PooledTapStore(roots[True])
    assert raw.manifest.normalized is False and normalized.manifest.normalized is True
    rows = raw.rows_for(sequences)
    assert sorted(rows.tolist()) == list(range(len(sequences)))
    for name in spec.sites:
        stored = raw.load(name, rows)  # (n, L, 2 * w)
        stored_normalized = normalized.load(name, normalized.rows_for(sequences))
        for position, sequence in enumerate(sequences):
            encoded = tokenizer([sequence], padding=True, return_tensors="pt")
            states = capture_states(tiny_model, encoded["input_ids"], encoded["attention_mask"], spec)
            layers = raw.layers(name)
            expected = torch.stack([masked_mean_var(states[(name, layer)], encoded["attention_mask"])[0] for layer in layers])
            expected_normalized = torch.stack([
                masked_mean_var(F.normalize(states[(name, layer)], dim=-1), encoded["attention_mask"])[0] for layer in layers
            ])
            torch.testing.assert_close(stored[position].float(), expected, rtol=2e-2, atol=2e-3)
            torch.testing.assert_close(stored_normalized[position].float(), expected_normalized, rtol=2e-2, atol=2e-3)


def test_token_store_round_trip(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    sequences, _ = synthetic_proteins(9, seed=4)
    spec = TapSpec(sites=("q", "mlp_up", "resid_post"), blocks=(0, -1), per_head=False)
    embed_tokens(tiny_model, tokenizer, sequences, spec, tmp_path, "tiny", max_tokens=100, device=CPU)
    store = TokenTapStore(tmp_path)
    for sequence in sequences:
        encoded = tokenizer([sequence], padding=True, return_tensors="pt")
        states = capture_states(tiny_model, encoded["input_ids"], encoded["attention_mask"], spec)
        for name in spec.sites:
            stored = store.read(sequence, name)  # (L, r + 2, w)
            assert stored.shape[1] == len(sequence) + SPECIAL_TOKENS
            expected = torch.stack([states[(name, layer)][0] for layer in (0, 2)])
            torch.testing.assert_close(stored.float(), expected.to(torch.bfloat16).float(), rtol=2e-2, atol=2e-3)


def test_head_view_splits_mean_and_variance_by_head() -> None:
    pooled = torch.arange(2 * 3 * 16, dtype=torch.float32).view(2, 3, 16)  # n=2, L=3, w=8
    viewed = head_view(pooled, heads=4)  # (2, 12, 4)
    assert viewed.shape == (2, 12, 4)
    layer, head = 1, 2
    torch.testing.assert_close(viewed[:, layer * 4 + head, :2], pooled[:, layer, 4:6])
    torch.testing.assert_close(viewed[:, layer * 4 + head, 2:], pooled[:, layer, 12:14])


# -- probe bank ----------------------------------------------------------------------------------------------------


def reference_loss(params: dict[str, torch.Tensor], X: torch.Tensor, targets: torch.Tensor, loss: str, kind: str) -> torch.Tensor:
    """The summed per-group mean loss through autograd, for checking the bank's hand-written gradients."""
    if kind == "linear":
        Z = torch.einsum("ngw,gwc->ngc", X, params["weight"]) + params["bias"]
    else:
        hidden = F.gelu(torch.einsum("ngw,gwm->ngm", X, params["weight1"]) + params["bias1"])
        Z = torch.einsum("ngm,gmc->ngc", hidden, params["weight2"]) + params["bias2"]
    if loss == "multiclass":
        return sum(F.cross_entropy(Z[:, group], targets) for group in range(Z.shape[1]))
    expanded = targets.unsqueeze(1).expand_as(Z)
    if loss == "multilabel":
        return sum(F.binary_cross_entropy_with_logits(Z[:, group], expanded[:, group]) for group in range(Z.shape[1]))
    return sum(F.mse_loss(Z[:, group], expanded[:, group]) for group in range(Z.shape[1]))


@pytest.mark.parametrize("kind", ["linear", "mlp"])
@pytest.mark.parametrize("loss", ["multiclass", "multilabel", "regression"])
def test_probe_bank_gradients_match_autograd(kind: str, loss: str) -> None:
    generator = torch.Generator().manual_seed(5)
    groups, width, outputs, rows = 3, 6, 4, 9
    settings = ProbeSettings(kind=kind, hidden=7, dropout=0.0)
    bank = ProbeBank([f"p{g}" for g in range(groups)], width, outputs, loss, settings,
                     torch.full((groups,), 1e-3), torch.zeros(groups), seed=1, device=CPU)
    X = torch.randn(rows, groups, width, generator=generator)
    if loss == "multiclass":
        targets = torch.randint(0, outputs, (rows,), generator=generator)
    elif loss == "multilabel":
        targets = (torch.rand(rows, outputs, generator=generator) > 0.5).float()
    else:
        targets = torch.randn(rows, outputs, generator=generator)
    mean_loss = bank.accumulate(X, targets)
    params = {name: value.clone().requires_grad_(True) for name, value in bank.params.items()}
    total = reference_loss(params, X, targets, loss, kind)
    total.backward()
    torch.testing.assert_close(mean_loss.sum(), total.detach(), rtol=1e-5, atol=1e-6)
    for name, value in params.items():
        torch.testing.assert_close(bank.grads[name], value.grad, rtol=1e-4, atol=1e-6)


def test_probe_bank_adamw_matches_torch() -> None:
    generator = torch.Generator().manual_seed(6)
    rates, decays = torch.tensor([1e-2, 3e-3]), torch.tensor([0.1, 0.0])
    bank = ProbeBank(["a", "b"], 5, 3, "multiclass", ProbeSettings(), rates, decays, seed=2, device=CPU)
    reference = [
        {name: value[group].clone().requires_grad_(True) for name, value in bank.params.items()} for group in range(2)
    ]
    optimizers = [
        torch.optim.AdamW(list(reference[group].values()), lr=float(rates[group]), weight_decay=float(decays[group]))
        for group in range(2)
    ]
    for _ in range(4):
        X = torch.randn(8, 2, 5, generator=generator)
        targets = torch.randint(0, 3, (8,), generator=generator)
        bank.accumulate(X, targets)
        for group in range(2):
            for name, value in reference[group].items():
                value.grad = bank.grads[name][group].clone()
            optimizers[group].step()
        bank.step()
        for group in range(2):
            for name, value in reference[group].items():
                torch.testing.assert_close(bank.params[name][group], value.detach(), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("heads", [1, 4])
def test_shared_input_probes_match_the_per_group_path(heads: int) -> None:
    generator = torch.Generator().manual_seed(8)
    settings = TrainingSettings(learning_rates=(1e-2, 1e-3), weight_decays=(0.0,))
    layers, width, outputs, rows, slot = 3, 16, 5, 64, 1
    head_width = width // heads
    names = [f"s:{layer}:h{head}" for layer in range(layers) for head in range(heads)]
    bank = make_bank(names, head_width, outputs, "multiclass", settings, seed=0, device=CPU)
    mean = torch.randn(len(names), head_width, generator=generator)
    variance = torch.rand(len(names), head_width, generator=generator) + 0.5
    bank.set_standardization(mean.repeat(2, 1), variance.repeat(2, 1))
    reference = ProbeBank(bank.names, head_width, outputs, "multiclass", bank.settings, bank.learning_rates, bank.weight_decays, seed=0, device=CPU)
    reference.params = {name: value.clone() for name, value in bank.params.items()}
    reference.set_standardization(mean.repeat(2, 1), variance.repeat(2, 1))
    X = (2.0 * torch.randn(rows, width, generator=generator) + 1.0).to(torch.bfloat16)  # (n, w), as a hook delivers it
    targets = torch.randint(0, outputs, (rows,), generator=generator)
    index = torch.tensor([setting * layers * heads + slot * heads + head for setting in range(2) for head in range(heads)])
    starts = [setting * layers * heads + slot * heads for setting in range(2)]
    grouped = X.float().view(rows, heads, head_width)  # (n, H, d)

    losses = bank.shared_accumulate(X, targets, index, heads)
    expected = torch.cat([reference.accumulate(grouped, targets, start, start + heads) for start in starts])
    torch.testing.assert_close(losses, expected, rtol=2e-2, atol=2e-3)
    for name in ("weight", "bias"):
        torch.testing.assert_close(bank.grads[name], reference.grads[name], rtol=5e-2, atol=2e-3)
    Z = bank.shared_logits(X, index, heads)  # (n, K * H, C)
    expected_Z = torch.cat([reference.logits(grouped, start, start + heads) for start in starts], dim=1)
    torch.testing.assert_close(Z, expected_Z, rtol=2e-2, atol=3e-2)
    sums = bank.shared_loss_sums(X, targets, index, heads)
    expected_sums = torch.cat([reference.loss_sums(grouped, targets, start, start + heads) for start in starts])
    torch.testing.assert_close(sums, expected_sums, rtol=2e-2, atol=1e-1)


def test_observe_keeps_best_weights_and_stops_after_patience() -> None:
    bank = ProbeBank(["a", "b"], 3, 2, "multiclass", ProbeSettings(), torch.full((2,), 1e-2), torch.zeros(2), seed=0, device=CPU)
    assert bank.observe(torch.tensor([1.0, 1.0]), patience=2) == 2
    first = bank.best["weight"].clone()
    bank.params["weight"].add_(1.0)
    assert bank.observe(torch.tensor([0.5, 2.0]), patience=2) == 2  # a improves, b misses once
    torch.testing.assert_close(bank.best["weight"][0], bank.params["weight"][0])
    torch.testing.assert_close(bank.best["weight"][1], first[1])
    assert bank.observe(torch.tensor([0.6, 2.0]), patience=2) == 1  # b misses twice and stops
    assert bank.stopped.tolist() == [False, True]


def test_select_settings_keeps_each_probes_best_setting() -> None:
    settings = TrainingSettings(learning_rates=(1e-2, 1e-3), weight_decays=(0.0,))
    bank = make_bank(["x", "y", "z"], 4, 2, "multiclass", settings, seed=0, device=CPU)
    torch.testing.assert_close(bank.initial["weight"][:3], bank.initial["weight"][3:])  # one initialization per probe
    bank.best_loss = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4, 0.5])
    reduced, chosen = bank.select_settings(2)
    assert chosen.tolist() == [1, 0, 0]
    assert [name.split("|")[0] for name in reduced.names] == ["x", "y", "z"]
    torch.testing.assert_close(reduced.learning_rates, torch.tensor([1e-3, 1e-2, 1e-2]))
    torch.testing.assert_close(reduced.best["weight"][0], bank.best["weight"][3])


# -- scores --------------------------------------------------------------------------------------------------------


def test_mcc_matches_sklearn() -> None:
    from sklearn.metrics import matthews_corrcoef

    rng = np.random.default_rng(0)
    truth, predicted = rng.integers(0, 4, 200), rng.integers(0, 4, 200)
    predicted[:120] = truth[:120]
    confusion = torch.zeros(4, 4)
    for t, p in zip(truth, predicted):
        confusion[t, p] += 1
    assert float(multiclass_mcc(confusion)) == pytest.approx(matthews_corrcoef(truth, predicted), abs=1e-9)
    binary_truth, binary_predicted = truth % 2, predicted % 2
    counts = torch.tensor([
        np.sum((binary_predicted == 1) & (binary_truth == 1)), np.sum((binary_predicted == 1) & (binary_truth == 0)),
        np.sum((binary_predicted == 0) & (binary_truth == 1)), np.sum((binary_predicted == 0) & (binary_truth == 0)),
    ], dtype=torch.float64)
    assert float(binary_mcc(counts)) == pytest.approx(matthews_corrcoef(binary_truth, binary_predicted), abs=1e-9)


def test_score_accumulator_resamples_match_direct_scores() -> None:
    from sklearn.metrics import matthews_corrcoef

    generator = torch.Generator().manual_seed(3)
    proteins, groups, classes = 40, 2, 3
    plan = ResamplePlan.draw(proteins, count=3, fraction=0.5, seed=1)
    owners = torch.repeat_interleave(torch.arange(proteins), 5)  # five residues per protein
    targets = torch.randint(0, classes, (len(owners),), generator=generator)
    Z = torch.randn(len(owners), groups, classes, generator=generator)
    Z[:120, :, :] += 4.0 * F.one_hot(targets[:120], classes).unsqueeze(1)
    accumulator = ScoreAccumulator("multiclass", groups, classes, plan)
    indexed = ScoreAccumulator("multiclass", groups, classes, plan)
    for start in range(0, len(owners), 37):  # batches that split proteins arbitrarily
        accumulator.add(Z[start: start + 37], targets[start: start + 37], owners[start: start + 37])
        index = torch.unique(owners[start: start + 37], return_inverse=True)
        for group in range(groups):  # one group at a time, the batch's protein index computed once
            indexed.add(Z[start: start + 37, group: group + 1], targets[start: start + 37], owners[start: start + 37], group, index)
    scores = accumulator.scores()  # (4, 2)
    assert torch.equal(indexed.confusion, accumulator.confusion)
    for row in range(plan.rows):
        kept = plan.membership[row, owners].bool()
        for group in range(groups):
            expected = matthews_corrcoef(targets[kept].numpy(), Z[kept, group].argmax(-1).numpy())
            assert float(scores[row, group]) == pytest.approx(expected, abs=1e-9)

    multilabel = ScoreAccumulator("multilabel", groups, 2, plan)
    labels = (torch.rand(proteins, 2, generator=generator) > 0.5).float()
    logits = torch.randn(proteins, groups, 2, generator=generator)
    multilabel.add(logits, labels, torch.arange(proteins))
    for row in range(plan.rows):
        kept = plan.membership[row].bool()
        expected = matthews_corrcoef(labels[kept].reshape(-1).numpy(), (logits[kept, 0] > 0).float().reshape(-1).numpy())
        assert float(multilabel.scores()[row, 0]) == pytest.approx(expected, abs=1e-9)

    regression = ScoreAccumulator("regression", groups, 3, plan)
    values = torch.randn(proteins, 3, generator=generator)
    predictions = values.unsqueeze(1) + 0.3 * torch.randn(proteins, groups, 3, generator=generator)
    regression.add(predictions, values, torch.arange(proteins))
    cosine = F.cosine_similarity(predictions[:, 1], values, dim=-1)
    assert float(regression.scores()[0, 1]) == pytest.approx(float(cosine.mean()), abs=1e-6)
    per_target = (predictions[:, 1, 2] * values[:, 2]).sum() / (predictions[:, 1, 2].norm() * values[:, 2].norm())
    assert float(regression.elementwise()[0, 1, 2]) == pytest.approx(float(per_target), abs=1e-6)


# -- sequence controls ---------------------------------------------------------------------------------------------


def test_scramble_keeps_the_first_residue_and_the_composition() -> None:
    sequence = "MKTAYIAKQRQISFVKSHFSRQ"
    shuffled = scramble(sequence, scramble_seed(sequence, 0))
    assert shuffled[0] == "M" and sorted(shuffled) == sorted(sequence) and shuffled != sequence
    assert scramble(sequence, scramble_seed(sequence, 0)) == shuffled
    label = np.array([0, 1, 1, 0, 1])
    assert sorted(shuffled_labels(label, 3).tolist()) == sorted(label.tolist())


def test_prot_param_values_are_percent_composition() -> None:
    values = prot_param_values(["MAAAK"], ["length", "percent_a", "percent_k", "percent_w"])
    np.testing.assert_allclose(values[0], [5.0, 60.0, 20.0, 0.0])


def test_untrained_copy_resets_every_parameter(tiny_model: ESMplusplusModel) -> None:
    reset = untrained_copy(tiny_model, seed=0)
    block = reset.transformer.blocks[0]
    assert torch.equal(block.ffn[0].weight, torch.ones_like(block.ffn[0].weight))
    assert not torch.equal(block.ffn[1].weight, tiny_model.transformer.blocks[0].ffn[1].weight)
    assert block.ffn[1].weight.std() > 0.03  # PyTorch's default init, wider than the model's std 0.02 init


# -- end to end ----------------------------------------------------------------------------------------------------


def conditions_of(scores: pd.DataFrame) -> set[tuple[str, str, str]]:
    return set(zip(scores["plm"], scores["probe"], scores["control"]))


def headline_scores(scores: pd.DataFrame) -> pd.Series:
    """Each probe's full-test score with the trained model, the trained probe and the original input."""
    # Bracket access throughout: `frame.resample` is a DataFrame method, not the column.
    kept = (scores["plm"] == "trained") & (scores["probe"] == "trained") & (scores["control"] == "original") & (scores["resample"] == 0)
    return scores[kept].set_index("probe_name")["score"]


def test_protein_tasks_end_to_end(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    synthetic_dataset(tmp_path / "data")
    names = ["localization", "prot_param", "go_cc", "metal_ion_binding"]
    datasets = {name: load_concept(name, local_dir=tmp_path / "data") for name in names}
    spec = TapSpec(sites=("embedding", "q", "mlp_act", "resid_post"))
    root = tmp_path / "run"
    write_stores(tiny_model, tokenizer, "tiny", datasets, root, spec, controls=True, seed=0, max_tokens=256, device=CPU)
    for key in STORE_VARIANTS:
        assert (store_path(root, *key) / "manifest.json").exists()
    assert PooledTapStore(store_path(root, "untrained", "scrambled")).manifest.variant == "untrained_scrambled"
    settings = TrainingSettings(learning_rates=(3e-2, 3e-3), weight_decays=(0.0,), batch_size=16, max_epochs=40, patience=6, resamples=3)
    train_tasks(names, root, "linear", spec, settings, CPU, read_task=datasets.__getitem__, read_model=lambda: (tiny_model, tokenizer))

    for name in names:
        folder = root / "results" / "linear" / name
        scores = pd.read_parquet(folder / "scores.parquet")
        full = scores[scores["family"] == "full"]
        assert conditions_of(full) == {(plm, probe, control) for plm in PLM_STATES for probe in PROBE_STATES for control in CONTROLS}
        heads = scores[scores["family"] == "head"]
        assert set(heads["probe_name"]) == {f"q:{layer}:h{head}" for layer in range(3) for head in range(4)}
        assert conditions_of(heads) == {("trained", "trained", "original")}
        assert np.isfinite(scores["score"]).all()
        headline = headline_scores(full)
        assert set(headline.index) == {"embedding:0", "q:0", "q:1", "q:2", "mlp_act:0", "mlp_act:1", "mlp_act:2", "resid_post:0", "resid_post:1", "resid_post:2"}
        assert headline["embedding:0"] > 0.8, name
        training = pd.read_parquet(folder / "training.parquet")
        assert set(training["learning_rate"]) <= {3e-2, 3e-3}
        weights = load_file(str(folder / "probes.safetensors"))
        outputs = datasets[name].task.outputs
        assert weights["embedding.full.best.weight"].shape == (1, 128, outputs)
        assert weights["q.head.best.weight"].shape == (12, 32, outputs)
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert record["rows"] == {"train": 90, "validation": 30, "test": 30}
    for name in ("prot_param", "go_cc"):
        elementwise = pd.read_parquet(root / "results" / "linear" / name / "elementwise.parquet")
        assert set(elementwise["label_index"]) == set(range(datasets[name].task.outputs))
    # A second call finds every task finished and trains nothing.
    train_tasks(names, root, "linear", spec, settings, CPU, read_task=lambda name: pytest.fail("retrained"), read_model=lambda: pytest.fail("reloaded"))


def test_protein_shared_path_matches_the_per_group_path() -> None:
    generator = torch.Generator().manual_seed(11)
    settings = TrainingSettings(learning_rates=(1e-2, 1e-3), weight_decays=(0.0,))
    probes, width, outputs, rows = 3, 8, 4, 40
    bank = make_bank([f"s:{layer}" for layer in range(probes)], width, outputs, "multilabel", settings, seed=0, device=CPU)
    mean = torch.randn(probes, width, generator=generator)
    variance = torch.rand(probes, width, generator=generator) + 0.5
    bank.set_standardization(mean.repeat(2, 1), variance.repeat(2, 1))
    X = (torch.randn(rows, probes, width, generator=generator) + mean).to(torch.bfloat16)  # (n, P, w), as a store holds it
    targets = (torch.rand(rows, outputs, generator=generator) > 0.6).float()
    shared = validation_loss(bank, X, targets, settings_count=2, batch=16)  # the shared-input path of a linear bank
    per_group = torch.cat([bank.loss_sums(X.float(), targets, start, start + probes) for start in (0, probes)]) / (rows * outputs)
    torch.testing.assert_close(shared, per_group, rtol=2e-2, atol=2e-3)
    plan = ResamplePlan.draw(rows, count=0)
    fused, reference = (ScoreAccumulator("multilabel", bank.groups, outputs, plan, per_label=False) for _ in range(2))
    score_bank(bank, "current", X, targets, fused)
    reference.add(torch.cat([bank.logits(X.float(), start, start + probes) for start in (0, probes)], dim=1), targets, torch.arange(rows))
    torch.testing.assert_close(fused.scores(), reference.scores(), rtol=0, atol=0.1)


def test_selection_prefers_the_fastest_label_within_tolerance() -> None:
    rows = []
    for label, seconds, lift in (("slow", 100.0, 0.0), ("fast", 30.0, -0.002)):
        for task in ("a", "b"):
            for rate, score in ((1e-3, 0.80), (1e-2, 0.70)):
                for probe in ("p:0", "p:1"):
                    rows.append({"label": label, "task": task, "family": "full", "probe_name": probe, "learning_rate": rate,
                                 "weight_decay": 0.0, "validation_score": score + lift, "task_seconds": seconds / 2,
                                 "batch_size": 64 if label == "slow" else 512, "max_tokens": 0})
    sweeps = pd.DataFrame(rows)
    ranked = rank_settings(sweeps)
    assert ranked.iloc[0]["label"] == "slow" and ranked.iloc[0]["learning_rate"] == 1e-3
    assert ranked.iloc[0]["mean_shortfall"] == pytest.approx(0.0)
    assert ranked.set_index(["label", "learning_rate"]).loc[("fast", 1e-3), "seconds"] == pytest.approx(30.0)
    assert choose_setting(ranked, tolerance=0.005)["label"] == "fast"
    assert choose_setting(ranked, tolerance=0.001)["label"] == "slow"


def test_sweep_then_select(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    synthetic_dataset(tmp_path / "data")
    names = ["localization", "go_cc", "ssp_q3"]
    datasets = {name: load_concept(name, local_dir=tmp_path / "data") for name in names}
    spec = TapSpec(sites=("embedding", "q", "resid_post"))
    root = tmp_path / "run"
    write_stores(tiny_model, tokenizer, "tiny", {name: datasets[name] for name in names[:2]}, root, spec, controls=False, seed=0, max_tokens=256, device=CPU)
    grid = {"learning_rates": (3e-2, 3e-3), "weight_decays": (0.0, 0.1)}
    for label, batch in (("b16", 16), ("b32", 32)):
        settings = TrainingSettings(**grid, batch_size=batch, max_tokens=300, max_epochs=8, patience=4, controls=False)
        train_tasks(names, root, label, spec, settings, CPU, read_task=datasets.__getitem__, read_model=lambda: (tiny_model, tokenizer), sweep=True)
    sweep = pd.read_parquet(root / "sweeps" / "b16" / "go_cc" / "sweep.parquet")
    assert len(sweep) == 4 * (1 + 3 + 3 + 3 * 4)  # every grid setting of embedding, q, resid_post, and q's heads
    assert set(zip(sweep["learning_rate"], sweep["weight_decay"])) == {(3e-2, 0.0), (3e-2, 0.1), (3e-3, 0.0), (3e-3, 0.1)}
    assert np.isfinite(sweep["validation_score"]).all()
    assert not (root / "results").exists()  # a sweep scores validation only and writes no results
    residue = pd.read_parquet(root / "sweeps" / "b16" / "ssp_q3" / "sweep.parquet")
    assert len(residue) == 4 * (1 + 3 + 3 + 3 * 4)
    assert residue[residue["probe_name"] == "embedding:0"]["validation_score"].max() > 0.9
    ranked = rank_settings(read_sweeps(root, ["b16", "b32"]))
    assert len(ranked) == 8 and set(ranked["tasks"]) == {3}
    assert ranked["mean_shortfall"].min() >= 0.0


def test_residue_task_end_to_end(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    synthetic_dataset(tmp_path / "data")
    data = load_concept("ssp_q3", local_dir=tmp_path / "data")
    spec = TapSpec(sites=("embedding", "k_rot", "mlp_gate", "resid_post"))
    settings = TrainingSettings(learning_rates=(3e-2,), weight_decays=(0.0,), max_tokens=300, max_epochs=12, patience=4, resamples=3)
    root = tmp_path / "run"
    train_tasks(["ssp_q3"], root, "linear", spec, settings, CPU, read_task=lambda name: data, read_model=lambda: (tiny_model, tokenizer))
    folder = root / "results" / "linear" / "ssp_q3"
    scores = pd.read_parquet(folder / "scores.parquet")
    full = scores[scores["family"] == "full"]
    assert conditions_of(full) == {(plm, probe, control) for plm in PLM_STATES for probe in PROBE_STATES for control in CONTROLS}
    assert headline_scores(full)["embedding:0"] > 0.9
    heads = scores[scores["family"] == "head"]
    assert set(heads["probe_name"]) == {f"k_rot:{layer}:h{head}" for layer in range(3) for head in range(4)}
    weights = load_file(str(folder / "probes.safetensors"))
    assert weights["mlp_gate.full.best.weight"].shape == (3, 256, 3)
    assert weights["k_rot.head.best.weight"].shape == (12, 16, 3)
    # without controls the test split is scored in token-budget batches; the probes and their scores are the same
    plain = replace(settings, controls=False)
    train_tasks(["ssp_q3"], root, "plain", spec, plain, CPU, read_task=lambda name: data, read_model=lambda: (tiny_model, tokenizer))
    alone = pd.read_parquet(root / "results" / "plain" / "ssp_q3" / "scores.parquet")
    assert conditions_of(alone) == {("trained", "trained", "original")}
    pd.testing.assert_series_equal(headline_scores(alone[alone["family"] == "full"]).sort_index(), headline_scores(full).sort_index(), atol=1e-3)


# -- interventions and latent views --------------------------------------------------------------------------------


PHOSPHO_NAMES = ("-", "5-glutamyl_glycerylphosphorylethanolamine", "Phosphohistidine", "Phosphoserine", "Phosphothreonine", "Phosphotyrosine")


def phospho_task(proteins: list[tuple[str, str, list[int]]]) -> ConceptData:
    """A phosphorylation task holding every protein in each split, as (id, sequence, per-residue labels)."""
    task = ConceptTask("phosphorylation", "residue", "multiclass", len(PHOSPHO_NAMES), PHOSPHO_NAMES)
    split = ConceptSplit(
        ids=[protein_id for protein_id, _, _ in proteins],
        sequences=[sequence for _, sequence, _ in proteins],
        labels=[np.asarray(labels) for _, _, labels in proteins],
    )
    return ConceptData(task, {"train": split, "validation": ConceptSplit([], [], []), "test": ConceptSplit([], [], [])})


def test_scanning_mutants_label_the_predicted_outcome() -> None:
    sequences, classes = scanning_mutants(PhosphoProtein("p", "MSTAY", (1,)))
    assert sequences[0] == "MSTAY" and classes[0] == ACTIVE
    assert len(sequences) == 1 + 3 * 3
    labelled = dict(zip(sequences[1:], classes[1:].tolist()))
    assert (labelled["MATAY"], labelled["MGTAY"], labelled["METAY"]) == (INACTIVE, INACTIVE, ACTIVE)
    assert {labelled[sequence] for sequence in ("MSAAY", "MSGAY", "MSEAY", "MSTAA", "MSTAG", "MSTAE")} == {UNKNOWN}


def test_phosphoproteins_keep_annotated_s_t_and_y() -> None:
    data = phospho_task([
        ("a", "MSTY", [0, 3, 0, 5]),  # phosphoserine at 1, phosphotyrosine at 3
        ("b", "MKKK", [0, 0, 0, 0]),  # no site
        ("c", "MSHS", [0, 0, 2, 4]),  # a phosphohistidine, and a threonine label on a serine: neither counts
        ("d", "MSTY", [0, 3, 0, 0]),  # a repeat of a's sequence
    ])
    assert phosphoproteins(data) == [PhosphoProtein("a", "MSTY", (1, 3))]


def test_pooled_means_match_a_masked_mean(tiny_model: ESMplusplusModel, tokenizer: object) -> None:
    sequences = ["MKTAYIAK", "MSEQ", "MLLAVLYC"]
    spec = TapSpec(sites=("embedding", "attn_out", "mlp_act", "resid_post"), per_head=False)
    pooled = pooled_means(tiny_model, tokenizer, sequences, spec, CPU, max_tokens=30)  # one padded batch
    layouts = site_layouts(spec, model_dims(tiny_model))
    for index, sequence in enumerate(sequences):
        encoded = tokenizer([sequence], return_tensors="pt")
        states = capture_states(tiny_model, encoded["input_ids"], encoded["attention_mask"], spec)
        for site, layout in layouts.items():
            expected = torch.stack([states[(site, layer)][0].mean(dim=0) for layer in layout.layers])  # (L, w)
            torch.testing.assert_close(pooled[site][index], expected, rtol=1e-4, atol=1e-5)


def test_fit_logistic_matches_sklearn() -> None:
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = rng.integers(0, 3, size=60)
    X[:, 0] += y  # a weak signal, so the penalty matters
    reference = LogisticRegression(C=1.0, max_iter=5000, tol=1e-12).fit(X, y)
    X_groups = torch.tensor(X, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)  # two identical probes
    weight, bias = fit_logistic(X_groups, torch.from_numpy(y), classes=3)
    probabilities = torch.softmax(torch.baddbmm(bias, X_groups, weight), dim=-1)  # (2, 60, 3)
    for group in range(2):
        np.testing.assert_allclose(probabilities[group].numpy(), reference.predict_proba(X), atol=1e-4)
    # fewer mutants than columns, the regime of a protein's scan: separable, so the penalty sets the solution
    wide = rng.normal(size=(30, 80))
    labels = rng.integers(0, 3, size=30)
    wide[:, :3] += np.eye(3)[labels]
    reference = LogisticRegression(C=1.0, max_iter=5000, tol=1e-12).fit(wide, labels)
    groups = torch.tensor(np.stack([wide, wide[:, ::-1].copy()]), dtype=torch.float32)  # (2, 30, 80), columns reversed
    weight, bias = fit_logistic(groups, torch.from_numpy(labels), classes=3, budget=2**20)  # one probe per chunk
    probabilities = torch.softmax(torch.baddbmm(bias, groups, weight), dim=-1)
    np.testing.assert_allclose(probabilities[0].numpy(), reference.predict_proba(wide), atol=1e-4)
    np.testing.assert_allclose(probabilities[1].numpy(), probabilities[0].numpy(), atol=1e-5)
    np.testing.assert_allclose(weight[1].flip(0).numpy(), weight[0].numpy(), atol=1e-4)


def test_phospho_interventions_end_to_end(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    proteins = []
    for index in range(3):
        sequence = "M" + "".join(rng.choice(list("STYAKLE"), size=23))
        stys = [position for position, residue in enumerate(sequence) if residue in "STY"]
        labels = [0] * len(sequence)
        for position in stys[:2]:
            labels[position] = PHOSPHO_NAMES.index({"S": "Phosphoserine", "T": "Phosphothreonine", "Y": "Phosphotyrosine"}[sequence[position]])
        proteins.append((f"p{index}", sequence, labels))
    data = phospho_task(proteins)
    spec = TapSpec(sites=("embedding", "mlp_act", "resid_post"), per_head=False)
    untrained = untrained_copy(tiny_model, seed=0)
    out = tmp_path / "phospho"
    scores = run_phospho_interventions(tiny_model, untrained, tokenizer, data, spec, out, CPU, proteins=2, max_tokens=512)
    assert scores["protein"].nunique() == 2 and len(list(out.glob("*.parquet"))) == 2
    assert set(zip(scores["plm"], scores["labels"])) == {("trained", "true"), ("trained", "shuffled"), ("untrained", "true")}
    assert set(scores["inputs"]) == {"raw", "standardized"}
    assert len(scores) == 2 * 3 * 2 * (1 + 3 + 3)  # proteins, conditions, input treatments, site layers
    assert scores["mcc"].between(-1.0, 1.0).all()
    tests = paired_tests(scores)
    assert len(tests) == 2 * (1 + 3 + 3) and tests["p_bonferroni"].between(0.0, 1.0).all()
    written = {path.name: path.stat().st_mtime_ns for path in out.glob("*.parquet")}
    again = run_phospho_interventions(tiny_model, untrained, tokenizer, data, spec, out, CPU, proteins=2)
    assert {path.name: path.stat().st_mtime_ns for path in out.glob("*.parquet")} == written  # resumed, nothing refit
    pd.testing.assert_frame_equal(again, scores)


def test_principal_components_and_layer_projections(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    generator = torch.Generator().manual_seed(4)
    X = torch.randn(200, 4, generator=generator) * torch.tensor([5.0, 2.0, 0.5, 0.1])  # (n, w), widest along axis 0
    mean, axes, explained = principal_components(X)
    assert axes.shape == (4, 2) and float(explained[0]) > float(explained[1]) > 0.0
    assert abs(float(axes[0, 0])) > 0.95
    synthetic_dataset(tmp_path / "data")
    data = load_concept("go_cc", local_dir=tmp_path / "data")
    spec = TapSpec(sites=("resid_post",))
    root = tmp_path / "run"
    write_stores(tiny_model, tokenizer, "tiny", {"go_cc": data}, root, spec, controls=True, seed=0, max_tokens=256, device=CPU)
    groups = label_sets(data, top=2)
    assert set(groups.values()) <= {"a rich", "l rich", "none", "other"}
    test_sequences = data.splits["test"].sequences
    frame = project_layers(
        PooledTapStore(store_path(root, "trained", "original")), groups, "resid_post", [0, 2],
        PooledTapStore(store_path(root, "trained", "scrambled")), test_sequences,
    )
    assert len(frame) == 2 * (len(groups) + len(test_sequences))
    assert (frame["group"] == SCRAMBLED).sum() == 2 * len(test_sequences)
    assert np.isfinite(frame[["pc1", "pc2", "explained_1", "explained_2"]].to_numpy()).all()
    main(["pca", "--root", str(root), "--tasks", "go_cc", "--layers", "0,2", "--top", "2", "--local-data", str(tmp_path / "data")])
    written = pd.read_parquet(root / "latent" / "pca" / "go_cc.parquet")
    pd.testing.assert_frame_equal(written.drop(columns="task"), frame)
    main(["figures", "--root", str(root), "--out", str(tmp_path / "figures")])
    assert (tmp_path / "figures" / "pca_go_cc_resid_post.png").exists()


def synthetic_scores(task: str, sites: tuple[str, ...], layers: int, heads: int, resamples: int, seed: int) -> pd.DataFrame:
    """A scores table shaped like a run's: every condition of every full-width probe, and the heads of `q`."""
    rng = np.random.default_rng(seed)
    rows = []
    for site in sites:
        for layer in [0] if site in ("embedding", "final_norm") else range(layers):
            for plm in PLM_STATES:
                for probe in PROBE_STATES:
                    for control in CONTROLS:
                        rows += [(task, "full", f"{site}:{layer}", resample, plm, probe, control, rng.uniform()) for resample in range(resamples + 1)]
            if site == "q":
                for head in range(heads):
                    rows += [(task, "head", f"q:{layer}:h{head}", resample, "trained", "trained", "original", rng.uniform()) for resample in range(resamples + 1)]
    return pd.DataFrame(rows, columns=["task", "family", "probe_name", "resample", "plm", "probe", "control", "score"])


def test_figures_from_result_tables(tmp_path: Path) -> None:
    root, out = tmp_path / "run", tmp_path / "figures"
    sites = ("embedding", "q", "mlp_act", "resid_post", "final_norm")
    for index, task in enumerate(("localization", "prot_param")):
        folder = root / "results" / "linear" / task
        folder.mkdir(parents=True)
        scores = synthetic_scores(task, sites, layers=3, heads=4, resamples=3, seed=index)
        scores.to_parquet(folder / "scores.parquet", index=False)
        (folder / "run.json").write_text(json.dumps({"task": {"name": task, "label_names": ["length", "mass"]}}), encoding="utf-8")
    elementwise = synthetic_scores("prot_param", ("embedding", "resid_post"), layers=3, heads=0, resamples=3, seed=2)
    elementwise = elementwise[(elementwise["plm"] == "trained") & (elementwise["probe"] == "trained") & (elementwise["control"] == "original")]
    pd.concat([elementwise.assign(label_index=0), elementwise.assign(label_index=1)]).drop(columns=["plm", "probe", "control"]).to_parquet(
        root / "results" / "linear" / "prot_param" / "elementwise.parquet", index=False)
    tests = pd.DataFrame([
        {"inputs": inputs, "site": site, "layer": layer, "trained_true": 0.5, "trained_true_std": 0.1, "trained_shuffled": 0.0,
         "trained_shuffled_std": 0.05, "untrained_true": 0.1, "untrained_true_std": 0.05}
        for inputs in ("raw", "standardized") for site in sites for layer in ([0] if site in ("embedding", "final_norm") else range(3))
    ])
    (root / "interventions" / "phospho").mkdir(parents=True)
    tests.to_parquet(root / "interventions" / "phospho" / "tests.parquet", index=False)
    main(["figures", "--root", str(root), "--out", str(out), "--blocks", "3"])
    for name in ("depth_curves", "site_maps", "best_states", "peak_layers", "controls", "heads", "prot_param_targets", "phospho"):
        assert (out / f"{name}.png").stat().st_size > 10_000, name
    table = pd.read_csv(out / "best_states.csv")
    assert list(table["task"]) == ["prot_param", "localization"]
    summary = pd.read_parquet(out / "summary.parquet")
    trained = figures.headline(summary)
    assert len(trained[trained["family"] == "full"]) == 2 * (2 + 3 * 3)
    best = trained[(trained["task"] == "localization") & (trained["family"] == "full")]["full"].max()
    assert table.set_index("task").loc["localization", "any_score"] == pytest.approx(best, abs=1e-4)
    ranked = pd.DataFrame([
        {"label": label, "learning_rate": rate, "weight_decay": decay, "mean_shortfall": rate * decay, "batch_size": batch, "max_tokens": 0}
        for label, batch in (("b64", 64), ("b256", 256)) for rate in (1e-3, 1e-2) for decay in (0.0, 0.1)
    ])
    assert figures.sweep_curves({"protein": ranked}, out / "sweeps.png").exists()
