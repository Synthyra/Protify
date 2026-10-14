"""Tap probing on a tiny random ESMC: every hooked state against a manual forward, the stores, the probe bank's
gradients and optimizer, the scores, and both trainers end to end on small synthetic tasks."""

import json
import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from pathlib import Path
from safetensors.torch import load_file

try:
    from src.protify.base_models.esmc import get_esmc_tokenizer
    from src.protify.interpretability.capture import capture_states, model_dims
    from src.protify.interpretability.cli import STORE_VARIANTS, store_path, train_tasks, write_stores
    from src.protify.interpretability.concepts import load_concept, prot_param_values, scramble, scramble_seed, shuffled_labels
    from src.protify.interpretability.pooled_store import PooledTapStore, embed_pooled, masked_mean_var
    from src.protify.interpretability.probe_bank import ProbeBank, ProbeSettings
    from src.protify.interpretability.protein_probing import head_view
    from src.protify.interpretability.residue_probing import untrained_copy
    from src.protify.interpretability.scores import ResamplePlan, ScoreAccumulator, binary_mcc, multiclass_mcc
    from src.protify.interpretability.sites import ALL_SITES, ModelDims, TapSpec, parse_sites, site_layouts
    from src.protify.interpretability.token_batches import SPECIAL_TOKENS, token_batches
    from src.protify.interpretability.token_store import TokenTapStore, embed_tokens
    from src.protify.interpretability.training import CONTROLS, PLM_STATES, PROBE_STATES, TrainingSettings, make_bank
except ImportError:
    from protify.base_models.esmc import get_esmc_tokenizer
    from protify.interpretability.capture import capture_states, model_dims
    from protify.interpretability.cli import STORE_VARIANTS, store_path, train_tasks, write_stores
    from protify.interpretability.concepts import load_concept, prot_param_values, scramble, scramble_seed, shuffled_labels
    from protify.interpretability.pooled_store import PooledTapStore, embed_pooled, masked_mean_var
    from protify.interpretability.probe_bank import ProbeBank, ProbeSettings
    from protify.interpretability.protein_probing import head_view
    from protify.interpretability.residue_probing import untrained_copy
    from protify.interpretability.scores import ResamplePlan, ScoreAccumulator, binary_mcc, multiclass_mcc
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
    for start in range(0, len(owners), 37):  # batches that split proteins arbitrarily
        accumulator.add(Z[start: start + 37], targets[start: start + 37], owners[start: start + 37])
    scores = accumulator.scores()  # (4, 2)
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
        full = scores[scores.family == "full"]
        conditions = set(zip(full.plm, full.probe, full.control))
        assert conditions == {(plm, probe, control) for plm in PLM_STATES for probe in PROBE_STATES for control in CONTROLS}
        heads = scores[scores.family == "head"]
        assert set(heads.probe_name) == {f"q:{layer}:h{head}" for layer in range(3) for head in range(4)}
        assert set(zip(heads.plm, heads.probe, heads.control)) == {("trained", "trained", "original")}
        assert np.isfinite(scores.score).all()
        headline = full[(full.plm == "trained") & (full.probe == "trained") & (full.control == "original") & (full.resample == 0)]
        assert set(headline.probe_name) == {"embedding:0", "q:0", "q:1", "q:2", "mlp_act:0", "mlp_act:1", "mlp_act:2", "resid_post:0", "resid_post:1", "resid_post:2"}
        assert headline.set_index("probe_name").score["embedding:0"] > 0.8, name
        training = pd.read_parquet(folder / "training.parquet")
        assert set(training.learning_rate) <= {3e-2, 3e-3}
        weights = load_file(str(folder / "probes.safetensors"))
        outputs = datasets[name].task.outputs
        assert weights["embedding.full.best.weight"].shape == (1, 128, outputs)
        assert weights["q.head.best.weight"].shape == (12, 32, outputs)
        record = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        assert record["rows"] == {"train": 90, "validation": 30, "test": 30}
    for name in ("prot_param", "go_cc"):
        elementwise = pd.read_parquet(root / "results" / "linear" / name / "elementwise.parquet")
        assert set(elementwise.label_index) == set(range(datasets[name].task.outputs))
    # A second call finds every task finished and trains nothing.
    train_tasks(names, root, "linear", spec, settings, CPU, read_task=lambda name: pytest.fail("retrained"), read_model=lambda: pytest.fail("reloaded"))


def test_residue_task_end_to_end(tiny_model: ESMplusplusModel, tokenizer: object, tmp_path: Path) -> None:
    synthetic_dataset(tmp_path / "data")
    data = load_concept("ssp_q3", local_dir=tmp_path / "data")
    spec = TapSpec(sites=("embedding", "k_rot", "mlp_gate", "resid_post"))
    settings = TrainingSettings(learning_rates=(3e-2,), weight_decays=(0.0,), max_tokens=300, max_epochs=12, patience=4, resamples=3)
    root = tmp_path / "run"
    train_tasks(["ssp_q3"], root, "linear", spec, settings, CPU, read_task=lambda name: data, read_model=lambda: (tiny_model, tokenizer))
    folder = root / "results" / "linear" / "ssp_q3"
    scores = pd.read_parquet(folder / "scores.parquet")
    full = scores[scores.family == "full"]
    assert set(zip(full.plm, full.probe, full.control)) == {
        (plm, probe, control) for plm in PLM_STATES for probe in PROBE_STATES for control in CONTROLS
    }
    headline = full[(full.plm == "trained") & (full.probe == "trained") & (full.control == "original") & (full.resample == 0)]
    assert headline.set_index("probe_name").score["embedding:0"] > 0.9
    heads = scores[scores.family == "head"]
    assert set(heads.probe_name) == {f"k_rot:{layer}:h{head}" for layer in range(3) for head in range(4)}
    weights = load_file(str(folder / "probes.safetensors"))
    assert weights["mlp_gate.full.best.weight"].shape == (3, 256, 3)
    assert weights["k_rot.head.best.weight"].shape == (12, 16, 3)
