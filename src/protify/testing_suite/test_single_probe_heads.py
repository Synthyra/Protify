"""The single-probe experiment's heads, per-seed extras, and methods, each against what defines it.

- `linear` reads the clamped joined input through one Linear layer.
- `faithful` keeps the squared-error mean: every shared parameter gets exactly the gradient a
  plain squared-error head gets, and `beta_nll` at beta 0 is the Gaussian negative log likelihood.
- The quantile clip's bounds are the training rows' quantiles and nothing else's.
- The last-layer Laplace variance grows away from the training data, and for a linear
  regression head equals the closed-form Bayesian linear regression predictive variance.
- The loss predictor's folds are stratified, and no fold head ever trains on a row it scores.
- `temperature_pnorm` chooses its norm order on V_fit alone.
- Scheme T refits every new method's V_fit choices and, given the maps `build_methods` fitted,
  reproduces `score_method` on each repeat.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset


try:
    from src.protify.data.dataset_classes import EmbeddingStandardizer
    from src.protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
    from src.protify.probes.projected_probe import ProjectedProbe, ProjectedProbeConfig, gaussian_negative_log_likelihood
    from src.protify.uncertainty import methods
    from src.protify.uncertainty.batched_calibration import temperature_objective, variance_objective
    from src.protify.uncertainty.cell import cell_from_mapping
    from src.protify.uncertainty.conformal_sets import SPREAD_FLOOR_FRACTION
    from src.protify.uncertainty.exchangeable import (
        _map_objectives,
        exchangeable_scores,
        fit_maps,
        method_scores,
        plan_runs,
        reference_splits,
        score_plan,
    )
    from src.protify.uncertainty.laplace import fit_kfac, latent_variance
    from src.protify.uncertainty.loss_head import cross_fitted_losses, per_row_losses, stratified_folds
    from src.protify.uncertainty.methods import LAPLACE_PRIOR_PRECISIONS, PNORM_ORDERS, build_methods, pnorm_aurc, pnorm_uncertainty
    from src.protify.uncertainty.metrics import CLASSIFICATION_KINDS
    from src.protify.uncertainty.predictions import SplitPredictions, load_seed_predictions, save_seed_predictions
    from src.protify.uncertainty.quantile_clip import ClippedStandardizer, fit_clipped_standardizer
    from src.protify.uncertainty.report import score_method
    from src.protify.uncertainty.resampling import exchangeable_design
except ImportError:
    from protify.data.dataset_classes import EmbeddingStandardizer
    from protify.probes.get_probe import ProbeArguments, get_probe, rebuild_probe_from_saved_config
    from protify.probes.projected_probe import ProjectedProbe, ProjectedProbeConfig, gaussian_negative_log_likelihood
    from protify.uncertainty import methods
    from protify.uncertainty.batched_calibration import temperature_objective, variance_objective
    from protify.uncertainty.cell import cell_from_mapping
    from protify.uncertainty.conformal_sets import SPREAD_FLOOR_FRACTION
    from protify.uncertainty.exchangeable import (
        _map_objectives,
        exchangeable_scores,
        fit_maps,
        method_scores,
        plan_runs,
        reference_splits,
        score_plan,
    )
    from protify.uncertainty.laplace import fit_kfac, latent_variance
    from protify.uncertainty.loss_head import cross_fitted_losses, per_row_losses, stratified_folds
    from protify.uncertainty.methods import LAPLACE_PRIOR_PRECISIONS, PNORM_ORDERS, build_methods, pnorm_aurc, pnorm_uncertainty
    from protify.uncertainty.metrics import CLASSIFICATION_KINDS
    from protify.uncertainty.predictions import SplitPredictions, load_seed_predictions, save_seed_predictions
    from protify.uncertainty.quantile_clip import ClippedStandardizer, fit_clipped_standardizer
    from protify.uncertainty.report import score_method
    from protify.uncertainty.resampling import exchangeable_design


ROWS = 240
SEEDS = (0, 1)
LEVELS = (0.8, 0.9)
LABEL_TYPES = {"binary": "singlelabel", "multiclass": "singlelabel", "multilabel": "multilabel", "regression": "regression"}
CLASSES = {"binary": 2, "multiclass": 4, "multilabel": 5}


class Rows(Dataset):
    """Feature rows and labels, collated as the joined-branch collator hands them to a probe."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        # features: (n, d), labels: (n,) or (n, c)
        self.features = torch.as_tensor(features, dtype=torch.float32)  # (n, d)
        self.labels = torch.as_tensor(labels)  # (n,) or (n, c)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]  # (d,), (c,) or ()


def collate(batch):
    return {"embeddings": torch.stack([item[0] for item in batch]), "labels": torch.stack([item[1] for item in batch])}


def _projected(**options) -> ProjectedProbe:
    settings = {"input_size": 12, "projection_size": 8, "dropout": 0.0, "branch_widths": [5, 7], "num_labels": 1, "task_type": "regression"}
    return ProjectedProbe(ProjectedProbeConfig(**{**settings, **options}))


# The linear head.

def test_the_linear_head_is_one_linear_layer_on_the_clamped_input():
    torch.manual_seed(0)
    arguments = ProbeArguments(probe_type="projected_linear", input_size=12, num_labels=3, task_type="singlelabel", branch_widths=[5, 7])
    probe = get_probe(arguments)
    assert probe.trunk == "linear" and probe.config.output == "linear"
    assert sum(parameter.numel() for parameter in probe.parameters()) == 12 * 3 + 3
    rows = 20.0 * torch.randn(6, 12)  # (6, 12)
    expected = F.linear(rows.clamp(-10.0, 10.0), probe.output.weight, probe.output.bias)  # (6, 3)
    assert torch.equal(probe(rows).logits, expected)

    rebuilt = rebuild_probe_from_saved_config("projected_linear", False, probe.config.to_dict())
    assert rebuilt.trunk == "linear"
    unclamped = _projected(trunk="linear", input_clip=None, num_labels=3, task_type="singlelabel")
    assert torch.equal(unclamped.trunk_features(rows), rows)


# Faithful and beta-NLL.

def test_faithful_trains_the_mean_exactly_as_squared_error_does():
    torch.manual_seed(1)
    faithful = _projected(output="faithful")
    plain = _projected()
    plain.load_state_dict({name: value for name, value in faithful.state_dict().items() if not name.startswith("variance_output")})
    rows, labels = torch.randn(16, 12), torch.randn(16)

    faithful(rows, labels=labels).loss.backward()
    plain(rows, labels=labels).loss.backward()

    shared = dict(plain.named_parameters())
    for name, parameter in faithful.named_parameters():
        if name.startswith("variance_output"):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
            continue
        assert torch.allclose(parameter.grad, shared[name].grad, rtol=1e-6, atol=1e-7), name

    mean, variance = faithful.eval().predict_with_variance(rows)  # (16, 1), (16,)
    assert mean.shape == (16, 1) and variance.shape == (16,) and bool((variance > 0).all())


def test_beta_nll_at_beta_zero_is_the_gaussian_negative_log_likelihood():
    torch.manual_seed(2)
    probe = _projected(output="beta_nll", beta=0.0)
    rows, labels = torch.randn(16, 12), torch.randn(16)
    loss = probe(rows, labels=labels).loss
    mean, variance = probe.eval().predict_with_variance(rows)
    expected = F.gaussian_nll_loss(mean.reshape(-1), labels, variance)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert gaussian_negative_log_likelihood(mean.reshape(-1), variance, labels).mean().item() == pytest.approx(expected.item(), rel=1e-6)

    with pytest.raises(ValueError, match="regression"):
        _projected(output="faithful", num_labels=2, task_type="singlelabel")


# The quantile clip.

def test_quantile_clip_bounds_are_the_training_rows_quantiles_alone():
    rng = np.random.default_rng(3)
    train = rng.normal(size=(500, 6)).astype(np.float32)  # (n, d)
    held_out = 50.0 + rng.normal(size=(100, 6)).astype(np.float32)  # rows the bounds must not see
    standardizer = EmbeddingStandardizer.fit_numpy(train)
    standardized = standardizer.transform_numpy(train)  # (n, d)
    clipped = fit_clipped_standardizer(standardizer, standardized, (0.001, 0.999))

    assert isinstance(clipped, ClippedStandardizer)
    assert np.allclose(clipped.lower.numpy(), np.quantile(standardized, 0.001, axis=0), atol=1e-6)
    assert np.allclose(clipped.upper.numpy(), np.quantile(standardized, 0.999, axis=0), atol=1e-6)
    transformed = clipped.transform_numpy(held_out)  # (100, d)
    assert np.array_equal(transformed, np.broadcast_to(clipped.upper.numpy(), transformed.shape))
    assert torch.equal(clipped.transform_tensor(torch.from_numpy(held_out[0])), torch.from_numpy(transformed[0]))
    subset = clipped.select_features(np.array([1, 4]))
    assert np.array_equal(subset.upper.numpy(), clipped.upper.numpy()[[1, 4]])


# The last-layer Laplace posterior.

def _linear_regression_head(width: int) -> ProjectedProbe:
    return _projected(input_size=width, branch_widths=None, trunk="linear", input_clip=None)


def test_laplace_regression_variance_is_the_closed_form_bayesian_linear_regression_one():
    torch.manual_seed(4)
    rng = np.random.default_rng(4)
    width, rows = 6, 80
    features = rng.normal(size=(rows, width)).astype(np.float32)  # (n, d)
    labels = (features @ rng.normal(size=width) + 0.5 * rng.normal(size=rows)).astype(np.float32)  # (n,)
    queries = (3.0 * rng.normal(size=(10, width))).astype(np.float32)  # (q, d)
    probe = _linear_regression_head(width)
    device = torch.device("cpu")

    factors = fit_kfac(probe, Rows(features, labels), collate, kind="regression", batch_size=32, device=device)
    variance = latent_variance(
        probe, factors, Rows(queries, np.zeros(10, dtype=np.float32)), collate, LAPLACE_PRIOR_PRECISIONS,
        kind="regression", batch_size=4, device=device,
    )  # (q, L)

    with torch.no_grad():
        residuals = probe(torch.from_numpy(features)).logits.reshape(-1).double().numpy() - labels  # (n,)
    assert factors.noise_variance == pytest.approx(float(np.mean(residuals ** 2)), rel=1e-9)
    design = np.concatenate([features, np.ones((rows, 1))], axis=1).astype(np.float64)  # (n, d + 1)
    query_design = np.concatenate([queries, np.ones((10, 1))], axis=1).astype(np.float64)  # (q, d + 1)
    for index, precision in enumerate(LAPLACE_PRIOR_PRECISIONS):
        posterior = design.T @ design / factors.noise_variance + precision * np.eye(width + 1)  # (d + 1, d + 1)
        expected = np.einsum("qi,ij,qj->q", query_design, np.linalg.inv(posterior), query_design)  # (q,)
        assert np.allclose(variance[:, index], expected, rtol=2e-5), precision


@pytest.mark.parametrize("kind", ["regression", "multiclass", "multilabel"])
def test_laplace_variance_grows_away_from_the_training_data(kind):
    torch.manual_seed(5)
    rng = np.random.default_rng(5)
    width = 8
    features = rng.normal(size=(200, width)).astype(np.float32)  # (n, d)
    features[:, 4:] *= 0.05  # the training rows barely vary along the last four features
    classes = {"regression": 1, "multiclass": 3, "multilabel": 4}[kind]
    task = {"regression": "regression", "multiclass": "singlelabel", "multilabel": "multilabel"}[kind]
    probe = _projected(input_size=width, branch_widths=None, trunk="linear", input_clip=None, num_labels=classes, task_type=task)
    if kind == "regression":
        labels = rng.normal(size=200).astype(np.float32)
    elif kind == "multiclass":
        labels = rng.integers(0, classes, size=200)
    else:
        labels = (rng.random((200, classes)) < 0.4).astype(np.float32)
    device = torch.device("cpu")
    factors = fit_kfac(probe, Rows(features, labels), collate, kind=kind, batch_size=64, device=device)

    direction = np.zeros(width, dtype=np.float32)
    direction[6] = 1.0
    queries = np.stack([features.mean(axis=0) + distance * direction for distance in (0.0, 1.0, 4.0, 16.0)])  # (4, d)
    placeholder = np.zeros((4,) + labels.shape[1:], dtype=labels.dtype)
    variance = latent_variance(
        probe, factors, Rows(queries, placeholder), collate, LAPLACE_PRIOR_PRECISIONS, kind=kind, batch_size=4, device=device,
    )  # (4, L) or (4, L, c)
    expected_shape = (4, len(LAPLACE_PRIOR_PRECISIONS)) + ((classes,) if kind == "multiclass" else ())
    assert variance.shape == expected_shape
    assert np.all(np.diff(variance, axis=0) > 0.0)


# The loss predictor's folds.

def test_folds_are_stratified_seeded_and_cover_every_row():
    labels = np.repeat([0, 1, 2], [30, 20, 10])
    folds = stratified_folds(list(labels), "singlelabel", 5, seed=42)
    assert np.array_equal(folds, stratified_folds(list(labels), "singlelabel", 5, seed=42))
    assert not np.array_equal(folds, stratified_folds(list(labels), "singlelabel", 5, seed=43))
    for label, count in ((0, 30), (1, 20), (2, 10)):
        assert np.array_equal(np.bincount(folds[labels == label], minlength=5), np.full(5, count // 5))


def test_no_fold_head_trains_on_a_row_it_scores():
    folds = stratified_folds(list(np.arange(40) % 2), "singlelabel", 4, seed=0)
    trained_on, scored = [], []

    def fit(rows):
        trained_on.append(set(rows.tolist()))
        return len(trained_on) - 1

    def score(head, rows):
        scored.append((head, set(rows.tolist())))
        return np.full(rows.shape[0], float(head))

    losses = cross_fitted_losses(folds, fit, score)
    assert len(trained_on) == 4
    for head, rows in scored:
        assert not rows & trained_on[head]
        assert rows | trained_on[head] == set(range(40))
    assert set().union(*(rows for _, rows in scored)) == set(range(40))
    assert np.array_equal(losses, folds.astype(np.float64))


def test_per_row_losses_are_the_training_losses():
    logits = np.array([[2.0, -1.0, 0.5], [0.0, 0.0, 3.0]])
    labels = np.array([0, 1])
    expected = F.cross_entropy(torch.tensor(logits), torch.tensor(labels), reduction="none").numpy()
    assert np.allclose(per_row_losses("multiclass", labels, logits), expected)
    targets = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    expected = F.binary_cross_entropy_with_logits(torch.tensor(logits), torch.tensor(targets), reduction="none").mean(dim=1).numpy()
    assert np.allclose(per_row_losses("multilabel", targets, logits), expected)
    assert np.allclose(per_row_losses("regression", np.array([1.0, 2.0]), np.array([1.5, 0.0])), [0.25, 4.0])


# Synthetic seed predictions with every extra.

def _labels(kind: str, rng: np.random.Generator, rows: int = ROWS) -> np.ndarray:
    if kind == "regression":
        return rng.normal(size=rows).astype(np.float32)  # (n,)
    if kind == "multilabel":
        labels = (rng.random((rows, CLASSES[kind])) < 0.3).astype(np.float32)  # (n, c)
        labels[labels.sum(axis=1) == 0, 0] = 1.0
        return labels  # (n, c)
    return rng.integers(0, CLASSES[kind], size=rows)  # (n,)


def _split(kind: str, labels: np.ndarray, rng: np.random.Generator) -> SplitPredictions:
    """One seed's predictions on one set of rows, with a Laplace variance and a loss prediction."""
    # labels: (n,) or (n, c)
    rows = labels.shape[0]
    grid = len(LAPLACE_PRIOR_PRECISIONS)
    spread = rng.uniform(0.2, 2.0, size=rows).astype(np.float32)  # (n,)
    scales = np.logspace(1.0, -3.0, grid).astype(np.float32)  # (L,)
    if kind == "regression":
        outputs = (labels + spread * rng.normal(size=rows)).astype(np.float32)  # (n,)
        laplace = (spread[:, None] ** 2 * scales[None, :]).astype(np.float32)  # (n, L)
    else:
        target = labels if kind == "multilabel" else np.eye(CLASSES[kind])[labels]  # (n, c)
        outputs = (2.0 * target - 1.0 + spread[:, None] * rng.normal(size=target.shape)).astype(np.float32)  # (n, c)
        per_row = spread[:, None] * scales[None, :]  # (n, L)
        laplace = per_row if kind == "multilabel" else per_row[:, :, None] * rng.uniform(0.5, 1.5, size=(1, 1, target.shape[1]))
        laplace = laplace.astype(np.float32)  # (n, L) or (n, L, c)
    loss = np.log(spread ** 2 + 0.1 * rng.random(rows)).astype(np.float32)  # (n,)
    distance = rng.gamma(2.0, size=rows).astype(np.float32)  # (n,)
    return SplitPredictions(labels=labels, outputs=outputs, nn_distance=distance, laplace_variance=laplace, loss_prediction=loss)


def _seed_predictions(kind: str, seed: int = 11) -> dict[int, dict[str, SplitPredictions]]:
    rng = np.random.default_rng(seed)
    labels = {split: _labels(kind, rng, rows) for split, rows in (("v_fit", 120), ("v_cal", 80), ("test", 100))}
    return {run: {split: _split(kind, labels[split], rng) for split in labels} for run in SEEDS}


def test_new_fields_round_trip_and_older_files_load_without_them(tmp_path):
    predictions = _seed_predictions("multiclass")[SEEDS[0]]
    path = str(tmp_path / "seed.npz")
    save_seed_predictions(path, predictions)
    loaded = load_seed_predictions(path)
    assert np.array_equal(loaded["test"].laplace_variance, predictions["test"].laplace_variance)
    assert np.array_equal(loaded["v_fit"].loss_prediction, predictions["v_fit"].loss_prediction)

    older = {split: SplitPredictions(labels=value.labels, outputs=value.outputs) for split, value in predictions.items()}
    save_seed_predictions(path, older)
    assert load_seed_predictions(path)["test"].laplace_variance is None


def test_pnorm_ignores_a_shift_of_the_logits_as_softmax_does():
    rng = np.random.default_rng(14)
    logits = rng.normal(size=(50, 5))  # (n, c)
    shifted = logits + rng.normal(scale=10.0, size=(50, 1))  # (n, c)
    for order in PNORM_ORDERS:
        assert np.allclose(pnorm_uncertainty(shifted, order), pnorm_uncertainty(logits, order))

    # Centered, two logits are (-d, d), so a binary task's score carries no ranking.
    binary = rng.normal(size=(50, 2))  # (n, 2)
    assert np.allclose(pnorm_uncertainty(binary, 2.0), -(2.0 ** -0.5))


def test_pnorm_chooses_its_order_on_v_fit_alone():
    kind = "multiclass"
    seed_predictions = _seed_predictions(kind)
    method = next(method for method in build_methods(seed_predictions, "projected", kind, pnorm=True) if method.name == "temperature_pnorm")
    assert method.score_only
    rng = np.random.default_rng(12)
    for run in method.runs:
        splits = seed_predictions[run.seed]
        fit = splits["v_fit"]
        expected = PNORM_ORDERS[int(np.argmin([pnorm_aurc(fit.outputs, fit.labels, order) for order in PNORM_ORDERS]))]
        assert run.fitted["p"] == expected
        assert np.array_equal(run.splits["test"].uncertainty, pnorm_uncertainty(splits["test"].outputs, expected))

        # Other splits' outputs change, and the order stays.
        shuffled = {
            name: split if name == "v_fit" else SplitPredictions(**{**vars(split), "outputs": rng.permutation(split.outputs)})
            for name, split in splits.items()
        }
        again = next(method for method in build_methods({run.seed: shuffled}, "projected", kind, pnorm=True) if method.name == "temperature_pnorm")
        assert again.runs[0].fitted["p"] == expected

    assert "temperature_pnorm" not in {method.name for method in build_methods(seed_predictions, "projected", kind)}
    for other in ("binary", "multilabel", "regression"):
        built = build_methods(_seed_predictions(other), "projected", other, pnorm=True)
        assert "temperature_pnorm" not in {method.name for method in built}, other


@pytest.mark.parametrize("kind", ["binary", "multilabel", "regression"])
def test_every_new_method_fits_on_v_fit_alone(kind, monkeypatch):
    seed_predictions = _seed_predictions(kind)
    seen = []
    fit_temperature = methods.fit_temperature
    fit_variance = methods.fit_variance_calibration

    def spy_temperature(logits, labels, **options):
        seen.append(np.asarray(labels).copy())
        return fit_temperature(logits, labels, **options)

    def spy_variance(mean, variance, labels):
        seen.append(np.asarray(labels).copy())
        return fit_variance(mean, variance, labels)

    monkeypatch.setattr(methods, "fit_temperature", spy_temperature)
    monkeypatch.setattr(methods, "fit_variance_calibration", spy_variance)
    built = {method.name: method for method in build_methods(seed_predictions, "linear", kind, pnorm=True)}
    assert {"laplace_temperature", "loss_head", "temperature"} <= set(built)
    v_fit = seed_predictions[SEEDS[0]]["v_fit"].labels
    assert seen and all(np.array_equal(labels, v_fit) for labels in seen)
    laplace = built["laplace_temperature"].runs[0].fitted
    assert laplace["prior_precision"] in LAPLACE_PRIOR_PRECISIONS


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel", "regression"])
def test_scheme_t_refits_the_new_methods_and_matches_score_method(kind):
    rng = np.random.default_rng(13)
    labels = _labels(kind, rng)
    test_splits = {seed: _split(kind, labels, rng) for seed in SEEDS}
    repeats = 3
    design = exchangeable_design(labels, LABEL_TYPES[kind], "toy", repeats, seed=0)
    train_label_std = 1.0 if kind == "regression" else 0.0
    repeat_scores = exchangeable_scores(test_splits, "projected", kind, design, LEVELS, train_label_std, None, pnorm=True)
    assert all(check["passed"] for check in repeat_scores.checks), [check for check in repeat_scores.checks if not check["passed"]]
    names = {method.name for method in repeat_scores.methods}
    assert {"laplace_temperature", "loss_head"} <= names
    assert ("temperature_pnorm" in names) == (kind == "multiclass")

    fit_labels = labels.astype(np.int64) if kind in CLASSIFICATION_KINDS else labels
    compared = 0
    for repeat in range(repeats):
        splits = reference_splits(test_splits, design, repeat)
        built = build_methods(splits, "projected", kind, pnorm=True)
        plans = plan_runs(built, splits, "projected", kind)
        roles = splits[SEEDS[0]]
        role_labels = {role: roles[role].labels for role in ("v_fit", "v_cal", "test")}
        window = slice(repeat, repeat + 1)
        for method in built:
            if method.name not in ("laplace_temperature", "loss_head", "temperature_pnorm"):
                continue
            reference = score_method(method, kind, role_labels, LEVELS, train_label_std, None)
            runs = [plan for plan in plans if plan.method == method.name]
            given = [
                score_plan(
                    plan, kind, fit_labels, design, window, {key: np.array([value]) for key, value in plan.fitted.items()},
                    LEVELS, SPREAD_FLOOR_FRACTION * train_label_std, None,
                )
                for plan in runs
            ]
            mine = method_scores(runs, given)[0]
            for metric, values in mine.overall.items():
                if metric in reference["test"] and reference["test"][metric] is not None:
                    assert float(values[0]) == pytest.approx(reference["test"][metric], rel=1e-6, abs=1e-9), (method.name, metric)
                    compared += 1
            for plan in runs:
                batched = fit_maps(plan, kind, fit_labels, design.fit_rows[window])
                reference_map = {key: np.array([value]) for key, value in plan.fitted.items()}
                objectives = _map_objectives(plan, kind, fit_labels, design.fit_rows[window], batched, reference_map)
                assert objectives[0] <= objectives[1] + 1e-6 * max(1.0, abs(objectives[1])), plan.method
    assert compared > 10


def test_cell_settings_refuse_what_the_head_cannot_do():
    mapping = {
        "cell_id": "toy", "experiment_id": "toy", "output_dir": "toy",
        "uncertainty": {"branch_specs": ["ESM2-8:mean"], "head": "sngp", "laplace": True},
    }
    with pytest.raises(ValueError, match="one Linear"):
        cell_from_mapping(mapping)
    with pytest.raises(ValueError, match="loss_head_folds"):
        cell_from_mapping({**mapping, "uncertainty": {"branch_specs": ["ESM2-8:mean"], "loss_head_folds": 1}})
    with pytest.raises(ValueError, match="weight_decay"):
        cell_from_mapping({**mapping, "uncertainty": {"branch_specs": ["ESM2-8:mean"], "weight_decay": -1.0}})
    cell = cell_from_mapping({**mapping, "uncertainty": {"branch_specs": ["ESM2-8:mean"], "head": "linear", "laplace": True, "loss_head_folds": 5}})
    assert cell.uncertainty.weight_decay is None and cell.uncertainty.pnorm_uncertainty is False


def test_objective_helpers_agree_with_the_batched_ones():
    rng = np.random.default_rng(14)
    logits = rng.normal(size=(50, 3)).astype(np.float32)
    labels = rng.integers(0, 3, size=50)
    rows = np.arange(50)[None]  # (1, 50)
    batched = temperature_objective(logits, labels, True, rows, np.array([1.7]))[0]
    assert methods.temperature_nll(logits, labels, "multiclass", 1.7) == pytest.approx(batched, rel=1e-10)
    mean, variance, targets = rng.normal(size=50), rng.uniform(0.1, 1.0, size=50), rng.normal(size=50)
    batched = variance_objective(mean, variance, targets, rows, np.array([0.8]), np.array([0.3]))[0]
    assert methods.variance_nll(mean, variance, targets, 0.8, 0.3) == pytest.approx(batched, rel=1e-10)
