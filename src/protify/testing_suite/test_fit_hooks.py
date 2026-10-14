"""`FitHooks`, the one seam through which a training dataset changes how `TrainerMixin._train` fits a probe.

Each test trains a tiny MLP probe for real, on CPU, so the hooks are seen in the order and with the values the
trainer gives them, not in a mock of it.
"""

import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import EarlyStoppingCallback, Trainer

from protify.probes.fit_hooks import FitHooks
from protify.probes.get_probe import ProbeArguments
from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
from protify.probes.trainers import TrainerArguments, TrainerMixin


WIDTH, ROWS = 4, 16


class _Rows(Dataset):
    """Pooled embeddings whose label is the sign of the first feature, so the probe has something to learn."""

    def __init__(self, seed: int, fit_hooks: FitHooks | None = None) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.embeddings = torch.randn(ROWS, WIDTH, generator=generator)  # (n, d)
        self.labels = (self.embeddings[:, 0] > 0).long()  # (n,)
        if fit_hooks is not None:
            self.fit_hooks = fit_hooks

    def __len__(self) -> int:
        return ROWS

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"embeddings": self.embeddings[index], "labels": self.labels[index]}


class _PosteriorProbe(MLPProbe):
    """An MLP probe that defines `fit_posterior`, recording the rows it was given."""

    def fit_posterior(self, rows) -> None:
        self.posterior_rows = rows


class _RecordingTrainer(Trainer):
    def evaluate(self, eval_dataset=None, *args, **kwargs):
        self.evaluated = [*getattr(self, "evaluated", []), eval_dataset]
        return super().evaluate(eval_dataset, *args, **kwargs)


class _RecordingHooks(FitHooks):
    def __init__(self, reserves_test: bool = False) -> None:
        self.reserves_test = reserves_test
        self.calls: list[str] = []
        self.sentinel_rows = [object()]

    def before_training(self, train_dataset, training_arguments, probe_args, stopping) -> None:
        self.calls.append("before_training")
        self.stopping = dict(stopping)
        self.training_arguments = training_arguments

    def build_trainer(self, **trainer_arguments) -> Trainer:
        self.calls.append("build_trainer")
        self.trainer = _RecordingTrainer(**trainer_arguments)
        return self.trainer

    def stopping_callback(self, stopping) -> EarlyStoppingCallback:
        self.calls.append("stopping_callback")
        return super().stopping_callback(stopping)

    def after_training(self, trainer, train_dataset) -> None:
        self.calls.append("after_training")
        assert trainer is self.trainer

    def posterior_rows(self, trainer, train_dataset):
        self.calls.append("posterior_rows")
        return self.sentinel_rows


def _fit(tmp_path: Path, train: _Rows, valid: _Rows, test: _Rows, model: MLPProbe):
    mixin = TrainerMixin(TrainerArguments(
        model_save_dir=str(tmp_path), task_type="singlelabel", num_epochs=2, patience=1, probe_batch_size=4,
        torch_compile=False, balanced_regression_metrics=False, make_plots=False, seed=7,
    ))
    mixin.probe_args = ProbeArguments(probe_type="mlp", input_size=WIDTH, hidden_size=8, n_layers=0, num_labels=2, dropout=0.0)
    return mixin._train(model, train, valid, test, data_collator=None, tokenizer=None, log_id="hooks",
                        model_name="toy", data_name="sign")


def _probe(posterior: bool = False) -> MLPProbe:
    torch.manual_seed(0)
    config = MLPProbeConfig(input_size=WIDTH, hidden_size=8, n_layers=0, num_labels=2, dropout=0.0, task_type="singlelabel")
    return _PosteriorProbe(config) if posterior else MLPProbe(config)


def test_hooks_run_in_order_and_the_posterior_reads_their_rows(tmp_path: Path) -> None:
    hooks = _RecordingHooks()
    model, valid_metrics, test_metrics, _, _ = _fit(tmp_path, _Rows(0, hooks), _Rows(1), _Rows(2), _probe(posterior=True))

    assert hooks.calls == ["before_training", "stopping_callback", "build_trainer", "after_training", "posterior_rows"]
    assert hooks.stopping["early_stopping_patience"] == 1
    assert model.posterior_rows is hooks.sentinel_rows
    assert "eval_loss" in valid_metrics and "test_loss" in test_metrics


def test_the_first_evaluation_reads_test_unless_the_hooks_reserve_it(tmp_path: Path) -> None:
    valid, test = _Rows(1), _Rows(2)
    plain = _RecordingHooks(reserves_test=False)
    _fit(tmp_path / "plain", _Rows(0, plain), valid, test, _probe())
    reserving = _RecordingHooks(reserves_test=True)
    _fit(tmp_path / "reserving", _Rows(0, reserving), valid, test, _probe())

    assert plain.trainer.evaluated[0] is test
    assert reserving.trainer.evaluated[0] is valid
    for trainer in (plain.trainer, reserving.trainer):
        # Each epoch's evaluation passes None, which is the trainer's eval set, validation; the final
        # evaluation reads validation by name.
        assert trainer.eval_dataset is valid
        assert all(dataset is None for dataset in trainer.evaluated[1:-1])
        assert trainer.evaluated[-1] is valid


def test_a_dataset_without_hooks_fits_with_the_defaults(tmp_path: Path) -> None:
    model, valid_metrics, test_metrics, _, _ = _fit(tmp_path, _Rows(0), _Rows(1), _Rows(2), _probe(posterior=True))

    assert isinstance(model.posterior_rows, DataLoader)  # the default posterior rows: the training loader
    assert len(model.posterior_rows.dataset) == ROWS
    assert "eval_loss" in valid_metrics and "test_loss" in test_metrics


def test_the_hooks_set_how_many_draws_an_epoch_reads(tmp_path: Path) -> None:
    class HalfEpochs(FitHooks):
        def draws_per_epoch(self, train_dataset) -> int:
            return len(train_dataset) // 2

    arguments = TrainerArguments(model_save_dir=str(tmp_path), torch_compile=False, probe_batch_size=4)
    probe_args = ProbeArguments(probe_type="mlp", input_size=WIDTH, num_labels=2)
    arguments.for_datasets(probe_args, _Rows(0, HalfEpochs()), _Rows(1), _Rows(2))
    assert arguments.train_data_size == ROWS // 2
    arguments.for_datasets(probe_args, _Rows(0), _Rows(1), _Rows(2))
    assert arguments.train_data_size == ROWS
