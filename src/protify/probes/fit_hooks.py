"""What a caller outside Protify may change about one probe fit.

`TrainerMixin._train` reads the hooks a training dataset carries as `fit_hooks`, and these
defaults when it carries none: a Hugging Face `Trainer` that reads the split once per epoch,
early stopping on the validation loss, and the test split evaluated before training and after.
A subclass changes only the steps it overrides.
"""

from collections.abc import Iterable, Mapping
from typing import Any
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments


class FitHooks:
    # A frozen release reads its test labels only after its checkpoint is chosen: the first
    # evaluation reads validation instead, and a multilabel release with observed labels selects
    # on the observed loss alone, leaving its complete metrics to the saved predictions.
    reserves_test: bool = False

    def draws_per_epoch(self, train_dataset: Any) -> int:
        """Training rows one epoch reads, which sets the evaluation cadence."""
        return len(train_dataset)

    def build_trainer(self, **trainer_arguments: Any) -> Trainer:
        return Trainer(**trainer_arguments)

    def stopping_callback(self, stopping: Mapping[str, int | float]) -> TrainerCallback:
        """The callback that ends training and chooses the checkpoint kept."""
        return EarlyStoppingCallback(**stopping)

    def before_training(self, train_dataset: Any, training_arguments: TrainingArguments, probe_args: Any,
                        stopping: Mapping[str, int | float]) -> None:
        """Runs once the training arguments resolve, before the trainer is built."""

    def after_training(self, trainer: Trainer, train_dataset: Any) -> None:
        """Runs when training ends, before any posterior fit and the final evaluations."""

    def posterior_rows(self, trainer: Trainer, train_dataset: Any) -> Iterable[Any]:
        """The batches a probe that defines `fit_posterior` fits its posterior on."""
        return trainer.get_train_dataloader()


def fit_hooks_of(train_dataset: Any) -> FitHooks:
    hooks = getattr(train_dataset, 'fit_hooks', None)
    return FitHooks() if hooks is None else hooks
