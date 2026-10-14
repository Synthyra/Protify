"""The rigorous benchmark list and the two datasets registered for it.

These datasets need no new loading code: `process_datasets` already turns a variable
length list of label indices into a multi-hot vector, and `_select_train_valid_test_splits`
already carves a validation split out of train when a dataset ships only train and test.
What these tests pin is the registration itself, and the task coverage that makes the list
worth running as a set.
"""

import json
import pytest


try:
    from src.protify.data.supported_datasets import (
        get_dataset_source,
        resolve_dataset_name,
        rigorous_benchmark,
        supported_datasets,
    )
except ImportError:
    from protify.data.supported_datasets import (
        get_dataset_source,
        resolve_dataset_name,
        rigorous_benchmark,
        supported_datasets,
    )


EXPECTED_SOURCES = {
    'gold-ppi': 'Synthyra/bernett_gold_ppi',
    'PPA-ppi': 'Synthyra/ppi_affinity',
    'designed-binder': 'Synthyra/designed_binder_metrics',
    'GO-rigor': 'Synthyra/GO-Rigorous',
    'Subcellular': 'GleghornLab/SL_13',
    'optimal-ph-rigor': 'GleghornLab/optimal_ph_rigor',
    'enzyme-kcat': 'GleghornLab/enzyme_kcat',
}


def test_every_benchmark_entry_resolves_to_a_source():
    for data_name in rigorous_benchmark:
        assert resolve_dataset_name(data_name) in supported_datasets, data_name
        assert get_dataset_source(data_name) == EXPECTED_SOURCES[data_name]


def test_the_benchmark_is_exactly_the_seven_intended_datasets():
    """A silent addition or removal would change what a reported comparison covers."""
    assert set(rigorous_benchmark) == set(EXPECTED_SOURCES)
    assert len(rigorous_benchmark) == len(set(rigorous_benchmark))


@pytest.mark.parametrize('data_name', ['designed-binder', 'GO-rigor'])
def test_the_private_datasets_are_registered(data_name):
    """Both live under the Synthyra organization and need HF_TOKEN, but must be nameable."""
    assert data_name in supported_datasets
    assert supported_datasets[data_name].startswith('Synthyra/')


def test_the_benchmark_is_selectable_as_a_data_names_keyword():
    """--data_names rigorous_benchmark has to expand the same way the other lists do."""
    try:
        from src.protify.data import data_mixin
    except ImportError:
        from protify.data import data_mixin

    source = data_mixin.__file__
    with open(source, encoding='utf-8') as handle:
        text = handle.read()
    assert "data_names[0] == 'rigorous_benchmark'" in text


def test_result_rows_record_the_probe_that_actually_ran():
    """Two arms of the campaign differ only in what the probe read, so the row must say.

    A full-width SAE run and a feature-selected one share a model name and a dataset, so
    without input_size and feature_selection in the row they are the same row twice. And
    probe_args has to win over logger_args, which is a startup snapshot of the command
    line and still shows the default for anything the run changed.
    """
    from types import SimpleNamespace

    try:
        from src.protify.logger import PROBE_RESULT_IDENTITY_KEYS, MetricsLogger
    except ImportError:
        from protify.logger import PROBE_RESULT_IDENTITY_KEYS, MetricsLogger

    assert 'input_size' in PROBE_RESULT_IDENTITY_KEYS
    assert 'feature_selection' in PROBE_RESULT_IDENTITY_KEYS

    logger = MetricsLogger(SimpleNamespace(hidden_size=8192, feature_selection=False, input_size=2560))
    # What the run actually used, after selection narrowed the input and a sweep widened
    # the hidden layer.
    logger.probe_args = SimpleNamespace(hidden_size=1024, feature_selection=True, input_size=256)

    row = logger._with_probe_result_identity({'eval_mcc': 0.5})

    assert row['input_size'] == 256
    assert row['hidden_size'] == 1024
    assert row['feature_selection'] is True


def test_result_identity_falls_back_to_the_command_line_snapshot():
    """Not every caller has probe_args, and a missing key must not drop out of the row."""
    from types import SimpleNamespace

    try:
        from src.protify.logger import MetricsLogger
    except ImportError:
        from protify.logger import MetricsLogger

    logger = MetricsLogger(SimpleNamespace(probe_type='mlp', num_labels=3))

    row = logger._with_probe_result_identity({'eval_mcc': 0.5})

    assert row['probe_type'] == 'mlp'
    assert row['num_labels'] == 3


def test_single_precision_metrics_are_averaged_not_dropped():
    """np.float32 is not a Python float, unlike np.float64, so isinstance misses it.

    A metric that misses the numeric branch keeps only the first run's value, loses its
    seed spread, and then cannot be serialized, which discards the whole results table at
    the final write. That combination only appears with more than one run, which is why it
    survived until a multi-seed campaign.
    """
    import numpy as np

    try:
        from src.protify.probes.trainers import TrainerMixin
    except ImportError:
        from protify.probes.trainers import TrainerMixin

    runs = [
        {'eval_mcc': np.float32(0.40), 'eval_bins': np.array([1, 2]), 'eval_note': 'a'},
        {'eval_mcc': np.float32(0.50), 'eval_bins': np.array([1, 2]), 'eval_note': 'a'},
    ]

    aggregated = TrainerMixin._aggregate_metrics(TrainerMixin, runs)

    assert aggregated['eval_mcc'] == '0.4500±0.0500'
    assert aggregated['eval_mcc_mean'] == pytest.approx(0.45)
    assert isinstance(aggregated['eval_mcc_mean'], float)
    # A non-numeric value keeps the first run's, but as something JSON can encode.
    assert aggregated['eval_bins'] == [1, 2]

    json.dumps(aggregated)


def test_a_boolean_flag_is_not_averaged_across_seeds():
    """Averaging a flag would turn True and True into the string '1.0000±0.0000'."""
    try:
        from src.protify.probes.trainers import TrainerMixin
    except ImportError:
        from protify.probes.trainers import TrainerMixin

    aggregated = TrainerMixin._aggregate_metrics(
        TrainerMixin, [{'feature_selection': True}, {'feature_selection': True}]
    )

    assert aggregated['feature_selection'] is True
