"""Capping each split at a row count, the same rows for every seed and every consumer.

A capped split has to be a function of its own rows and `--split_row_cap_seed` alone. A run
trains several seeds, and its caches may be filled in one pass over several datasets; if the
kept rows moved with the global seed, or with which other datasets were loaded, the seeds
would train on different data and the caches would miss rows a later run reads.

What has to hold: exact counts, class shares and regression deciles kept, a uniform draw
where a label gives a row no stratum, row order preserved, a split at or under its cap
untouched, filtering before the cap, and dropped rows absent from `all_seqs`, so nothing
embeds a protein no split reads. The datasets are small in-memory stand-ins served through
a patched loader, and every test runs the real `get_data`; nothing downloads.
"""

import builtins
import random
import sys
import numpy as np
import pytest
import torch

from datasets import Dataset, DatasetDict

from protify.data.data_mixin import DataArguments, DataMixin, capped_row_indices, stratified_quotas


# Residues that survive sanitization, read as base-20 digits so a sequence names its row.
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
# The first residue names the split, so no sequence is shared between splits.
SPLIT_PREFIX = {'train': 'M', 'valid': 'K', 'test': 'W'}
PLACES = (2, 1, 0)


def _sequence(split_name: str, row: int) -> str:
    """A sequence that encodes its split and row and comes through sanitization unchanged."""
    digits = ''.join(ALPHABET[(row // 20 ** place) % 20] for place in PLACES)
    return f'{SPLIT_PREFIX[split_name]}{digits}'


def _row(sequence: str) -> int:
    """The row `_sequence` encoded."""
    return sum(ALPHABET.index(digit) * 20 ** place for digit, place in zip(sequence[1:], PLACES, strict=True))


def _split(split_name: str, labels: list) -> Dataset:
    return Dataset.from_dict(
        {'seqs': [_sequence(split_name, row) for row in range(len(labels))], 'labels': labels}
    )


def _toy(train_labels: list, valid_labels: list | None = None, test_labels: list | None = None) -> dict:
    """Three splits, the evaluation splits defaulting to the first 20 training labels."""
    return {
        'train': _split('train', train_labels),
        'valid': _split('valid', valid_labels if valid_labels is not None else train_labels[:20]),
        'test': _split('test', test_labels if test_labels is not None else train_labels[:20]),
    }


def _load(monkeypatch, splits_by_path: dict, **data_arguments) -> tuple:
    """The real `get_data` over in-memory datasets, one fresh DataMixin per call."""
    def load_dataset(path: str, revision: str | None = None) -> DatasetDict:
        return DatasetDict(splits_by_path[path])

    monkeypatch.setattr(sys.modules[DataMixin.__module__], 'load_dataset', load_dataset)
    arguments = DataArguments(data_names=list(splits_by_path), **data_arguments)
    return DataMixin(arguments).get_data()


def _kept_rows(split: Dataset) -> list[int]:
    return [_row(sequence) for sequence in split['seqs']]


def _classes(count: int, classes: int = 3) -> list[int]:
    return [row % classes for row in range(count)]


@pytest.fixture
def set_global_seed(monkeypatch):
    """Protify's own `set_global_seed`, with every generator it reseeds put back afterwards."""
    seed_utils = sys.modules[sys.modules[DataMixin.__module__].get_global_seed.__module__]
    monkeypatch.setattr(seed_utils, '_GLOBAL_SEED', seed_utils._GLOBAL_SEED)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield seed_utils.set_global_seed
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)


def test_each_capped_split_keeps_exactly_its_cap(monkeypatch):
    toy = _toy(_classes(300), _classes(90), _classes(60))

    datasets, _ = _load(monkeypatch, {'research/toy': toy}, max_train_rows=100, max_valid_rows=31, max_test_rows=7)

    train, valid, test = datasets['toy'][:3]
    assert (len(train), len(valid), len(test)) == (100, 31, 7)


def test_every_class_keeps_its_share_within_one_row(monkeypatch):
    # 70%, 20%, and 10% of 1000 rows, interleaved so no class sits in one block of rows.
    labels = [0 if row % 10 < 7 else 1 if row % 10 < 9 else 2 for row in range(1000)]

    datasets, _ = _load(monkeypatch, {'research/toy': _toy(labels)}, max_train_rows=97)

    kept = np.bincount(list(datasets['toy'][0]['labels']), minlength=3)  # (3,)
    exact = 97 * np.bincount(labels) / len(labels)  # (3,) 67.9, 19.4, 9.7
    assert kept.sum() == 97
    assert np.all(np.abs(kept - exact) < 1)
    # Flooring leaves 67, 19, and 9; the two spare rows go to the largest remainders, .9 and .7.
    assert kept.tolist() == [68, 19, 10]


def test_largest_remainder_quotas_sum_to_the_cap_and_stay_within_one_row():
    sizes = np.array([700, 200, 100])  # (3,)

    for cap in range(1, 1000, 37):
        quotas = stratified_quotas(sizes, cap)  # (3,)
        assert quotas.sum() == cap
        assert np.all(np.abs(quotas - cap * sizes / sizes.sum()) < 1)

    # Equal remainders go to the stratum listed first, so the quotas never depend on chance.
    assert stratified_quotas(np.array([1, 1, 1]), 2).tolist() == [1, 1, 0]


def test_a_capped_regression_split_keeps_every_decile(monkeypatch):
    # 200 distinct targets, none integer-valued, in an order unrelated to their rank.
    labels = [(row * 37) % 200 + 0.5 for row in range(200)]

    datasets, _ = _load(monkeypatch, {'research/toy': _toy(labels)}, max_train_rows=20)

    train = datasets['toy'][0]
    assert datasets['toy'][4] == 'regression'
    # Each decile holds 20 of the 200 targets, ranked 0 to 199, so it keeps 2 of the 20 rows.
    kept_deciles = [int(label - 0.5) // 20 for label in train['labels']]
    assert np.bincount(kept_deciles, minlength=10).tolist() == [2] * 10


def test_multilabel_rows_are_drawn_uniformly(monkeypatch):
    """A list label gives a row no single stratum, so the draw ignores labels entirely.

    Two splits of one size with different label lists keep the same rows, which a draw
    stratified by label would not.
    """
    kept = []
    for labels in (
        [[row % 2, row // 2 % 2, 1] for row in range(120)],
        [[1, 0, int(row % 5 == 0)] for row in range(120)],
    ):
        datasets, _ = _load(monkeypatch, {'research/toy': _toy(labels)}, max_train_rows=25)

        train = datasets['toy'][0]
        assert len(train) == 25
        assert datasets['toy'][4] == 'multilabel'
        kept.append(_kept_rows(train))

    assert kept[0] == kept[1]


def test_the_kept_rows_ignore_every_global_seed(monkeypatch, set_global_seed):
    """The training seed changes per run seed; the rows a cell trains on may not."""
    kept = []
    for global_seed in (1, 2):
        set_global_seed(global_seed)  # Python, numpy, and torch
        datasets, _ = _load(
            monkeypatch,
            {'research/toy': _toy(_classes(300), _classes(90), _classes(60))},
            max_train_rows=50,
            max_valid_rows=10,
            max_test_rows=10,
        )
        kept.append([_kept_rows(split) for split in datasets['toy'][:3]])

    assert kept[0] == kept[1]


def test_the_cap_seed_is_what_moves_the_draw(monkeypatch):
    kept = []
    for seed in (0, 1):
        datasets, _ = _load(
            monkeypatch, {'research/toy': _toy(_classes(300))}, max_train_rows=50, split_row_cap_seed=seed
        )
        kept.append(_kept_rows(datasets['toy'][0]))

    assert kept[0] != kept[1]


def test_two_separate_mixins_keep_the_same_rows(monkeypatch):
    kept = []
    for _ in range(2):
        datasets, _ = _load(monkeypatch, {'research/toy': _toy(_classes(300))}, max_train_rows=40)
        kept.append(_kept_rows(datasets['toy'][0]))

    assert kept[0] == kept[1]


def test_the_draw_never_reads_pythons_salted_hash(monkeypatch):
    """`hash(str)` is salted per process, so a draw seeded through it would change on every rerun."""
    def salted_hash(value):
        raise AssertionError(f'hash({value!r}) differs between processes')

    labels_by_kind = {
        'classes': _classes(100),
        'regression': [row + 0.5 for row in range(100)],
        'multilabel': [[row % 2, 1] for row in range(100)],
    }
    expected = {
        kind: capped_row_indices(labels, 10, 0, 'train') for kind, labels in labels_by_kind.items()
    }

    with monkeypatch.context() as patched:
        patched.setattr(builtins, 'hash', salted_hash)
        drawn = {
            kind: capped_row_indices(labels, 10, 0, 'train') for kind, labels in labels_by_kind.items()
        }

    for kind in labels_by_kind:
        np.testing.assert_array_equal(drawn[kind], expected[kind])


def test_a_dataset_keeps_its_rows_when_loaded_beside_others(monkeypatch):
    """The precompute loads every dataset at once and a cell loads one; both must keep the same rows."""
    toy = _toy(_classes(300))
    other = _toy(_classes(500, classes=4))

    alone, _ = _load(monkeypatch, {'research/toy': toy}, max_train_rows=50)
    together, _ = _load(monkeypatch, {'research/other': other, 'research/toy': toy}, max_train_rows=50)

    assert _kept_rows(alone['toy'][0]) == _kept_rows(together['toy'][0])


def test_kept_rows_keep_their_order(monkeypatch):
    datasets, _ = _load(
        monkeypatch,
        {'research/toy': _toy(_classes(300), _classes(90), _classes(60))},
        max_train_rows=50,
        max_valid_rows=10,
        max_test_rows=10,
    )

    for split in datasets['toy'][:3]:
        rows = _kept_rows(split)
        assert rows == sorted(set(rows))


def test_a_split_at_or_under_its_cap_is_untouched(monkeypatch, capsys):
    """An uncapped dataset keeps identical rows in identical order, so its row digests do not move."""
    toy = _toy(_classes(40), _classes(20), _classes(10))

    uncapped, uncapped_sequences = _load(monkeypatch, {'research/toy': toy})
    capped, capped_sequences = _load(
        monkeypatch, {'research/toy': toy}, max_train_rows=40, max_valid_rows=21, max_test_rows=1000
    )

    for before, after in zip(uncapped['toy'][:3], capped['toy'][:3], strict=True):
        assert list(after['seqs']) == list(before['seqs'])
        assert list(after['labels']) == list(before['labels'])
    assert capped_sequences == uncapped_sequences
    assert ': capped ' not in capsys.readouterr().out


def test_the_cap_counts_only_rows_that_survive_filtering(monkeypatch):
    """A row with no label, or with no residue left after sanitization, never takes a kept slot."""
    sequences = [_sequence('train', row) for row in range(100)]
    labels = [row % 2 for row in range(100)]
    for row in range(0, 100, 10):
        labels[row] = None  # dropped by the missing-value filter
    for row in range(5, 100, 10):
        sequences[row] = '1234'  # sanitizes to nothing, so the empty-sequence filter drops it
    dropped = set(range(0, 100, 10)) | set(range(5, 100, 10))
    toy = {
        'train': Dataset.from_dict({'seqs': sequences, 'labels': labels}),
        'valid': _split('valid', _classes(10, classes=2)),
        'test': _split('test', _classes(10, classes=2)),
    }

    # 80 rows survive, so a cap of 80 keeps all of them.
    at_cap, _ = _load(monkeypatch, {'research/toy': toy}, max_train_rows=80)
    assert sorted(_kept_rows(at_cap['toy'][0])) == sorted(set(range(100)) - dropped)

    under_cap, _ = _load(monkeypatch, {'research/toy': toy}, max_train_rows=60)
    kept = _kept_rows(under_cap['toy'][0])
    assert len(kept) == 60
    assert not set(kept) & dropped


def test_dropped_rows_never_reach_all_seqs(monkeypatch):
    """The cap runs before all_seqs is collected, so no protein is embedded for a row no split keeps."""
    toy = _toy(_classes(300), _classes(90), _classes(60))

    datasets, all_seqs = _load(monkeypatch, {'research/toy': toy}, max_train_rows=30, max_valid_rows=5, max_test_rows=5)

    kept = {sequence for split in datasets['toy'][:3] for sequence in split['seqs']}
    assert len(kept) == 40
    assert set(all_seqs) == kept


def test_each_capped_split_prints_one_line(monkeypatch, capsys):
    toy = _toy(_classes(300), _classes(90), _classes(60))

    _load(monkeypatch, {'research/toy': toy}, max_train_rows=100, max_test_rows=7, split_row_cap_seed=3)

    lines = [line for line in capsys.readouterr().out.splitlines() if ': capped ' in line]
    assert lines == [
        'toy train: capped 300 rows to 100 (split_row_cap_seed 3)',
        'toy test: capped 60 rows to 7 (split_row_cap_seed 3)',
    ]


@pytest.mark.parametrize(
    'name, value',
    [('max_train_rows', 0), ('max_valid_rows', -5), ('max_test_rows', 2.5), ('max_train_rows', True)],
)
def test_a_cap_that_is_not_a_positive_row_count_is_rejected(name, value):
    with pytest.raises(ValueError, match=f'{name} must be a positive number of rows'):
        DataArguments(data_names=[], **{name: value})


def test_a_negative_cap_seed_is_rejected():
    with pytest.raises(ValueError, match='split_row_cap_seed must be a non-negative integer'):
        DataArguments(data_names=[], split_row_cap_seed=-1)


def test_main_parses_the_caps_and_hands_them_to_data_arguments(monkeypatch):
    """MainProcess builds DataArguments from the parsed namespace, so a parsed cap reaches the loader."""
    from protify.main import parse_arguments

    monkeypatch.setattr(sys.modules[parse_arguments.__module__], 'load_secrets_env', lambda **_: [])
    monkeypatch.setattr(sys, 'argv', ['main.py'])
    defaults = parse_arguments()
    assert (defaults.max_train_rows, defaults.max_valid_rows, defaults.max_test_rows) == (None, None, None)
    assert defaults.split_row_cap_seed == 0

    monkeypatch.setattr(sys, 'argv', [
        'main.py', '--max_train_rows', '5000', '--max_valid_rows', '600',
        '--max_test_rows', '700', '--split_row_cap_seed', '4',
    ])
    data_arguments = DataArguments(**vars(parse_arguments()))  # as apply_current_settings builds it
    assert data_arguments.split_row_caps == {'train': 5000, 'valid': 600, 'test': 700}
    assert data_arguments.split_row_cap_seed == 4
