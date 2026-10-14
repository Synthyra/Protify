import sys
import pytest

from pathlib import Path
from datasets import Dataset, DatasetDict


try:
    from src.protify.data.data_mixin import DataArguments, DataMixin
except ImportError:
    try:
        from protify.data.data_mixin import DataArguments, DataMixin
    except ImportError:
        from ..data.data_mixin import DataArguments, DataMixin


AA_SUFFIXES = "LMNPQRSTVWY"


def _sequence_for_idx(idx: int) -> str:
    return f"ACDEFGHIK{AA_SUFFIXES[idx % len(AA_SUFFIXES)]}"


def _write_csv_split(path: Path, n_rows: int) -> None:
    lines = ["seqs,labels"]
    for idx in range(n_rows):
        lines.append(f"{_sequence_for_idx(idx)},{idx % 2}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fasta_split(path: Path, n_rows: int) -> None:
    records = []
    for idx in range(n_rows):
        records.append(f">seq_{idx} label={idx % 2}")
        records.append(_sequence_for_idx(idx))
    path.write_text("\n".join(records) + "\n", encoding="utf-8")


def test_local_split_alias_synthesizes_missing_test(tmp_path: Path) -> None:
    _write_csv_split(tmp_path / "train.csv", 20)
    _write_csv_split(tmp_path / "validation.csv", 3)

    data_args = DataArguments(data_names=[], data_dirs=[str(tmp_path)])
    datasets_by_name, all_seqs = DataMixin(data_args).get_data()
    train_set, valid_set, test_set, num_labels, label_type, ppi = datasets_by_name[
        tmp_path.name
    ]

    assert len(train_set) + len(test_set) == 20
    assert len(valid_set) == 3
    assert len(test_set) > 0
    assert num_labels == 2
    assert label_type == "singlelabel"
    assert ppi is False
    assert _sequence_for_idx(0) in all_seqs


def test_split_alias_synthesizes_missing_valid() -> None:
    mixin = DataMixin()
    dataset = {
        "train": Dataset.from_dict(
            {
                "seqs": [f"ACD{idx}" for idx in range(20)],
                "labels": [idx % 2 for idx in range(20)],
            }
        ),
        "testing": Dataset.from_dict(
            {
                "seqs": ["ACDT", "ACDV"],
                "labels": [0, 1],
            }
        ),
    }

    train_set, valid_set, test_set = mixin._select_train_valid_test_splits(
        dataset,
        "alias_dataset",
    )

    assert len(train_set) + len(valid_set) == 20
    assert len(valid_set) > 0
    assert len(test_set) == 2


def test_local_fasta_split_files_with_labels(tmp_path: Path) -> None:
    _write_fasta_split(tmp_path / "train.fasta", 12)
    _write_fasta_split(tmp_path / "val.faa", 2)
    _write_fasta_split(tmp_path / "testing.fa", 2)

    data_args = DataArguments(data_names=[], data_dirs=[str(tmp_path)])
    datasets_by_name, all_seqs = DataMixin(data_args).get_data()
    train_set, valid_set, test_set, num_labels, label_type, ppi = datasets_by_name[
        tmp_path.name
    ]

    assert train_set.column_names == ["seqs", "labels"]
    assert len(train_set) == 12
    assert len(valid_set) == 2
    assert len(test_set) == 2
    assert num_labels == 2
    assert label_type == "singlelabel"
    assert ppi is False
    assert _sequence_for_idx(0) in all_seqs


def test_fasta_split_requires_label_metadata(tmp_path: Path) -> None:
    fasta_path = tmp_path / "train.fa"
    fasta_path.write_text(">seq_without_label\nACDEFGHIK\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="label="):
        DataMixin()._read_fasta_file(str(fasta_path))


def test_hub_datasets_load_at_their_pinned_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    def record_load_dataset(path: str, revision: str | None = None) -> DatasetDict:
        requested.append((path, revision))
        split = Dataset.from_dict(
            {
                "seqs": [_sequence_for_idx(idx) for idx in range(4)],
                "labels": [idx % 2 for idx in range(4)],
            }
        )
        return DatasetDict({"train": split, "valid": split, "test": split})

    monkeypatch.setattr(sys.modules[DataMixin.__module__], "load_dataset", record_load_dataset)
    data_args = DataArguments(
        data_names=["research/first", "research/second"],
        data_revisions=["a" * 40, "b" * 40],
    )
    datasets_by_name, _all_seqs = DataMixin(data_args).get_data()

    assert data_args.data_revisions == ["a" * 40, "b" * 40]
    assert requested == [("research/first", "a" * 40), ("research/second", "b" * 40)]
    assert set(datasets_by_name) == {"first", "second"}


def test_unpinned_hub_datasets_load_the_default_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []

    def record_load_dataset(path: str, revision: str | None = None) -> DatasetDict:
        requested.append((path, revision))
        raise LookupError("stop after the load request")

    monkeypatch.setattr(sys.modules[DataMixin.__module__], "load_dataset", record_load_dataset)
    with pytest.raises(LookupError):
        DataMixin(DataArguments(data_names=["research/first"])).get_data()

    assert requested == [("research/first", None)]


def test_revisions_must_align_with_dataset_names() -> None:
    with pytest.raises(ValueError, match="one for one"):
        DataArguments(data_names=["research/first", "research/second"], data_revisions=["a" * 40])


def test_revisions_reject_names_that_are_not_one_dataset() -> None:
    with pytest.raises(ValueError, match="standard_benchmark"):
        DataArguments(data_names=["standard_benchmark"], data_revisions=["a" * 40])
