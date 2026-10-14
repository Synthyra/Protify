"""Clean a table of protein-protein interactions, print the splits built from it, check them, and push them.

Atlas's and DatasetDev's `paired_data` modules held the same functions, and both import them
from here. The package does not import this module, so `foundry.datasets` stays free of pandas.
`datasets` is imported when a split is pushed, and not before.
"""

from __future__ import annotations

import contextlib
import numpy as np
import pandas as pd

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict

from .leakage import LeakageReport, check_split_disjointness


if TYPE_CHECKING:
    from datasets import Dataset


__all__ = [
    "align_split_columns",
    "check_pair_split_disjointness",
    "clean_ppi_df",
    "frame_to_dataset",
    "print_ppi_df_info",
    "push_split_config_to_hub",
]

# The columns a row needs a value in to be kept, where the table has them.
REQUIRED_COLUMNS = ("IdA", "IdB", "OrgA", "OrgB")

# Each side of a pair: its protein column and the column holding that protein's cluster.
SIDES = (("IdA", "cluster_a"), ("IdB", "cluster_b"))


def clean_ppi_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop incomplete rows and repeated pairs, keeping each pair's first row, on a fresh index.

    A row is incomplete when it lacks a value in any of `REQUIRED_COLUMNS` the table has; missing
    values elsewhere are kept. A pair counts once whichever way round it is written, so (A, B)
    repeats (B, A). Identifiers are compared as strings, so `1` and `"1"` are the same protein.
    `_IdMin` and `_IdMax` serve as scratch columns, so a table's own columns of those names are
    dropped. `IdA` and `IdB` must be columns, or pandas raises `KeyError`. Each step prints the
    table's size, and the input is left unchanged.
    """
    print(f"Current df size: {len(df)}")
    required = [column for column in REQUIRED_COLUMNS if column in df.columns]
    if required:
        df = df.dropna(subset=required)
        print(f"After dropping nas (required cols only): {len(df)}")

    # n: the rows left once the incomplete ones are dropped.
    id_a = df["IdA"].astype(str)  # (n,)
    id_b = df["IdB"].astype(str)  # (n,)
    in_order = id_a <= id_b  # (n,)
    df = df.assign(
        _IdMin=np.where(in_order, id_a, id_b),  # (n,)
        _IdMax=np.where(in_order, id_b, id_a),  # (n,)
    )
    before = len(df)
    df = df.drop_duplicates(subset=["_IdMin", "_IdMax"]).drop(columns=["_IdMin", "_IdMax"])
    print(f"After dropping undirected duplicates: {before} -> {len(df)}")
    return df.reset_index(drop=True)


def print_ppi_df_info(train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    """Print the size of each split, then its first five rows, in the order train, test, valid."""
    print(f"Train df size: {len(train_df)}")
    print(f"Test df size: {len(test_df)}")
    print(f"Valid df size: {len(valid_df)}")
    print(f"Training df head: {train_df.head()}")
    print(f"Test df head: {test_df.head()}")
    print(f"Valid df head: {valid_df.head()}")


def check_pair_split_disjointness(
    splits: Mapping[str, pd.DataFrame],
    seq_dict: Mapping[object, str],
) -> LeakageReport:
    """What the splits of an interaction table share: proteins, sequences, and clusters.

    A split's proteins are every `IdA` and `IdB` it holds, and `seq_dict` gives each one's
    sequence, so two identifiers with one sequence count as shared. Clusters are checked when every
    split has `cluster_a` and `cluster_b`, the clusters of `IdA` and `IdB`. Identifiers and
    clusters compare as strings, so `1` and `"1"` are one protein.
    """
    clustered = all({"cluster_a", "cluster_b"} <= set(frame.columns) for frame in splits.values())
    fields = ("protein", "sequence", "cluster") if clustered else ("protein", "sequence")
    rows_by_split = {split: _members(frame, seq_dict, clustered) for split, frame in splits.items()}
    return check_split_disjointness(rows_by_split, fields)


class PairMember(TypedDict):
    """One distinct protein of a split, as `check_split_disjointness` reads it."""

    protein: str
    sequence: str
    cluster: NotRequired[str]


def _members(frame: pd.DataFrame, seq_dict: Mapping[object, str], clustered: bool) -> Iterator[PairMember]:
    """Each distinct protein of a split, with its sequence and, when `clustered`, its cluster."""
    columns = ["protein", "cluster"] if clustered else ["protein"]
    # Each side is (r, len(columns)), r the split's rows.
    sides = [
        frame[[protein, cluster] if clustered else [protein]].set_axis(columns, axis=1)
        for protein, cluster in SIDES
    ]
    members = pd.concat(sides, ignore_index=True).drop_duplicates()  # (m, len(columns)), m distinct proteins
    for row in members.itertuples(index=False):
        record: PairMember = {"protein": str(row.protein), "sequence": seq_dict[row.protein]}
        if clustered:
            record["cluster"] = str(row.cluster)
        yield record


def align_split_columns(*frames: pd.DataFrame) -> list[pd.DataFrame]:
    """The frames on one column set, in order of first appearance, each missing column filled with a default.

    A column the first frame to hold it types as an integer is filled with -1, as a float with NaN, and
    otherwise with an empty string. Each frame is copied.
    """
    columns: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        for column in frame.columns:
            if column not in seen:
                columns.append(column)
                seen.add(column)

    aligned: list[pd.DataFrame] = []
    for frame in frames:
        frame = frame.copy()
        for column in columns:
            if column in frame.columns:
                continue
            reference = next((other[column].dtype for other in frames if column in other.columns), None)
            is_int = False
            is_float = False
            if reference is not None:
                with contextlib.suppress(TypeError):
                    is_int = np.issubdtype(reference, np.integer)
                    is_float = np.issubdtype(reference, np.floating)
            if is_int:
                frame[column] = -1
            elif is_float:
                frame[column] = np.nan
            else:
                frame[column] = ""
        aligned.append(frame[columns])
    return aligned


def frame_to_dataset(frame: pd.DataFrame) -> Dataset:
    """A `datasets.Dataset` of the frame, `int8` columns widened to `int32` and object columns made strings."""
    from datasets import Dataset

    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == "int8":
            frame[column] = frame[column].astype("int32")
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str)
    frame = frame.reset_index(drop=True)
    return Dataset.from_pandas(frame, preserve_index=False)


def push_split_config_to_hub(
    *,
    hf_repo: str,
    prefix: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    extra_splits: Mapping[str, pd.DataFrame] | None = None,
) -> None:
    """Upload train, valid, test, and any extra splits as one private config of a dataset repository.

    The frames are aligned on one column set with `align_split_columns`, so the `DatasetDict` has a
    uniform schema, and pushed as the config `prefix`, which the Hub viewer previews inline. The splits
    are named `train`, `valid`, `test`, then `extra_splits` in its order. The repository is private,
    nothing is read back, and no token is checked: the `datasets` library reads the one `huggingface_hub`
    finds. A release whose bytes must be verified goes through `publish_dataset` instead.
    """
    from datasets import DatasetDict

    extra = dict(extra_splits or {})
    names = ["train", "valid", "test", *extra]
    aligned = align_split_columns(train_df, valid_df, test_df, *extra.values())
    splits = DatasetDict({name: frame_to_dataset(frame) for name, frame in zip(names, aligned, strict=True)})
    splits.push_to_hub(hf_repo, config_name=prefix, private=True)
