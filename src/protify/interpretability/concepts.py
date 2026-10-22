"""The concept probing tasks: their splits, label tensors, and the target rules of the sequence controls.

Each task is one config of a Hugging Face dataset with `train`, `validation` and `test` parquet files (`id`,
`sequence`, `label`) and a `metadata.json` naming the task level (protein or residue), the task type and the label
names. Protein tasks are regression (several standardized values), multilabel or binary (sigmoid outputs) or
multiclass (one softmax); residue tasks are multiclass with one class per residue.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from torch import Tensor


DEFAULT_REPO = "Synthyra/protein_concept_probing"
SPLITS = ("train", "validation", "test")
PROTEIN_TASKS = (
    "prot_param", "interpro_binding_site", "interpro_active_site", "interpro_repeat", "interpro_conserved_site",
    "interpro_family", "interpro_domain", "interpro_homologous_superfamily", "go_mf", "go_bp", "go_cc",
    "metal_ion_binding", "localization",
)
RESIDUE_TASKS = (
    "lipidation", "topology", "peptide", "functional_sites", "phosphorylation", "secondary_structure",
    "post_translational_modification", "ssp_q3", "ssp_q8",
)
ALL_TASKS = PROTEIN_TASKS + RESIDUE_TASKS
# Non-canonical residues read as their closest canonical one when ProtParam values are recomputed.
AMBIGUOUS_RESIDUES = str.maketrans({"U": "C", "O": "L", "B": "N", "Z": "Q", "J": "L", "X": "G"})


@dataclass(frozen=True)
class ConceptTask:
    name: str
    level: str  # protein or residue
    kind: str  # regression, multilabel, binary or multiclass
    num_labels: int
    label_names: tuple[str, ...]

    @property
    def loss(self) -> str:
        """The probe loss: softmax cross entropy, per-output sigmoid cross entropy, or squared error."""
        return {"regression": "regression", "multilabel": "multilabel", "binary": "multilabel", "multiclass": "multiclass"}[self.kind]

    @property
    def outputs(self) -> int:
        return 1 if self.kind == "binary" else self.num_labels


@dataclass
class ConceptSplit:
    ids: list[str]
    sequences: list[str]
    labels: list[np.ndarray]  # (C,) or () per protein, or (r,) per residue


@dataclass
class ConceptData:
    task: ConceptTask
    splits: dict[str, ConceptSplit]
    target_mean: np.ndarray | None = None  # (C,) regression standardization, fit on train
    target_std: np.ndarray | None = None  # (C,)

    def protein_targets(self, split: str, rows: Sequence[int] | None = None) -> Tensor:
        """Protein-level targets: (n,) class indices, or (n, C) float values (regression standardized)."""
        labels = self.splits[split].labels
        chosen = labels if rows is None else [labels[row] for row in rows]
        if self.task.kind == "multiclass":
            return torch.tensor(np.asarray(chosen, dtype=np.int64))  # (n,)
        values = np.stack([np.atleast_1d(np.asarray(label, dtype=np.float64)) for label in chosen])  # (n, C)
        if self.task.kind == "regression":
            values = (values - self.target_mean) / self.target_std
        return torch.tensor(values, dtype=torch.float32)  # (n, C)

    def standardize_values(self, values: np.ndarray) -> np.ndarray:
        # values: (n, C) raw regression targets
        return (values - self.target_mean) / self.target_std


def _read_config(repo: str, name: str, revision: str | None, local_dir: Path | None) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    if local_dir is not None:
        folder = local_dir / name
        frames = {split: pd.read_parquet(folder / f"{split}.parquet") for split in SPLITS}
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        return frames, metadata
    from huggingface_hub import hf_hub_download

    def fetch(filename: str) -> str:
        return hf_hub_download(repo, f"{name}/{filename}", repo_type="dataset", revision=revision)

    frames = {split: pd.read_parquet(fetch(f"{split}.parquet")) for split in SPLITS}
    metadata = json.loads(Path(fetch("metadata.json")).read_text(encoding="utf-8"))
    return frames, metadata


def load_concept(
    name: str,
    repo: str = DEFAULT_REPO,
    revision: str | None = None,
    local_dir: str | Path | None = None,
) -> ConceptData:
    """One task's three splits from the Hub (or a local copy laid out the same way)."""
    if name not in ALL_TASKS:
        raise ValueError(f"Unknown concept task {name!r}; known tasks are {list(ALL_TASKS)}.")
    frames, metadata = _read_config(repo, name, revision, Path(local_dir) if local_dir is not None else None)
    task = ConceptTask(
        name=name,
        level=str(metadata["task_level"]),
        kind=str(metadata["task_type"]),
        num_labels=int(metadata["num_labels"]),
        label_names=tuple(metadata["label_names"]),
    )
    splits = {
        split: ConceptSplit(
            ids=frame["id"].astype(str).tolist(),
            sequences=frame["sequence"].tolist(),
            labels=[np.asarray(label) for label in frame["label"]],
        )
        for split, frame in frames.items()
    }
    data = ConceptData(task, splits)
    if task.kind == "regression":
        train = np.stack([np.asarray(label, dtype=np.float64) for label in splits["train"].labels])  # (n, C)
        data.target_mean = train.mean(axis=0)
        data.target_std = train.std(axis=0)  # population, as sklearn's StandardScaler
        data.target_std[data.target_std == 0] = 1.0
    return data


def scramble(sequence: str, seed: int) -> str:
    """Shuffle every residue after the first, keeping the initiator in place."""
    rng = np.random.default_rng(seed)
    tail = list(sequence[1:])
    rng.shuffle(tail)
    return sequence[:1] + "".join(tail)


def scramble_seed(sequence: str, seed: int) -> int:
    """A seed that depends on the sequence and the run seed alone, so a scramble never depends on batch order."""
    import hashlib

    digest = hashlib.sha256(f"{seed}:{sequence}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def prot_param_values(sequences: Sequence[str], label_names: Sequence[str]) -> np.ndarray:
    """The 31 ProtParam targets of each sequence, in `label_names` order, as the scrambled control recomputes them."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    rows = []
    for sequence in sequences:
        cleaned = sequence.translate(AMBIGUOUS_RESIDUES)
        analysis = ProteinAnalysis(cleaned)
        helix, turn, sheet = analysis.secondary_structure_fraction()
        counts = analysis.count_amino_acids()
        values = {
            "length": float(len(sequence)),
            "mass": analysis.molecular_weight(),
            "fraction_helix_aas": helix,
            "fraction_turn_aas": turn,
            "fraction_sheet_aas": sheet,
            "instability_index": analysis.instability_index(),
            "gravy": analysis.gravy(),
            "isoelectric_point": analysis.isoelectric_point(),
            "charge_at_ph4_7": analysis.charge_at_pH(4.7),
            "charge_at_ph7_2": analysis.charge_at_pH(7.2),
            "charge_at_ph8": analysis.charge_at_pH(8.0),
        }
        for residue, count in counts.items():
            values[f"percent_{residue.lower()}"] = 100.0 * count / len(cleaned)
        rows.append([values[name] for name in label_names])
    return np.asarray(rows, dtype=np.float64)  # (n, C)


def shuffled_labels(label: np.ndarray, seed: int) -> np.ndarray:
    """The scrambled control's targets: the same labels in a random order within the protein."""
    rng = np.random.default_rng(seed)
    flat = np.asarray(label).reshape(-1).copy()
    rng.shuffle(flat)
    return flat.reshape(np.asarray(label).shape)
