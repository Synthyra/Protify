"""Render and check the Hub card of a dataset release, so every dataset's README reads the same way.

The card is what a person sees first on the Hub, so it is drawn from the release's own verified files, never typed: the
numbers come from `dataset.json` and `acceptance.json`, the homology claim from the acceptance report's split policy and
its twelve directed searches, and the columns from the split file's schema. A release built by an applied project, with no
acceptance report, carries `card_facts.json` instead: the same numbers, written by its exporter from the build's own receipts
(`manifest_homology` derives the homology part from a split record), and every rule below holds for it too. Only the summary and the statement of what a
model trained on the data may say are authored, in the dataset's workspace README under the headings of
`tools/templates/dataset.md`. The layout is `tools/templates/hub_dataset_card.md` and the rules are
[Hub datasets](../../../../../docs/conventions/hub_datasets.md).

    python -m foundry.datasets.hub_card render --release <release dir> --readme datasets/<id>/README.md \\
        --map datasets/<id> --template tools/templates/hub_dataset_card.md --output <card dir>
    python -m foundry.datasets.hub_card check --release <release dir> --card <card dir>
    python -m foundry.datasets.hub_card check-readme --readme datasets/<id>/README.md

`render` writes `README.md` into the card directory; `ws dataset map stage` then adds the map files and `hub_release
assemble` builds the release folder. `check` is what `hub_release publish` runs: it refuses a card that holds a placeholder,
a hash, a workspace path, a column it does not describe, or a homology statement that differs from the acceptance report.
"""

from __future__ import annotations

import argparse
import json
import re
import pyarrow.parquet as pq

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict


SPLITS = ("train", "validation", "test", "calibration")
HEADINGS = (
    "Homology leakage", "How it was built", "Columns", "Example rows",
    "What a model trained on it may say", "Sources and licence", "Files",
)
MAY_SAY = "What a model trained on it may say"
BUILT = "How it was built"
SUMMARY_WORDS = 120
MIN_STEPS = 3
EXAMPLE_ROWS = 3
PROTEIN_ID_SHOWN = 12
MAX_LINES = 200
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
MANIFEST_KEYS = ("builder:", "descends_from:", "record_sha256:", "qualification_sha256:")
LICENSES = {"cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0", "cc-by-nc-4.0", "apache-2.0", "mit", "odc-by"}
EM_DASH = "\u2014"

COLUMNS = {
    "name": "A readable name for the row: the first alias the source gives, for example `UniProtKB:P77619`, else the source record id.",
    "seqs": "The protein sequence the model reads.",
    "labels": "The target. For a multilabel task a sparse vector: `width` labels, the `indices` marked and their `values`.",
    "label_mask": "Which labels were observed. A label outside the mask is unknown, never absent.",
    "cluster_id": "The protein's finest cluster (0.9 identity). Every member of a cluster sits in one split.",
    "protein_id": "SHA-256 of the sequence. Protify's pinned format requires it; it is an id, not something to read.",
    "observation_id": "SHA-256 of the observation. An id for joining to the records, not something to read.",
    "aliases": "Other names the source gives the protein.",
    "source_groups": "What the split keeps together, such as the entries of one gene product.",
    "source_json": "Where the row came from: source file digest, record and measurement ids. For agents and audit.",
    "evidence_json": "Every assertion behind the row and why each was admitted or set aside. For agents and audit.",
    "observation_json": "The pair, its evidence and its source values. For agents and audit.",
    "partners_json": "The experiments that show the pair, and the partner's canonical identity where it is not a protein of the pair (a ligand's SMILES).",
    # The readable face of a protein-ligand pair (the `pli_v2` row layout of the production suite's paired export).
    "smiles": "The ligand as the canonical isomeric Kekule SMILES of its standardized molecule (RDKit).",
    "selfies": "The SELFIES of that SMILES, the string the ligand tower reads.",
    "ligand_storable": "False when the SELFIES-TED-LARGE store sets the ligand aside (over 1,022 symbols, or a tokenization that is not its symbols); the row stays.",
    "label": "The binary call: 1 when a structure or a measurement at or under the positive threshold states the pair, empty when the measurements decide neither.",
    "p_affinity": "The negative log10 molar affinity, the median of the exact measurements (Kd and Ki first, else IC50); empty when none is exact.",
    "affinity_type": "Which measurements `p_affinity` is the median of: `Kd`, `Ki`, `Kd+Ki` or `IC50`.",
    "site_residues": "The binding-site residues as 1-based positions in the sequence; empty for an affinity-only pair.",
    "site_interactions": "For each site residue with a PLIP interaction, the residue and its kinds: `hydrogen_bond`, `hydrophobic_contact`, `water_bridge`, `salt_bridge`, `pi_stacking`, `pi_cation`, `metal_complex`, `halogen_bond`.",
    "site_tier": "Where the site comes from: `experimental` (a solved structure) or `predicted` (a co-folded or template-modeled complex); empty with no site.",
    "sources": "The sources that state the pair, the naming source first.",
    "source_licenses": "The licence of each source, in the order of `sources`.",
    "evidence_tier": "What shows the pair: `structure`, `affinity`, or `structure_and_affinity`.",
    "pdb_entries": "The PDB entries whose structures show the pair.",
    "uniprot": "The UniProt accessions of the protein.",
    "seq_sha256": "SHA-256 of the sequence, the key of the canonical embedding stores. An id, not something to read.",
    "smiles_sha256": "SHA-256 of the canonical SMILES, the key of the ligand embedding store. An id, not something to read.",
}
READ_FIRST = ("name", "seqs", "labels", "label_mask")
# The example columns of a paired release whose split files carry readable columns beside the audit JSON, in this order where present.
PAIRED_EXAMPLE_COLUMNS = ("seqs", "smiles", "label", "p_affinity", "affinity_type", "site_tier", "sources")
REDUNDANT = {
    "split": "constant in each split file, so the file already says it",
    "length": "derivable from the sequence",
    "alphabet_flags": "derivable from the sequence",
}


class CardError(ValueError):
    """The release or README cannot yield a card; nothing is written."""


@dataclass(frozen=True, slots=True)
class Homology:
    """What the split guarantees, read from the acceptance report and never restated by hand."""

    enforced: float
    levels: tuple[float, ...]
    coverage: float
    searches: int
    largest: float | None
    rationale: str
    far: int | None = None
    held_out: int | None = None
    unit: str = "sequence"

    @property
    def measured(self) -> float | None:
        """The finest level, reported and not enforced, when the split enforces a coarser one."""
        return min(self.levels) if self.enforced > min(self.levels) and self.far is not None else None

    def claim(self) -> str:
        """The one line a reader needs: which identity is enforced and what the searches found."""
        found = f"found no pair above {self.largest:g} identity" if self.largest is not None else f"found no pair at {self.enforced:g} identity or more"
        if self.unit == "pair":
            found = f"found a highest pair identity of {self.largest:g}, below the {self.enforced:g} line" if self.largest is not None else found
            return (
                f"**Enforced: {self.enforced:g} identity on pairs.** The split unit is a pair of chain clusters at {self.enforced:g} identity, so a held-out "
                f"pair has no train pair whose two chains are both at {self.enforced:g} identity or more to its chains. **A chain can recur across "
                f"splits with a different partner**, so read the novelty of each held-out pair before scoring. {self.searches} directed searches "
                f"between the splits {found}."
            )
        text = (
            f"**Enforced: {self.enforced:g} identity.** No held-out sequence has a relative at {self.enforced:g} identity or more in "
            f"another split: every cluster at that level, and so every finer one, sits in exactly one split. {self.searches} directed "
            f"searches between the splits {found}."
        )
        if self.measured is not None:
            text += (
                f" **Measured, not enforced: {self.measured:g} identity.** {self.far:,} of the {self.held_out:,} held-out proteins "
                f"({100 * self.far / self.held_out:.0f}%) have no relative at {self.measured:g} identity or more in train; the rest "
                f"have one between {self.measured:g} and {self.enforced:g}."
            )
        return text

    def table(self) -> str:
        lines = ["| Identity | Guarantee |", "|---|---|"]
        for level in self.levels:
            if self.unit == "pair":
                guarantee = "**enforced** on pairs: no held-out pair has both chains at this identity to a train pair" if level == self.enforced else "not enforced: a chain may share this level across splits"
            elif level > self.enforced:
                guarantee = "implied: no cluster at this level spans two splits"
            elif level == self.enforced:
                guarantee = "**enforced**: no cluster at this level spans two splits"
            elif level == self.measured:
                guarantee = f"measured, not enforced: {self.far:,} of {self.held_out:,} held-out proteins are below this identity to train"
            else:
                guarantee = "not enforced: a held-out sequence may share this level with a training sequence"
            lines.append(f"| {level:g} | {guarantee} |")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class Facts:
    dataset_id: str
    task_type: str
    endpoint: str
    units: str | None
    labels: tuple[str, ...]
    transform: str | None
    splits: Mapping[str, Mapping[str, int | None]]
    homology: Homology
    criteria: tuple[int, int] | None
    target: str = "labels"
    configs: tuple[tuple[str, str], ...] = (("default", ""),)
    task: str | None = None
    acceptance: str | None = None
    rows_note: str | None = None
    unit_label: str = "Proteins"
    cluster_label: str = "Clusters"
    example_columns: tuple[str, ...] = ()
    column_notes: Mapping[str, str] = field(default_factory=dict)
    file_notes: Mapping[str, str] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return sum(counts["rows"] for counts in self.splits.values())


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CardError(f"{path.name} is missing from the release {path.parent}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CardError(f"{path.name} is not a JSON object")
    return value


def _far_counts(splits: Mapping[str, Any]) -> tuple[int, int]:
    """Held-out proteins below the finest level to train, and all held-out proteins, from a split record's strata."""
    far = held_out = 0
    for name in SPLITS[1:]:
        for stratum, counts in splits[name]["strata"].items():
            proteins = counts["held_out"]["proteins"]
            held_out += proteins
            far += proteins if stratum == "far" else 0
    if held_out < 1:
        raise CardError("the split record holds no held-out proteins to measure")
    return far, held_out


def _far_share(release: Path) -> tuple[int, int]:
    return _far_counts(_json(release / "canonical_split.json")["split"]["splits"])


class HomologyFacts(TypedDict):
    """The homology part of `card_facts.json`, as `manifest_homology` derives it from a split record."""

    unit: str
    enforced: float
    levels: list[float]
    coverage: float
    audit_threshold: float
    searches: int
    largest: float
    near_duplicates: int
    # Held-out proteins below the finest level to train, and all held-out proteins; None unless a sequence split enforces
    # a coarser level than its finest.
    far: int | None
    held_out: int | None


def manifest_homology(manifest: Mapping[str, Any], *, unit: str = "sequence", coverage: float | None = None) -> HomologyFacts:
    """The homology part of `card_facts.json`, read from the identity-stratified split record an applied build keeps.

    The record holds the clustering levels and their coverage, the enforced identity (`cluster_identity`), the directed
    searches between the splits (`homology_audit`) and, for a split that enforces a coarser level than its finest, the
    held-out proteins by measured band. An exporter calls this, so the card states what the build measured and nothing it typed.
    `coverage` is read from the record's first clustering unless the exporter gives it, as one whose record keeps no clustering does.
    """
    audit = manifest["homology_audit"]
    comparisons = audit["comparisons"]
    levels = [float(level) for level in manifest["levels"]]
    enforced = float(manifest["cluster_identity"])
    far = held_out = None
    if unit == "sequence" and enforced > min(levels):
        far, held_out = _far_counts(manifest["splits"])
    return {
        "unit": unit, "enforced": enforced, "levels": levels, "coverage": float(coverage if coverage is not None else manifest["clustering"][0]["parameters"]["min_coverage"]),
        "audit_threshold": float(audit["identity_threshold"]), "searches": len(comparisons),
        "largest": max(float(item["maximum_identity"]) for item in comparisons),
        "near_duplicates": sum(1 for item in comparisons if item["near_duplicates"]), "far": far, "held_out": held_out,
    }


def _declared_facts(card_facts: Mapping[str, Any]) -> Facts:
    """Facts from `card_facts.json`, refused on the same grounds as an acceptance report: a failed or mismatched audit."""
    try:
        homology = card_facts["homology"]
        enforced = float(homology["enforced"])
        if float(homology["audit_threshold"]) != enforced:
            raise CardError("the homology audit searched at a different identity than the split enforces")
        if homology["near_duplicates"] or int(homology["searches"]) < 1:
            raise CardError("the homology audit is empty or found near duplicates between splits")
        largest = None if homology["largest"] is None else float(homology["largest"])
        if largest is not None and largest >= enforced:
            raise CardError(f"a cross-split pair reached identity {largest}, at or above the enforced {enforced}")
        declared = card_facts["splits"]
        splits = {name: {"rows": int(declared[name]["rows"]), "proteins": declared[name].get("proteins"),
                         "clusters": declared[name].get("clusters")} for name in SPLITS}
        criteria = card_facts.get("criteria")
        return Facts(
            dataset_id=card_facts["dataset_id"], task_type=card_facts["task_type"], endpoint=card_facts["endpoint"],
            units=card_facts.get("units"), labels=tuple(card_facts.get("labels", ())), transform=card_facts.get("transform"),
            splits=splits,
            homology=Homology(enforced, tuple(float(level) for level in homology["levels"]), float(homology["coverage"]), int(homology["searches"]),
                              largest, str(homology.get("rationale", "")), homology.get("far"), homology.get("held_out"), homology.get("unit", "sequence")),
            criteria=None if criteria is None else (int(criteria[0]), int(criteria[1])), target=card_facts.get("target", "labels"),
            configs=tuple((item["name"], item.get("directory", "")) for item in card_facts.get("configs", [{"name": "default"}])),
            task=card_facts.get("task"), acceptance=card_facts.get("acceptance"), rows_note=card_facts.get("rows_note"),
            unit_label=card_facts.get("unit_label", "Proteins"), cluster_label=card_facts.get("cluster_label", "Clusters"),
            example_columns=tuple(card_facts.get("example_columns", ())), column_notes=dict(card_facts.get("column_notes", {})),
            file_notes=dict(card_facts.get("file_notes", {})),
        )
    except KeyError as error:
        raise CardError(f"card_facts.json lacks {error}") from error


def load_facts(release: Path) -> Facts:
    """The numbers of a release, from its manifest and its passing acceptance report, or from `card_facts.json`.

    A report that failed yields no card.
    """
    if (release / "card_facts.json").is_file():
        return _declared_facts(_json(release / "card_facts.json"))
    manifest, acceptance = _json(release / "dataset.json"), _json(release / "acceptance.json")
    if acceptance.get("criteria_passed") is not True:
        raise CardError("the acceptance report did not pass; no card is rendered for a release that failed its criteria")
    policy, audit = acceptance["split_policy"], acceptance["homology_audit"]
    enforced = float(policy["enforced_identity"])
    if float(audit["identity_threshold"]) != enforced:
        raise CardError("the homology audit searched at a different identity than the split enforces")
    comparisons = audit["comparisons"]
    if not comparisons or any(item["near_duplicates"] for item in comparisons):
        raise CardError("the homology audit is empty or found near duplicates between splits")
    largest = max(float(item["maximum_identity"]) for item in comparisons)
    if largest >= enforced:
        raise CardError(f"a cross-split pair reached identity {largest}, at or above the enforced {enforced}")
    counts = acceptance["split_counts"]
    splits = {name: {"rows": manifest["splits"][name]["rows"], "proteins": counts[name].get("proteins"),
                     "clusters": counts[name].get("clusters")} for name in SPLITS}
    far, held_out = _far_share(release) if enforced > min(float(level) for level in policy["levels"]) else (None, None)
    findings = acceptance["findings"]
    transform = manifest.get("target_transform", {}).get("name")
    return Facts(
        dataset_id=manifest["dataset_id"], task_type=manifest["task_type"], endpoint=manifest["endpoint"], units=manifest.get("units"),
        labels=tuple(manifest.get("label_vocabulary", ())), transform=transform, splits=splits,
        homology=Homology(enforced, tuple(float(level) for level in policy["levels"]), float(policy["cluster_coverage"]),
                          len(comparisons), largest, str(policy.get("rationale", "")), far, held_out),
        criteria=(sum(1 for item in findings if item["passed"]), len(findings)),
    )


def _sections(readme: str) -> dict[str, str]:
    """The README's `## ` sections by heading, each without its heading line."""
    parts = re.split(r"^## +(.+?)\s*$", readme, flags=re.MULTILINE)
    return {parts[index].strip(): parts[index + 1].strip() for index in range(1, len(parts) - 1, 2)}


def _title(readme: str) -> str:
    found = re.search(r"^# +(.+?)\s*$", readme, flags=re.MULTILINE)
    if found is None:
        raise CardError("the README has no title")
    return found.group(1)


def _summary(readme: str, sections: Mapping[str, str]) -> str:
    """The `## Summary` section, else the first paragraph under the title that is not an image, table or heading."""
    if "Summary" in sections:
        return " ".join(sections["Summary"].split())
    body = re.sub(r"\A---\n.*?\n---\n", "", readme, count=1, flags=re.DOTALL)
    body = re.sub(r"^# .*\n", "", body, count=1, flags=re.MULTILINE)
    for paragraph in re.split(r"\n\s*\n", body):
        text = paragraph.strip()
        if text and not text.startswith(("#", "!", "|", "<")):
            return " ".join(text.split())
    raise CardError("the README has no Summary section and no lead paragraph")


def _table(text: str, heading: str) -> list[list[str]]:
    """The body rows of the Markdown table under `## heading` of a generated pipeline.md."""
    block = re.search(rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)", text, flags=re.MULTILINE | re.DOTALL)
    if block is None:
        raise CardError(f"pipeline.md has no '{heading}' section")
    rows = [line for line in block.group(1).splitlines() if line.startswith("|")]
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in rows[2:]]


def _build_steps(sections: Mapping[str, str]) -> str:
    """The numbered steps of the README's `## How it was built`, each joined onto one line.

    The steps are the dataset author's account in order: what was taken from where, what was kept or removed and how many
    rows were left, and the setting of each step that has one. They are copied to the card as written, so a step must not
    link a workspace path, which the Hub cannot resolve.
    """
    text = sections.get(BUILT)
    if text is None:
        raise CardError(f"the README has no '## {BUILT}' section; add it before the card is rendered")
    items = re.findall(r"^(\d+)\. (.+(?:\n[ ]{2,}\S.*)*)", text, flags=re.MULTILINE)
    if len(items) < MIN_STEPS or [int(number) for number, _ in items] != list(range(1, len(items) + 1)):
        raise CardError(f"'## {BUILT}' needs a numbered list 1, 2, 3, ... of at least {MIN_STEPS} steps, one for each thing done in order")
    steps = [f"{number}. {' '.join(body.split())}" for number, body in items]
    if any("](" in step and ".." in step for step in steps):
        raise CardError("a build step links a workspace path; the card is read outside the workspace")
    return "\n".join(steps)


def _without_workspace_links(text: str) -> str:
    """A link that climbs out of the dataset's folder keeps its words and loses its target, which the Hub cannot resolve."""
    return re.sub(r"\[([^\]]+)\]\(\.\./[^)]*\)", r"\1", text)


def _prose(summary: str) -> str:
    """The summary, which must be prose: no list, table or heading."""
    if re.search(r"^\s*(?:[-*|#]|\d+\.) ", summary, flags=re.MULTILINE):
        raise CardError("the summary is a list or table; write it as a short paragraph of plain prose")
    return summary


def _steps(pipeline: str) -> str:
    """The pipeline as one short table: each step with the rows it takes in and leaves."""
    lines = ["| Step | Rows in | Rows out |", "|---|---|---|"]
    for row in _table(pipeline, "Steps"):
        if len(row) != 8:
            raise CardError("pipeline.md has a Steps row that does not have its eight cells; render the map again")
        _number, _stage, step, kind, _line, before, after, detail = row
        out = after or ("see splits" if kind == "split" else "")
        note = f" ({detail})" if kind == "filter" and detail else ""
        lines.append(f"| {step}{note} | {before} | {out} |")
    return "\n".join(lines)


def _sources(pipeline: str) -> tuple[str, str]:
    """The admitted sources as a list, and the licence id when every admitted source names the same one."""
    rows = _table(pipeline, "Sources")
    admitted = [row for row in rows if row[2].startswith(("rows", "reference"))]
    others = len(rows) - len(admitted)
    lines = [f"- {row[1]}: {row[3] or 'terms not recorded'}" for row in admitted]
    if others:
        lines.append(f"- {others} more {'source was' if others == 1 else 'sources were'} considered and set aside; each is listed with its reason in [pipeline.md](pipeline.md).")
    ids = {row[3].split(",")[0].strip().lower() for row in admitted}
    licence = next(iter(ids)) if len(ids) == 1 and next(iter(ids)) in LICENSES else "other"
    return "\n".join(lines), licence


def _size_category(rows: int) -> str:
    for ceiling, name in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"), (1_000_000, "100K<n<1M")):
        if rows < ceiling:
            return name
    return "1M<n<10M"


def _cut(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _split_path(release: Path, directory: str, name: str) -> Path:
    return release / directory / f"{name}.parquet"


def _columns_of(release: Path, directory: str) -> list[str]:
    return pq.ParquetFile(_split_path(release, directory, "train")).schema_arrow.names


def _cell(column: str, value: Any) -> str:
    """One example cell: a number in short form, a sequence in code font and cut, an empty value empty."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    text = _cut(str(value), 28)
    return f"`{text}`" if isinstance(value, str) and (column == "seqs" or column.startswith("seq_")) else text


def _example_cell(column: str, value: Any, facts: Facts) -> str:
    """A multiclass target shows its class name, which the index alone does not tell; every other cell is as `_cell` shows it."""
    return _label_text(value, facts) if column == facts.target and facts.task_type == "multiclass" else _cell(column, value)


def _example_rows(release: Path, facts: Facts) -> str:
    """A few rows of the test split as a table a person can scan, sequences and ids shortened."""
    file = pq.ParquetFile(_split_path(release, facts.configs[0][1], "test"))
    names = file.schema_arrow.names
    if facts.example_columns:
        missing = [column for column in facts.example_columns if column not in names]
        if missing:
            raise CardError(f"the example columns {missing} are not in the test split")
        rows = next(file.iter_batches(batch_size=EXAMPLE_ROWS, columns=list(facts.example_columns))).to_pylist()
        lines = [f"| {' | '.join(facts.example_columns)} |", f"|{'---|' * len(facts.example_columns)}"]
        lines += [f"| {' | '.join(_example_cell(column, row[column], facts) for column in facts.example_columns)} |" for row in rows]
        return "\n".join(lines)
    readable = [name for name in PAIRED_EXAMPLE_COLUMNS if name in names]
    if "labels" not in names and "seqs" in names and "observation_json" in names:
        # A paired release with a readable face (the `pli_v2` layout): its own columns say the pair better than the audit JSON.
        rows = next(file.iter_batches(batch_size=EXAMPLE_ROWS, columns=readable)).to_pylist()
        lines = [f"| {' | '.join(readable)} |", f"|{'---|' * len(readable)}"]
        lines += [f"| {' | '.join(_cell(column, row[column]) for column in readable)} |" for row in rows]
        return "\n".join(lines)
    wanted = [name for name in (*READ_FIRST, "aliases", "observation_json", "partners_json") if name in names]
    rows = next(file.iter_batches(batch_size=EXAMPLE_ROWS, columns=wanted)).to_pylist()
    if "seqs" in names:
        heading = ["Record", "Sequence", "Length", "Labels"]
        lines = [f"| {' | '.join(heading)} |", f"|{'---|' * len(heading)}"]
        for row in rows:
            record = row.get("name") or (row.get("aliases") or ["not recorded"])[0]
            lines.append(f"| {record} | `{_cut(row['seqs'], 28)}` | {len(row['seqs'])} | {_label_text(row['labels'], facts)} |")
        return "\n".join(lines)
    return _paired_example_rows([(json.loads(row["observation_json"]), json.loads(row["partners_json"])) for row in rows])


def _paired_example_rows(records: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """Example rows of a paired release: the partner a ligand pair names, or the two proteins of a protein pair.

    A protein pair names no partner (`ligand_smiles` is null and there is no `partner`), so a Partner column would be blank. Its
    two proteins are the observation's `proteins`, each a SHA-256 of the sequence, shown by its first `PROTEIN_ID_SHOWN` hex digits
    (the card holds no full digest) with the sequence itself in `canonical_split.json`.
    """
    def experiments(partner: dict[str, Any]) -> str:
        return ", ".join(partner.get("source_experiments", []))

    if any(partner.get("ligand_smiles") or partner.get("partner") for _, partner in records):
        lines = ["| Record | Partner | Experiments |", "|---|---|---|"]
        for observation, partner in records:
            smiles = partner.get("ligand_smiles") or partner.get("partner") or ""
            lines.append(f"| {observation['source']['record_id']} | `{_cut(str(smiles), 40)}` | {experiments(partner)} |")
        return "\n".join(lines)
    if not all({"left", "right"} <= set(observation.get("proteins", {})) for observation, _ in records):
        lines = ["| Record | Experiments |", "|---|---|"]
        return "\n".join(lines + [f"| {observation['source']['record_id']} | {experiments(partner)} |" for observation, partner in records])
    lines = ["| Record | Left protein | Right protein | Experiments |", "|---|---|---|---|"]
    for observation, partner in records:
        left, right = (observation["proteins"][side][:PROTEIN_ID_SHOWN] for side in ("left", "right"))
        lines.append(f"| {observation['source']['record_id']} | `{left}` | `{right}` | {experiments(partner)} |")
    return "\n".join(lines)


def _label_text(labels: Any, facts: Facts) -> str:
    if isinstance(labels, dict):
        names = [facts.labels[index] for index in labels["indices"]]
        return ", ".join(names[:4]) + (f" and {len(names) - 4} more" if len(names) > 4 else "")
    if facts.task_type == "multiclass":
        return facts.labels[int(labels)]
    return f"{labels:g}" + (f" {facts.units}" if facts.units else "")


def _described(facts: Facts, config: str, name: str) -> str | None:
    """A column's description: the facts' note for it in this config, else their note, else the shared one."""
    return facts.column_notes.get(f"{config}:{name}") or facts.column_notes.get(name) or COLUMNS.get(name)


def _new_columns(release: Path, facts: Facts) -> list[tuple[str, list[str], list[str]]]:
    """Each config's columns in file order, and the ones its table lists.

    A later config's table lists a column only when no earlier config has it or the facts describe it differently there
    (`observations:pkd`), and says in a line which columns it shares, so three configs do not repeat one table three times.
    """
    seen: set[str] = set()
    listed: list[tuple[str, list[str], list[str]]] = []
    for config, directory in facts.configs:
        names = _columns_of(release, directory)
        new = [name for name in names if name not in seen or f"{config}:{name}" in facts.column_notes]
        listed.append((config, names, new))
        seen |= set(names)
    return listed


def _columns_table(release: Path, facts: Facts) -> str:
    """One table of every column in file order, or one under a heading for each config when there are several."""
    tables: list[str] = []
    for config, names, new in _new_columns(release, facts):
        redundant = {name: REDUNDANT[name] for name in names if name in REDUNDANT}
        if redundant:
            raise CardError(f"the split files hold columns a reader does not need: {redundant}; drop them")
        unknown = [name for name in names if _described(facts, config, name) is None]
        if unknown:
            raise CardError(f"the split files hold columns the card cannot describe: {unknown}; add each to hub_card.COLUMNS or the facts' column_notes")
        lines = ["| Column | What it holds |", "|---|---|"]
        lines += [f"| `{name}` | {_described(facts, config, name)} |" for name in new]
        if len(facts.configs) == 1:
            tables.append("\n".join(lines))
            continue
        shared = [name for name in names if name not in new]
        note = f"Also holds, with the same meaning as above: {', '.join(f'`{name}`' for name in shared)}." if shared else ""
        tables.append(f"### {config}\n\n" + "\n\n".join(part for part in (note, "\n".join(lines) if new else "") if part))
    return "\n\n".join(tables)


def _files_table(release: Path, facts: Facts) -> str:
    notes = {
        "dataset.json": "manifest; pins each split file by digest",
        "export_receipt.json": "readback of the export against its sources",
        "acceptance.json": "the acceptance criteria and whether each passed",
        "canonical_split.json": "the split and its twelve homology searches",
        "evidence_admission.json": "every assertion admitted or set aside, and why",
        "source_qualification.json": "the sources, pinned by digest and revision",
        "label_vocabulary.json": "the label list, fitted on the train split alone",
        "build_receipt.json": "the build that made the release",
        "shortcut_audit.json": "what length, taxonomy and amino acid composition alone score for the non-enzyme label on each split",
        "release_record.json": "the receipts the release binds",
        "pipeline.md": "the pipeline as text, for agents",
        "card_facts.json": "the numbers this card states, written by the exporter from the build's receipts",
        "evaluation_negatives.parquet": "the measured non-interactions, for evaluation only, with each protein's split and distance from train",
        "sequences.parquet": "every distinct protein sequence of the splits and the panel, with its SHA-256",
        "ligands.parquet": "every distinct ligand of the splits and the panel: SMILES, SELFIES, symbol count and whether the SELFIES store holds it",
        "partners.json": "every source record behind each pair, including measurements the evidence rule set aside",
        "evaluation_panel.json": "how the evaluation panel was drawn, its counts by novelty and the findings it passed",
        "mask_tables.npz": "the lookup tables that cut the off-diagonal negatives of a batch (foundry.training.pair_masks), built from the train positives",
        "mask_report.json": "what the mask removes from sampler batches, check by check, and how its strata were measured",
        **facts.file_notes,
    }
    # A note the facts declare is listed even when the folder lacks the file: a release amended in place keeps files the folder does not hold.
    present = {path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file()} | {"pipeline.md"} | set(facts.file_notes)
    lines = ["| File | What it is |", "|---|---|"]
    if facts.configs == (("default", ""),):
        lines.append("| `train.parquet`, `validation.parquet`, `test.parquet`, `calibration.parquet` | the four splits |")
    else:
        lines += [f"| `{directory}{{train,validation,test,calibration}}.parquet` | the four splits of `{config}` |" for config, directory in facts.configs]
    lines += [f"| `{name}` | {note} |" for name, note in notes.items() if name in present]
    return "\n".join(lines)


def _split_table(facts: Facts) -> str:
    """Rows, proteins and clusters per split; the proteins column is left out when it repeats the rows."""
    proteins = any(facts.splits[name]["proteins"] not in (None, facts.splits[name]["rows"]) for name in SPLITS)
    head = ["Split", "Rows", *([facts.unit_label] if proteins else []), facts.cluster_label]
    lines = [f"| {' | '.join(head)} |", f"|{'---|' * len(head)}"]
    for name in SPLITS:
        counts = facts.splits[name]
        cells = [name, f"{counts['rows']:,}", *([_number(counts["proteins"])] if proteins else []), _number(counts["clusters"])]
        lines.append(f"| {' | '.join(cells)} |")
    return "\n".join(lines)


def _number(value: int | None) -> str:
    return f"{value:,}" if value is not None else "not recorded"


def _columns_note(release: Path, facts: Facts) -> str:
    """What to read first: the readable columns when the file has them, else where a pair's proteins are."""
    names = _columns_of(release, facts.configs[0][1])
    if "observation_json" not in names:
        return "The first columns are the ones to read. The ids and cluster columns after them are for joining and audit."
    if "seqs" in names:
        return ("Each row is one pair. The first columns are the ones to read; the hash ids after them join to the embedding stores, and the "
                "two JSON records at the end keep the pair's evidence and source records for audit.")
    return ("Each row is one pair, kept as two JSON records so that a pair keeps the role of each partner. The example rows show what a "
            "row says. The protein sequences are in `canonical_split.json`, keyed by the protein ids inside the records.")


def _task_line(facts: Facts) -> str:
    if facts.task:
        return facts.task
    if facts.task_type == "multilabel":
        return f"multilabel classification over {len(facts.labels)} labels"
    endpoint = facts.endpoint.replace("_", " ")
    if facts.task_type == "regression":
        return f"regression of {endpoint}" + (f" in {facts.units}" if facts.units else "") + (f", {facts.transform}" if facts.transform not in (None, "identity") else "")
    return f"{facts.task_type} classification of {endpoint}"


def _configs_yaml(facts: Facts) -> str:
    """The front matter's `configs`, so the viewer lists every split of every config, `calibration` with the rest."""
    lines: list[str] = []
    for index, (config, directory) in enumerate(facts.configs):
        splits = ", ".join(f"{{split: {name}, path: {directory}{name}.parquet}}" for name in SPLITS)
        lines += [f"- config_name: {config}", *(["  default: true"] if index == 0 and len(facts.configs) > 1 else []), f"  data_files: [{splits}]"]
    return "\n".join(lines)


def render_card(template: str, *, release: Path, readme: str, pipeline: str, status: str | None = None) -> str:
    """The card for a release: the template with every placeholder filled, which must leave no `{{` behind."""
    facts, sections = load_facts(release), _sections(readme)
    if MAY_SAY not in sections:
        raise CardError(f"the README has no '## {MAY_SAY}' section; add it before the card is rendered")
    summary = _prose(_summary(readme, sections))
    if len(summary.split()) > SUMMARY_WORDS:
        raise CardError(f"the summary is {len(summary.split())} words; the card allows {SUMMARY_WORDS}. Shorten it or add a '## Summary' section")
    sources, licence = _sources(pipeline)
    homology = facts.homology
    if facts.criteria is None and facts.acceptance is None:
        raise CardError("the facts give neither acceptance criteria nor an acceptance line")
    rationale = f"<details><summary>Why this level</summary>\n\n{homology.rationale}\n\n</details>\n" if homology.rationale else ""
    tags = "\n".join(f"- {tag}" for tag in ("protein", "biology", facts.task_type, "synthyra"))
    values = {
        "title": _title(readme), "dataset_id": facts.dataset_id, "summary": summary, "licence": licence,
        "size_category": _size_category(facts.rows), "tags": tags, "task_line": _task_line(facts),
        "rows_line": f"{facts.rows:,} across four splits" + (f" ({facts.rows_note})" if facts.rows_note else ""),
        "homology_line": f"enforced at {homology.enforced:g} identity" + (" on pairs" if homology.unit == "pair" else "") + (f", measured at {homology.measured:g}" if homology.measured else ""),
        "acceptance_line": facts.acceptance or f"{facts.criteria[0]:,} of {facts.criteria[1]:,} criteria passed", "configs": _configs_yaml(facts),
        "status_line": status or "private, building", "homology_claim": homology.claim(), "homology_table": homology.table(),
        "homology_clusters": f"Clustered at {', '.join(f'{level:g}' for level in homology.levels)} identity with {homology.coverage:g} coverage of both sequences (MMseqs2).",
        "split_table": _split_table(facts), "homology_rationale": rationale, "build_steps": _build_steps(sections), "pipeline_steps": _steps(pipeline),
        "columns_note": _columns_note(release, facts), "columns_table": _columns_table(release, facts), "example_rows": _example_rows(release, facts), "may_say": _without_workspace_links(sections[MAY_SAY]),
        "sources": sources, "files_table": _files_table(release, facts),
    }
    card = template
    for key, value in values.items():
        card = card.replace("{{" + key + "}}", value)
    left = PLACEHOLDER.findall(card)
    if left:
        raise CardError(f"the template holds placeholders this renderer does not fill: {sorted(set(left))}")
    return re.sub(r"\n{3,}", "\n\n", card).rstrip() + "\n"


def check_card(card: str, release: Path) -> list[str]:
    """Every way the card departs from the standard; empty when it conforms."""
    problems: list[str] = []
    if PLACEHOLDER.search(card):
        problems.append("the card holds an unresolved placeholder")
    if not card.startswith("---\n") or "\n---\n" not in card[4:]:
        return [*problems, "the card has no YAML front matter"]
    front, body = card[4:].split("\n---\n", 1)
    try:
        facts, unsupported = load_facts(release), None
    except CardError as error:
        facts, unsupported = None, error
    configs = facts.configs if facts else (("default", ""),)
    if any(key in front for key in MANIFEST_KEYS) or re.search(r"^(id|kind|status):", front, flags=re.MULTILINE):
        problems.append("the front matter is the workspace manifest; the Hub card carries license, tags and configs only")
    for config, directory in configs:
        for name in SPLITS:
            if f"path: {directory}{name}.parquet" not in front:
                problems.append(f"the front matter does not list the {name} split of {config} under configs")
    if HEX64.search(body):
        problems.append("the card shows a SHA-256; digests belong in dataset.json and the release record, not in the card")
    if "/vol/" in body or "C:/" in body or "~/" in body:
        problems.append("the card names a workspace or volume path")
    if EM_DASH in card:
        problems.append("the card uses an em dash")
    if len(card.splitlines()) > MAX_LINES:
        problems.append(f"the card runs to {len(card.splitlines())} lines; the limit is {MAX_LINES}")
    headings = re.findall(r"^## +(.+?)\s*$", body, flags=re.MULTILINE)
    if tuple(headings) != HEADINGS:
        problems.append(f"the sections are {headings}, not the standard {list(HEADINGS)} in order")
    if len(re.findall(r"^# +", body, flags=re.MULTILINE)) != 1:
        problems.append("the card needs exactly one title")
    built = body.split("## How it was built", 1)[-1].split("\n## ", 1)[0]
    if len(re.findall(r"^\d+\. ", built, flags=re.MULTILINE)) < MIN_STEPS:
        problems.append(f"'How it was built' does not open with a numbered list of at least {MIN_STEPS} steps")
    if "](../" in body:
        problems.append("the card links a workspace path")
    if not re.search(r"!\[How [^\]]+ was built\]\(pipeline\.(?:svg|gif)\)", body) or "(pipeline.md)" not in body:
        problems.append("the card does not show the pipeline map and link its text form")
    if facts is None:
        return [*problems, f"the release cannot support a card: {unsupported}"]
    opening = body.split("## Homology leakage", 1)[-1].lstrip("\n")
    if not opening.startswith(facts.homology.claim()):
        problems.append("the homology section does not open with the claim the acceptance report supports: " + facts.homology.claim())
    section = body.split("## Columns", 1)[-1].split("\n## ", 1)[0]
    chunks = [section] if len(configs) == 1 else re.split(r"^### .+$", section, flags=re.MULTILINE)[1:]
    if len(chunks) != len(configs):
        problems.append(f"the Columns section needs one table under a heading for each of the {len(configs)} configs")
    listed = _new_columns(release, facts)
    for (config, _directory), chunk, (_config, names, new) in zip(configs, chunks, listed, strict=False):
        actual = new
        if any(name in REDUNDANT for name in names):
            problems.append(f"the split files of {config} hold columns a reader does not need: {sorted(set(names) & set(REDUNDANT))}")
        documented = re.findall(r"^\| `([a-z_0-9]+)` \|", chunk, flags=re.MULTILINE)
        if documented != actual:
            problems.append(f"the Columns table{'' if len(configs) == 1 else ' of ' + config} lists {documented}, but the split files hold {actual}")
    return problems


def check_readme(readme: str) -> list[str]:
    """How a dataset's workspace README cannot yet feed a card: its summary, its numbered steps and its statement of what a model may say."""
    problems: list[str] = []
    sections = _sections(readme)
    try:
        summary = _prose(_summary(readme, sections))
        if len(summary.split()) > SUMMARY_WORDS:
            problems.append(f"the summary is {len(summary.split())} words; the card allows {SUMMARY_WORDS}")
    except CardError as error:
        problems.append(str(error))
    try:
        _build_steps(sections)
    except CardError as error:
        problems.append(str(error))
    if MAY_SAY not in sections:
        problems.append(f"the README has no '## {MAY_SAY}' section")
    if EM_DASH in readme:
        problems.append("the README uses an em dash")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    rendering = commands.add_parser("render", help="write README.md for a release into a card directory")
    rendering.add_argument("--release", type=Path, required=True)
    rendering.add_argument("--readme", type=Path, required=True, help="the dataset's workspace README")
    rendering.add_argument("--map", type=Path, required=True, help="the dataset directory holding its rendered pipeline.md")
    rendering.add_argument("--template", type=Path, required=True)
    rendering.add_argument("--output", type=Path, required=True)
    rendering.add_argument("--status", help="the status line, default 'private, building'")
    reading = commands.add_parser("check-readme", help="list how a dataset README cannot yet feed a card, before any release exists")
    reading.add_argument("--readme", type=Path, required=True)
    checking = commands.add_parser("check", help="list how a card departs from the standard")
    checking.add_argument("--release", type=Path, required=True)
    checking.add_argument("--card", type=Path, required=True, help="a directory holding README.md")
    arguments = parser.parse_args(argv)
    if arguments.command == "render":
        card = render_card(arguments.template.read_text(encoding="utf-8"), release=arguments.release,
                           readme=arguments.readme.read_text(encoding="utf-8"),
                           pipeline=(arguments.map / "pipeline.md").read_text(encoding="utf-8"), status=arguments.status)
        arguments.output.mkdir(parents=True, exist_ok=True)
        (arguments.output / "README.md").write_text(card, encoding="utf-8", newline="\n")
        print(json.dumps({"card": str(arguments.output / "README.md"), "lines": len(card.splitlines())}))
        return 0
    if arguments.command == "check-readme":
        problems = check_readme(arguments.readme.read_text(encoding="utf-8"))
        print(json.dumps({"problems": problems}, indent=1))
        return 1 if problems else 0
    problems = check_card((arguments.card / "README.md").read_text(encoding="utf-8"), arguments.release)
    print(json.dumps({"problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
