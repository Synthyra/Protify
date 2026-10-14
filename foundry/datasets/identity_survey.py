"""Survey the identity levels a split could enforce, before any split is made.

A split enforces one level of the hierarchy: no held-out protein has a relative at or above that identity in another
split, and every level below it is measured. Logan, 2026-09-29: cluster at 0.9, 0.7, 0.5 and 0.3, then pick the one that
balances the cluster count of each split while minimizing homology leakage; 0.9 for a very small, non-diverse dataset,
0.3 for a large, diverse one. `survey_identity_levels` puts numbers on that choice, and
docs/conventions/splits.md#choosing-the-enforced-identity says how to read them.

For each level the survey counts the units (clusters at that level) the dataset has, the proteins in units too large to
hold out, and the units a held-out split gets. When proteins carry labels it also counts the labels the level takes out of
reach: a label carried by enough units at the finest level to reach every held-out split, but by too few at this one,
because its proteins sit in a few large families. A level is well formed when it passes all three checks, and the
recommendation is the lowest well-formed level, never above `ceiling`; a dataset with no well-formed level is small or
concentrated, and the recommendation is the finest level it may enforce.
"""

from __future__ import annotations

import math

from collections import Counter
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .hierarchy import ClusterHierarchy
from .stratified import DEFAULT_EVALUATION_SHARE, DEFAULT_SETTINGS, EVALUATION_SPLITS, _cluster_quotas


# Lost labels a survey names per level; the count is always complete.
NAMED_LOST_LABELS = 25


@dataclass(frozen=True, slots=True)
class SurveySettings:
    """The limits a level must stay within to leave a well-formed split."""

    # A held-out split needs this many units for a metric over it to average over independent families: the standard error
    # of a mean over 50 units is 14 percent of the spread between them.
    min_units_per_split: int = 50
    # The most proteins that may sit in units too large to hold out, which a split can only put in train.
    max_locked_share: float = 0.10
    # A label reaches every held-out split when it has this many units to give: 5 percent of them in each split is one.
    min_label_units: int = 20
    # The most labels a level may take out of reach, as a share of the labels that had enough units at the finest level.
    max_lost_label_share: float = 0.10
    # As `StratifiedSettings.max_unit_share`: a unit over this share of a held-out split's clusters is never held out.
    max_unit_share: float = DEFAULT_SETTINGS.max_unit_share

    def __post_init__(self) -> None:
        for name in ("min_units_per_split", "min_label_units"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("max_locked_share", "max_lost_label_share", "max_unit_share"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")


DEFAULT_SURVEY_SETTINGS = SurveySettings()


@dataclass(frozen=True, slots=True)
class LevelSurvey:
    """What enforcing one level would leave: its units, the proteins it locks in train, and the labels it takes out of reach."""

    level: float
    # Clusters at this level, the units a split holds out whole, and how many of them hold a single finest cluster.
    units: int
    singleton_units: int
    # Finest clusters in the largest unit.
    largest_unit: int
    # The share of proteins in units over the size a split may hold out.
    locked_share: float
    # The units a held-out split gets, when the split draws units of the size the holdable ones have on average.
    units_per_split: float
    # Labels with enough holdable units to reach every held-out split, and those the finest level carried and this one does not.
    # None without labels.
    labels_carried: int | None
    labels_lost: int | None
    lost_share: float | None
    lost_labels: tuple[str, ...]
    failures: tuple[str, ...]

    @property
    def well_formed(self) -> bool:
        return not self.failures


@dataclass(frozen=True, slots=True)
class IdentitySurvey:
    """The levels of one hierarchy, lowest identity first, and the one to enforce."""

    levels: tuple[LevelSurvey, ...]
    recommended: float
    # No level has enough units per split, so the dataset is too small to enforce less than its finest level and its error bars are wide.
    small_dataset: bool
    ceiling: float
    proteins: int
    finest_clusters: int
    # The clusters a held-out split takes, and the size above which a unit is never held out.
    clusters_per_split: int
    max_unit: int
    labels: int | None
    settings: SurveySettings

    def level(self, identity: float) -> LevelSurvey:
        return next(item for item in self.levels if item.level == identity)

    def to_dict(self) -> dict[str, Any]:
        """A record that reads back from JSON unchanged: every field, and each level with whether it is well formed."""
        levels = [
            {**asdict(item), "lost_labels": list(item.lost_labels), "failures": list(item.failures), "well_formed": item.well_formed}
            for item in self.levels
        ]
        return {**asdict(self), "levels": levels}

    def table(self) -> str:
        """The survey as a Markdown table for a decision record, lowest identity first."""
        lines = ["| Level | Units | Singleton units | Largest unit | Locked share | Units per split | Labels lost | Well formed |",
                 "|---|---:|---:|---:|---:|---:|---:|---|"]
        for item in self.levels:
            lost = "" if item.labels_lost is None else f"{item.labels_lost} of {item.labels_carried + item.labels_lost}"
            verdict = "yes" if item.well_formed else "no: " + ", ".join(item.failures)
            lines.append(f"| {item.level} | {item.units:,} | {item.singleton_units:,} | {item.largest_unit:,} | {item.locked_share:.1%} "
                         f"| {item.units_per_split:,.0f} | {lost} | {verdict} |")
        return "\n".join(lines)


def survey_identity_levels(
    hierarchy: ClusterHierarchy, label_sets: Mapping[str, Collection[str]] | None = None, *,
    settings: SurveySettings = DEFAULT_SURVEY_SETTINGS, ceiling: float = 0.9, evaluation_share: float = DEFAULT_EVALUATION_SHARE,
) -> IdentitySurvey:
    """Survey every level of `hierarchy`, and recommend the lowest one that leaves a well-formed split.

    `label_sets` maps every protein to the labels it carries, each label to count once however many rows a protein has; a protein
    with no label maps to an empty collection. The finest clusters are counted the way the cluster policy counts them, so a
    unit's size is the number of finest clusters it holds and a held-out split takes `evaluation_share` of them (5 percent unless
    the split policy says another). `ceiling` is the highest level the dataset may enforce, 0.5 for an interaction dataset.
    """
    if ceiling not in hierarchy.levels:
        raise ValueError(f"the ceiling is one of the hierarchy's levels {hierarchy.levels}, got {ceiling!r}")
    finest = hierarchy.members[hierarchy.finest]
    proteins = sum(len(members) for members in finest.values())
    clusters_per_split = min(_cluster_quotas(len(finest), evaluation_share)[name] for name in EVALUATION_SPLITS)
    max_unit = max(1, math.floor(settings.max_unit_share * clusters_per_split))
    lineage = {cluster: hierarchy.lineage(cluster) for cluster in finest}
    finest_of = {protein: cluster for cluster, members in finest.items() for protein in members}
    if label_sets is not None and set(label_sets) != set(finest_of):
        missing, unexpected = sorted(set(finest_of) - set(label_sets)), sorted(set(label_sets) - set(finest_of))
        raise ValueError(f"label_sets must cover the hierarchy's proteins exactly: missing={missing[:5]}, unexpected={unexpected[:5]}")

    surveys: dict[float, LevelSurvey] = {}
    carried_at_finest: set[str] = set()
    # The finest level first, so that what a coarser level loses is measured against it.
    for index, level in enumerate(hierarchy.levels):
        size: Counter[str] = Counter()
        held: Counter[str] = Counter()
        for cluster, chain in lineage.items():
            size[chain[index]] += 1
            held[chain[index]] += len(finest[cluster])
        holdable = {unit for unit, count in size.items() if count <= max_unit}
        locked = sum(count for unit, count in held.items() if unit not in holdable)
        mean = sum(size[unit] for unit in holdable) / len(holdable) if holdable else 0.0
        units_per_split = clusters_per_split / mean if mean else 0.0
        carried = lost = None
        lost_labels: tuple[str, ...] = ()
        if label_sets is not None:
            span: dict[str, set[str]] = {}
            for protein, labels in label_sets.items():
                unit = lineage[finest_of[protein]][index]
                if unit in holdable:
                    for label in labels:
                        span.setdefault(label, set()).add(unit)
            reach = {label for label, units in span.items() if len(units) >= settings.min_label_units}
            if index == 0:
                carried_at_finest = reach
            carried, lost_set = len(reach), carried_at_finest - reach
            lost, lost_labels = len(lost_set), tuple(sorted(lost_set)[:NAMED_LOST_LABELS])
        lost_share = None if lost is None else lost / len(carried_at_finest) if carried_at_finest else 0.0
        failures = tuple(name for name, failed in (
            ("units_per_split", units_per_split < settings.min_units_per_split),
            ("locked_share", locked / proteins > settings.max_locked_share),
            ("labels_lost", lost_share is not None and lost_share > settings.max_lost_label_share),
        ) if failed)
        surveys[level] = LevelSurvey(
            level, len(size), sum(count == 1 for count in size.values()), max(size.values()), locked / proteins, units_per_split,
            carried, lost, lost_share, lost_labels, failures,
        )
    ordered = tuple(surveys[level] for level in sorted(surveys))
    allowed = [item for item in ordered if item.level <= ceiling]
    formed = [item for item in allowed if item.well_formed]
    recommended = formed[0].level if formed else allowed[-1].level
    return IdentitySurvey(
        ordered, recommended, "units_per_split" in surveys[hierarchy.finest].failures, ceiling, proteins, len(finest), clusters_per_split,
        max_unit, None if label_sets is None else len(carried_at_finest), settings,
    )
