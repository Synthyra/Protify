"""Identity-stratified splits: train, validation, calibration, and test, measured by distance from train.

Each evaluation split holds equal shares of proteins in four strata of identity to train: far
(below 0.3, or no hit), 0.3 to 0.5, 0.5 to 0.7, and 0.7 to 0.9. A protein at 0.9 or more is a
near-duplicate, and near-duplicates never cross splits; 0.9 is the only threshold enforced, and
the rest are measured. The accepted rule is
docs/decisions/foundry/datasets/2026-09-24_identity_stratified_splits.md, with the calibration
split of 2026-09-24_calibration_split.md, and docs/conventions/splits.md states it for builders.

How a split is made, from a `ClusterHierarchy` at 0.9, 0.7, 0.5, and 0.3:

- Validation, calibration, and test each get n proteins: 5% of the dataset, raised to 500 and
  capped at 10%. Each asks for n/4 in each stratum.
- A stratum is filled by holding out whole clusters: far with 0.3 clusters, and each nearer
  stratum with clusters one level finer whose parent keeps at least half its proteins in train,
  so the held-out cluster keeps relatives in train at the stratum's distance.
- Far is filled first. Within a stratum the largest units go first, each to the evaluation split
  that most reduces the distance from its targets, proteins per stratum and label shares, with
  ties broken on the seeded hash `assign_clusters_to_splits` uses. The three evaluation splits
  have identical targets and are treated alike, so they stay exchangeable. No unit may exceed 10%
  of an evaluation split.
- A stratum a dataset cannot fill passes its shortfall to the next nearer stratum, and what the
  nearest leaves returns to the farther ones.
- Every held-out protein is then searched against train. A protein at 0.9 or more moves to train
  with its whole 0.9 cluster, and the search repeats until none remains. The measured stratum,
  not the level a protein was held out at, is what reports and calibration use.

A dataset whose evaluation splits cannot reach half of n, or whose largest 0.9 cluster holds most
of its proteins, raises `StratificationInfeasible` and declares another scheme by name.

`StratifiedSettings.enforced_identity` moves the enforced threshold from 0.9 to 0.7 or 0.5, for a
dataset that must claim more than separation from near-duplicates: the interaction datasets, whose
2026-09-29 rule (Logan, after Bernett et al. 2026 enforced 40 percent) is to enforce 0.5 and
measure 0.3. Only strata whose floor is below the enforced identity remain (far and 0.3 to 0.5 at
0.5), a held-out unit is a whole cluster at the enforced level or coarser, and repair returns a
flagged cluster to train with every cluster of its enforced-level parent, so no two splits share
an enforced-level cluster. The default keeps the historical 0.9.

`allocation_unit="cluster"` selects the canonical release's versioned policy instead:
85/5/5/5 of finest-level clusters with largest-remainder rounding, counted independently of
their protein sizes. Stratum quotas use that same unit. Missing strata pass their quota on. A
cluster that repair returns to train stays there, and the quota it freed is filled again from
clusters not yet held out, each searched and repaired in turn; exact quotas that no eligible
cluster can restore raise with realized counts. Its repair also runs the audit's other nine
directed searches: a train protein or another evaluation split's protein that is a
near-duplicate of a held-out protein, such as a fragment of it, sends that cluster back to
train too, so the audit that ends every build finds nothing to refuse. The default remains the
historical protein policy so existing experiment inputs do not change.

`unit_order="seeded"` draws the units of a stratum in a seeded pseudo-random order instead of largest first. Largest first
gives the evaluation splits the big families and the training split the rest, which measurably shifts the label
distribution of a held-out split (on subcellular localization the total variation from train's labels was 0.13, against 0.055
for a seeded order, 2026-09-29). Under the cluster policy a second pass then fills any room a unit that did not fit left, largest
first, so exact quotas still hold. `balance_labels=True` makes the cluster policy weigh each unit's labels as well: a unit goes
to the evaluation split whose label counts it moves closest to the labels' shares, and is held out only when that helps. Both
are recorded in the manifest's settings only when they are not at their historical values, so a manifest written before they
existed still validates (`recorded_settings`).

Repair measures the whole assignment once, then only what each round changed. A cluster that goes
back to train is searched as a query against the evaluation splits and as a reference for the
proteins still held out, and a cluster a refill holds out is searched against the rest. The
finished assignment is measured as a whole once more, which gives the reported identities and finds
any flag the partial searches could not. Given a search whose identity for a pair does not depend
on the other references, this reaches the split that measuring everything each round reaches;
`incremental=False` does that, for a comparison.

A cross-split flag comes with an identity and no cluster, so repair must find which cluster the flagged protein matched.
`matches` does it in one search per pair of splits and names every match, so all of them return to train in that round.
Without it repair halves the references with the identity search, one search per halving. Enforcing 0.3 on 42 thousand
proteins flagged so many that the halving spent the better part of an hour on searches of a few sequences that take about a
second each (2026-09-29), which is what `matches` replaces; `build_identity_stratified_split` supplies it.
"""

from __future__ import annotations

import bisect
import math

from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from .hierarchy import HIERARCHY_LEVELS, ClusterHierarchy, cluster_hierarchy
from .homology import (
    ALL_PAIRS,
    ANSWER_SOURCES,
    SEARCH_EACH,
    SearchConfig,
    answered_by,
    matching_references,
    max_identity_to_reference,
)
from .leakage import check_cross_split_homology
from .mmseqs import CommandRunner
from .splits import _seed_rank


if TYPE_CHECKING:
    # Imported for its annotation only: numpy loads only for a split that asks for the table.
    from .pair_table import PairTable


STRATIFIED_SPLIT_NAMES = ("train", "validation", "calibration", "test")
EVALUATION_SPLITS = STRATIFIED_SPLIT_NAMES[1:]
STRATA = ("far", "0.3-0.5", "0.5-0.7", "0.7-0.9")
NEAR_DUPLICATE = "near_duplicate"
NEAR_DUPLICATE_IDENTITY = 0.9
# The identities a split may enforce: an identity to train at or above it is never held out. The rest of the hierarchy is measured.
ENFORCEABLE_IDENTITIES = HIERARCHY_LEVELS
# The orders a stratum's units are drawn in: the historical largest first, or a seeded pseudo-random order.
UNIT_ORDERS = ("largest_first", "seeded")
DEFAULT_UNIT_ORDER = UNIT_ORDERS[0]
# How the first allocation deals units to the evaluation splits: the greedy draw above, or whole components of related sequences by
# iterative stratification of their labels (`component_allocation`), which needs the all-pairs table and balances labels.
LABEL_ALLOCATIONS = ("greedy", "components")
DEFAULT_LABEL_ALLOCATION = LABEL_ALLOCATIONS[0]
# Logan, 2026-09-29: protein-protein, protein-ligand and CAMP datasets never enforce a looser identity than 0.5, because Bernett et
# al. 2026 enforced 40 percent and a release cannot claim independence with less. Most datasets should enforce 0.3, and a small
# or a paired dataset may not be able to, which is why the floor for these families is 0.5.
INTERACTION_ENFORCED_IDENTITY_LIMIT = 0.5
# The paired tasks of those families, and the dataset ids that carry them: `protein_interaction_*`, `protein_ligand_interaction_*`, and
# `protein_annotation_*` (the shared annotation release that trains CAMP and the Translator).
INTERACTION_TASKS = ("ppi", "pli", "camp")
INTERACTION_DATASET_PREFIXES = ("protein_interaction_", "protein_ligand_interaction_", "protein_annotation_")
# Each stratum's lowest identity to train; far also holds proteins with no hit.
STRATUM_FLOORS = {"far": 0.0, "0.3-0.5": 0.3, "0.5-0.7": 0.5, "0.7-0.9": 0.7}
# The hierarchy level whose clusters each stratum holds out.
UNIT_LEVEL = {"far": 0.3, "0.3-0.5": 0.5, "0.5-0.7": 0.7, "0.7-0.9": 0.9}


def strata_below(enforced_identity: float) -> tuple[str, ...]:
    """The strata whose identity to train lies below `enforced_identity`, far first: all four at 0.9, far and 0.3 to 0.5 at 0.5."""
    if enforced_identity not in ENFORCEABLE_IDENTITIES:
        raise ValueError(f"the enforced identity is one of {ENFORCEABLE_IDENTITIES}, got {enforced_identity!r}")
    return tuple(stratum for stratum in STRATA if STRATUM_FLOORS[stratum] < enforced_identity)


def enforced_identity_ceiling(*, dataset: str | None = None, task: str | None = None) -> float:
    """The highest identity a dataset, named by its id or its paired task, may enforce: 0.5 for an interaction dataset, else 0.9."""
    if task in INTERACTION_TASKS or (dataset is not None and dataset.startswith(INTERACTION_DATASET_PREFIXES)):
        return INTERACTION_ENFORCED_IDENTITY_LIMIT
    return max(ENFORCEABLE_IDENTITIES)


def require_interaction_identity(enforced_identity: float, *, dataset: str | None = None, task: str | None = None) -> None:
    """Raise when a protein-protein, protein-ligand or CAMP dataset, named by its id or its paired task, enforces more than 0.5.

    The rule lives in docs/conventions/splits.md, with its reasons; this assertion is what keeps a builder from quietly
    loosening it.
    """
    if enforced_identity > enforced_identity_ceiling(dataset=dataset, task=task):
        raise ValueError(
            f"{dataset or task} is an interaction dataset, which must enforce an identity of at most "
            f"{INTERACTION_ENFORCED_IDENTITY_LIMIT} between splits, not {enforced_identity}"
        )


# Identity to train: hits at e-value 1e-3 or below that cover at least half of the query.
MEASURE_SEARCH = SearchConfig(evalue_max=1e-3, min_coverage=0.5, coverage_mode=2)
# Conformal calibration merges adjacent strata until each group holds this many proteins.
CALIBRATION_GROUP_MINIMUM = 100

# Query proteins and reference proteins, each identifier to sequence, to each query's highest
# identity to the references, 0.0 without a hit; `max_identity_to_reference` with MEASURE_SEARCH.
# The protein policy asks it for held-out proteins against train. The cluster policy also asks it
# for every other pair of splits, as the audit does, and for a few queries against part of a split.
IdentitySearch = Callable[[Mapping[str, str], Mapping[str, str]], Mapping[str, float]]
# Query proteins, reference proteins and an identity to the references each query is at that identity or more to, sorted, an
# empty tuple for a query with none; `matching_references` with MEASURE_SEARCH. Repair asks it to name what a flagged
# protein matches. A split given none finds that out by halving the references with `IdentitySearch`, which costs a search
# for each halving and is slow when a split enforces 0.3, where near-duplicates across splits number in the thousands.
IdentityMatches = Callable[[Mapping[str, str], Mapping[str, str], float], Mapping[str, Collection[str]]]
# An identity and the share of the residues a reference set holds, to the proteins' names in the table's order and the table rows of each
# pair of proteins at that identity or more that a search against such a reference counts: `PairTable.counted_hits`. The
# `components` allocation joins what these pairs relate.
Relatives = Callable[[float, float], tuple[Sequence[str], Any, Any]]
# Told after the first allocation ("fill") and after each repair round and refill, by name and facts as keywords, such as
# `("repair", iteration=3, returned=41, ...)`. Every call also carries `requested` and `realized`, each evaluation split's quota and
# what it holds now, in the unit of the quota (clusters under the cluster policy), so a split that cannot fill its quotas shows the
# gap in the round it opens instead of when the split ends.
SplitProgress = Callable[..., None]


class StratificationInfeasible(ValueError):
    """The dataset cannot be split by identity strata, and must declare another scheme by name.

    The other schemes are positional folds for one protein's variants, or the published split.
    """


DEFAULT_EVALUATION_SHARE = 0.05
# Logan, 2026-09-29: a smaller evaluation share is allowed when the identity it must hold leaves too few groups, but never below
# 1,000 sequences in a split, and 1,000 finest clusters is preferred. The validator refuses the sequences and the recipes ask for
# the clusters in their acceptance floors.
MINIMUM_REDUCED_EVALUATION_PROTEINS = 1000


@dataclass(frozen=True, slots=True)
class StratifiedSettings:
    """The sizes, limits, and seed of one identity-stratified split."""

    # Each evaluation split: this share of proteins, raised to `evaluation_minimum`, capped at `evaluation_cap`. Under the cluster
    # policy it is the share of finest-level clusters, exact, in whole thousandths (0.05 is 85/5/5/5), and neither the minimum nor the
    # cap applies. A share below the default is for a paired dataset that cannot fill 5 percent at the identity it enforces.
    evaluation_share: float = DEFAULT_EVALUATION_SHARE
    evaluation_minimum: int = 500
    evaluation_cap: float = 0.10
    # The largest unit an evaluation split may take, as a share of its size n.
    max_unit_share: float = 0.10
    # Below this share of n in any evaluation split, the dataset needs another scheme.
    min_filled_share: float = 0.5
    # A 0.9 cluster holding more than this share of proteins means another scheme, as for a
    # deep mutational scan of one protein.
    dominant_cluster_share: float = 0.5
    seed: int = 42
    # Historical studies count proteins. Canonical releases explicitly select whole
    # finest-level clusters, with fixed 85/5/5/5 shares and no protein-count resizing.
    allocation_unit: str = "protein"
    # An identity to train at or above this is never held out: 0.9 keeps near-duplicates out of the evaluation
    # splits, and 0.5 also keeps out every relative that shares half its sequence. The strata above it are gone.
    enforced_identity: float = NEAR_DUPLICATE_IDENTITY
    # The order a stratum's units are drawn in. Largest first hands the evaluation splits the big families, which skews their label
    # distribution away from train's; a seeded order makes each evaluation split a sample of the dataset's families.
    unit_order: str = DEFAULT_UNIT_ORDER
    # Under the cluster policy, let each unit's labels choose its split and whether it is held out, after the exact cluster counts.
    # The protein policy always weighs labels when it has them. Needs `labels`.
    balance_labels: bool = False
    # How the first allocation deals units out. `components` keeps whole components of related sequences together, so repair has
    # little left to return to train, and stratifies their labels rarest first; it needs the cluster policy, `balance_labels` and
    # an enforced identity with one stratum (0.3), and `build_identity_stratified_split` answering searches from the all-pairs table.
    label_allocation: str = DEFAULT_LABEL_ALLOCATION
    # Never hold out a unit that would leave a label with fewer than this many train proteins when the dataset holds at least that
    # many, so a vocabulary fitted on train at this support keeps every label the dataset can give it. 0 turns it off. Needs
    # `labels`, and the greedy allocation, since the components allocation deals out whole components at once.
    train_label_floor: int = 0

    def __post_init__(self) -> None:
        if self.allocation_unit not in ("protein", "cluster"):
            raise ValueError("allocation_unit must be 'protein' or 'cluster'")
        if self.unit_order not in UNIT_ORDERS:
            raise ValueError(f"unit_order is one of {UNIT_ORDERS}, got {self.unit_order!r}")
        if type(self.balance_labels) is not bool:
            raise TypeError("balance_labels must be a boolean")
        if self.label_allocation not in LABEL_ALLOCATIONS:
            raise ValueError(f"label_allocation is one of {LABEL_ALLOCATIONS}, got {self.label_allocation!r}")
        if self.label_allocation == "components" and (self.allocation_unit != "cluster" or not self.balance_labels or len(strata_below(self.enforced_identity)) != 1):
            raise ValueError("label_allocation 'components' needs the cluster policy, balance_labels, and an enforced identity with one stratum (0.3)")
        strata_below(self.enforced_identity)
        for name in ("evaluation_share", "evaluation_cap", "max_unit_share", "min_filled_share", "dominant_cluster_share"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
        if self.evaluation_share > self.evaluation_cap:
            raise ValueError("evaluation_share cannot exceed evaluation_cap")
        if 3 * self.evaluation_cap >= 1.0:
            raise ValueError("three evaluation splits at evaluation_cap must leave proteins for train")
        for name in ("evaluation_minimum", "seed", "train_label_floor"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise TypeError(f"{name} must be an integer")
        if self.evaluation_minimum < 0:
            raise ValueError("evaluation_minimum must be nonnegative")
        if self.train_label_floor < 0:
            raise ValueError("train_label_floor must be nonnegative")
        if self.train_label_floor and self.label_allocation == "components":
            raise ValueError("train_label_floor needs the greedy allocation")


DEFAULT_SETTINGS = StratifiedSettings()


def recorded_settings(settings: StratifiedSettings) -> dict[str, object]:
    """The settings as a manifest records them, leaving out each option at its historical value.

    `unit_order` and `balance_labels` came after manifests were published. Omitting them at their defaults keeps every earlier
    manifest valid against the settings that produced it, and the digest of a rebuilt default split unchanged.
    """
    record = asdict(settings)
    if settings.unit_order == DEFAULT_UNIT_ORDER:
        del record["unit_order"]
    if not settings.balance_labels:
        del record["balance_labels"]
    if settings.label_allocation == DEFAULT_LABEL_ALLOCATION:
        del record["label_allocation"]
    if not settings.train_label_floor:
        del record["train_label_floor"]
    return record


def _label_tuple(label: object) -> tuple[str, ...]:
    """The labels of one protein: a class as text, or each member of a collection of labels once, sorted."""
    if isinstance(label, (list, tuple, set, frozenset)):
        return tuple(sorted({str(item) for item in label}))
    return (str(label),)


@dataclass(frozen=True, slots=True)
class IdentityStratifiedSplit:
    """Every protein's split, what each evaluation protein was held out for and measured at, and the manifest."""

    protein_to_split: dict[str, str]
    # Evaluation proteins only: the stratum whose quota they were held out to fill.
    held_out_stratum: dict[str, str]
    # Evaluation proteins only: the highest identity to the final train split.
    identity_to_train: dict[str, float]
    manifest: dict[str, object]

    @property
    def measured_stratum(self) -> dict[str, str]:
        enforced = float(self.manifest.get("near_duplicate_identity", NEAR_DUPLICATE_IDENTITY))
        return {protein: identity_stratum(identity, enforced) for protein, identity in self.identity_to_train.items()}

    def to_dict(self) -> dict[str, object]:
        """A JSON-compatible record: the manifest, then each protein's split and measurement."""
        return {
            **self.manifest,
            "protein_to_split": dict(sorted(self.protein_to_split.items())),
            "held_out_stratum": dict(sorted(self.held_out_stratum.items())),
            "identity_to_train": dict(sorted(self.identity_to_train.items())),
        }


def identity_stratum(identity: float, enforced_identity: float = NEAR_DUPLICATE_IDENTITY) -> str:
    """The stratum of an identity to train, or `NEAR_DUPLICATE` at `enforced_identity` or more (0.9 unless a split enforces less)."""
    if isinstance(identity, bool) or not 0.0 <= identity <= 1.0:
        raise ValueError(f"identity must be in [0, 1], got {identity!r}")
    if identity >= enforced_identity:
        return NEAR_DUPLICATE
    return next(stratum for stratum in reversed(strata_below(enforced_identity)) if identity >= STRATUM_FLOORS[stratum])


def evaluation_split_size(total: int, settings: StratifiedSettings = DEFAULT_SETTINGS) -> int:
    """n, the proteins each evaluation split asks for: a share of `total`, raised to a minimum, capped."""
    if settings.allocation_unit == "cluster":
        raise ValueError("cluster allocation has separate whole-cluster quotas for all four splits")
    requested = max(math.ceil(total * settings.evaluation_share), settings.evaluation_minimum)
    return min(requested, math.floor(total * settings.evaluation_cap))


def stratum_quotas(size: int, strata: Sequence[str] = STRATA) -> dict[str, int]:
    """`size` split over `strata` (all four by default) as evenly as integers allow, the farther strata taking the remainder."""
    base, extra = divmod(size, len(strata))
    return {stratum: base + (index < extra) for index, stratum in enumerate(strata)}


def _cluster_quotas(total: int, evaluation_share: float = DEFAULT_EVALUATION_SHARE) -> dict[str, int]:
    """Hamilton allocation of whole clusters, 85/5/5/5 unless a share is stated; equal remainders follow the recorded split order.

    `evaluation_share` is each evaluation split's share of the clusters, in thousandths, so 0.05 is the default 85/5/5/5 and 0.015 is
    95.5/1.5/1.5/1.5. The default allocates exactly as it did before the share was a setting.
    """
    thousandths = round(evaluation_share * 1000)
    if abs(evaluation_share * 1000 - thousandths) > 1e-6 or not 0 < 3 * thousandths < 1000:
        raise ValueError(f"the evaluation share must be a whole number of thousandths under one third, got {evaluation_share!r}")
    numerators = (1000 - 3 * thousandths, thousandths, thousandths, thousandths)
    sizes = [total * numerator // 1000 for numerator in numerators]
    order = sorted(range(4), key=lambda index: (-(total * numerators[index] % 1000), index))
    for index in order[:total - sum(sizes)]:
        sizes[index] += 1
    return dict(zip(STRATIFIED_SPLIT_NAMES, sizes, strict=True))


def quintile_labels(values: Mapping[str, float]) -> dict[str, str]:
    """A regression target's quintile, `q1` to `q5`, as the class each protein's label share counts.

    Cut points are the values at each fifth of the sorted values, so equal values share a quintile.
    """
    if not values:
        raise ValueError("no values to bin")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values.values()):
        raise ValueError("quintile values must be finite numbers")
    ordered = sorted(values.values())
    cuts = [ordered[math.floor(k * len(ordered) / 5)] for k in range(1, 5)]
    return {protein: f"q{bisect.bisect_right(cuts, value) + 1}" for protein, value in values.items()}


def calibration_groups(
    proteins_per_stratum: Mapping[str, int], minimum: int = CALIBRATION_GROUP_MINIMUM, strata: Sequence[str] = STRATA,
) -> list[list[str]]:
    """Adjacent strata merged, far first, until each group holds `minimum` calibration proteins.

    A last group short of `minimum` joins the one before it, so every group but a lone one holds
    at least `minimum`.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for stratum in strata:
        current.append(stratum)
        count += proteins_per_stratum.get(stratum, 0)
        if count >= minimum:
            groups.append(current)
            current, count = [], 0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return groups


def split_by_identity_strata(
    hierarchy: ClusterHierarchy,
    sequences: Mapping[str, str],
    search: IdentitySearch,
    *,
    labels: Mapping[str, object] | None = None,
    settings: StratifiedSettings = DEFAULT_SETTINGS,
    search_parameters: Mapping[str, object] | None = None,
    linked_groups: Mapping[str, Sequence[str]] | None = None,
    incremental: bool = True,
    matches: IdentityMatches | None = None,
    progress: SplitProgress | None = None,
    relatives: Relatives | None = None,
) -> IdentityStratifiedSplit:
    """Assign every protein of `hierarchy` to train, validation, calibration, or test by identity strata, then repair.

    `sequences` maps each protein to its sequence, for `search`. `labels`, when given, maps every
    protein to its class, or its quintile from `quintile_labels`, and each evaluation split then
    also aims for the dataset's label shares. `search_parameters` go into the manifest, to record
    how `search` measured identity. `linked_groups` names source groups whose proteins must
    stay together; overlapping groups form one allocation unit without changing cluster quotas.
    Under the cluster policy, each cluster repair returns to train is replaced by another one,
    which is measured and repaired in its turn, so exact quotas survive near-duplicates, and
    repair looks for near-duplicates across every pair of splits, not only held-out against train.
    `matches` names, in one search per pair of splits, every cluster a flagged protein is a
    near-duplicate of, so repair returns all of them in the round that flags it; without it, or for a
    protein it names nothing for, `search` halves the references, which names one cluster per round.
    `incremental=False` measures the whole assignment in every repair round instead of what the
    last round changed. It gives the same split, and only a comparison needs it. `progress` is
    called as `progress("fill", ...)` after the first allocation, as `progress("repair", iteration=..., ...)` after each repair
    round and as `progress("refill", ...)` after each refill, so a run that takes hours says where it stands. Each call also
    gives `requested` and `realized`, the quota of each evaluation split and what it holds, so a shortfall shows as it opens.
    `relatives` is what `label_allocation="components"` joins into units; see `Relatives`.
    """
    if hierarchy.levels != HIERARCHY_LEVELS:
        raise ValueError(f"identity strata need a hierarchy at {HIERARCHY_LEVELS}, got {hierarchy.levels}")
    proteins = hierarchy.proteins
    missing = sorted(set(proteins) - set(sequences))
    if missing:
        raise ValueError(f"sequences lack proteins of the hierarchy: {missing[:5]}")
    if labels is not None:
        missing, unexpected = sorted(set(proteins) - set(labels)), sorted(set(labels) - set(proteins))
        if missing or unexpected:
            raise ValueError(f"labels must cover the hierarchy's proteins exactly: missing={missing[:5]}, unexpected={unexpected[:5]}")

    state = _Assignment(hierarchy, labels, settings, linked_groups, incremental, matches, progress, relatives)
    state.fill()
    state.tell("fill", iteration=0)
    whole = False
    while True:
        state.repair(sequences, search, whole=whole)
        if state.refill():
            whole = False
            continue
        if state.exact():
            break
        # Rounds that measured only their changes end with one measurement of the finished split.
        whole = True
    return state.finished_split(state.identity, state.moves, search_parameters)


def build_identity_stratified_split(
    sequences: Mapping[str, str],
    directory: Path,
    *,
    labels: Mapping[str, object] | None = None,
    settings: StratifiedSettings = DEFAULT_SETTINGS,
    search_config: SearchConfig = MEASURE_SEARCH,
    runner: CommandRunner | None = None,
    linked_groups: Mapping[str, Sequence[str]] | None = None,
    checkpoint: Callable[[str, Callable[[], Any]], Any] | None = None,
    progress: SplitProgress | None = None,
    answer_from: str = SEARCH_EACH,
    **cluster_settings: Any,
) -> tuple[ClusterHierarchy, IdentityStratifiedSplit]:
    """Cluster `sequences` into the hierarchy with MMseqs2, then split by identity strata, searching with MMseqs2.

    Clusters go under `directory / "clusters"` and searches under `directory / "search"`, each a
    stage a rerun reuses. `cluster_settings` go to `MMseqsConfig`, such as `threads` or `binary`;
    `search_config` sets its own. `progress` hears of each repair round and refill.

    `answer_from=ALL_PAIRS` runs one search of every sequence against every other and answers each search repair and the
    audit ask from its hits (`PairTable`), where the default runs MMseqs2 for each: a search is a database build, an index
    and an alignment, and repair asks for dozens of them in every round. The manifest records `search_answered_by`. At a small
    evaluation share the table can miss what the audit of a finished release finds, since an e-value grows with the database
    searched: on 2026-10-04 a 97/1/1/1 split repaired from a table of 143,224 sequences left seven short train peptides at 0.54 to
    0.89 identity over 12 to 24 residues to held-out proteins, which a search against a held-out split of 1,200 sequences found.
    Repair such a split with the default.
    """
    if answer_from not in ANSWER_SOURCES:
        raise ValueError(f"answer_from is one of {ANSWER_SOURCES}, got {answer_from!r}")

    def stage(name: str, operation: Callable[[], Any]) -> Any:
        return operation() if checkpoint is None else checkpoint(name, operation)

    hierarchy = stage("clustering", lambda: cluster_hierarchy(sequences, directory / "clusters", runner=runner, **cluster_settings))
    tables: list[PairTable] = []

    def table() -> PairTable:
        if not tables:
            # numpy loads only for a split that asks for the table.
            from .pair_table import PairTable

            tables.append(PairTable.from_search(sequences, directory / "search", search_config, runner=runner))
        return tables[0]

    def search(queries: Mapping[str, str], references: Mapping[str, str]) -> Mapping[str, float]:
        if answer_from == ALL_PAIRS:
            return table().max_identity(queries, references)
        return max_identity_to_reference(queries, references, directory / "search", search_config, runner=runner)

    def matches(queries: Mapping[str, str], references: Mapping[str, str], identity: float) -> Mapping[str, Collection[str]]:
        if answer_from == ALL_PAIRS:
            return table().matching(queries, references, minimum_identity=identity)
        return matching_references(queries, references, directory / "search", search_config, minimum_identity=identity, runner=runner)

    def relatives(identity: float, reference_share: float) -> tuple[Sequence[str], Any, Any]:
        found = table()
        rows, targets = found.counted_hits(identity, reference_share)
        return found.names, rows, targets

    if settings.label_allocation == "components" and answer_from != ALL_PAIRS:
        raise ValueError("label_allocation 'components' joins the pairs of the all-against-all search: answer_from must be 'all_pairs'")
    split = stage("assignment", lambda: split_by_identity_strata(
        hierarchy, sequences, search, labels=labels, settings=settings,
        search_parameters=search_config.result_parameters(), linked_groups=linked_groups, matches=matches, progress=progress,
        relatives=relatives if settings.label_allocation == "components" else None,
    ))
    validate_identity_stratified_split(
        hierarchy, split, linked_groups=linked_groups, expected_allocation_unit=settings.allocation_unit,
    )
    if settings.allocation_unit == "cluster":
        sequences_by_split: dict[str, dict[str, str]] = {name: {} for name in STRATIFIED_SPLIT_NAMES}
        for protein, name in split.protein_to_split.items():
            sequences_by_split[name][protein] = sequences[protein]
        audit = stage("split_audit", lambda: check_cross_split_homology(
            sequences_by_split, search, search_parameters=search_config.result_parameters(),
            identity_threshold=settings.enforced_identity,
        ))
        audit.raise_if_leaking()
        split.manifest["homology_audit"] = audit.to_dict()
    if answer_from == ALL_PAIRS:
        split.manifest["search_answered_by"] = answered_by(search_config)
    return hierarchy, split


def validate_identity_stratified_split(
    hierarchy: ClusterHierarchy,
    split: IdentityStratifiedSplit,
    *,
    linked_groups: Mapping[str, Sequence[str]] | None = None,
    expected_allocation_unit: str | None = None,
) -> None:
    """Raise unless every protein has one known split, no 0.9 cluster spans splits, and no near-duplicate is held out.

    Written to run on a split read back from disk as well as on a fresh one.
    """
    proteins = set(hierarchy.proteins)
    policy = split.manifest.get("allocation_policy", "protein_count_v1")
    if policy not in ("protein_count_v1", "cluster_count_v1"):
        raise ValueError(f"unknown allocation policy {policy!r}")
    allocation_unit = "cluster" if policy == "cluster_count_v1" else "protein"
    if split.manifest.get("allocation_unit", "protein") != allocation_unit:
        raise ValueError("allocation policy and unit disagree")
    if split.manifest.get("settings", {}).get("allocation_unit", "protein") != allocation_unit:
        raise ValueError("allocation policy and settings disagree")
    if split.manifest.get("settings", {}).get("unit_order", DEFAULT_UNIT_ORDER) not in UNIT_ORDERS:
        raise ValueError(f"the recorded unit order is not one of {UNIT_ORDERS}")
    if expected_allocation_unit is not None and expected_allocation_unit != allocation_unit:
        raise ValueError(f"expected {expected_allocation_unit!r} allocation, got {allocation_unit!r}")
    if set(split.protein_to_split) != proteins:
        missing, unexpected = sorted(proteins - set(split.protein_to_split)), sorted(set(split.protein_to_split) - proteins)
        raise ValueError(f"split does not cover the hierarchy: missing={missing[:5]}, unexpected={unexpected[:5]}")
    unknown = sorted({name for name in split.protein_to_split.values() if name not in STRATIFIED_SPLIT_NAMES})
    if unknown:
        raise ValueError(f"unknown split names: {unknown}")
    enforced = split.manifest.get("near_duplicate_identity", NEAR_DUPLICATE_IDENTITY)
    if enforced not in ENFORCEABLE_IDENTITIES or split.manifest.get("settings", {}).get("enforced_identity", enforced) != enforced:
        raise ValueError(f"the recorded enforced identity {enforced!r} is not an enforceable level of the settings")
    strata = strata_below(enforced)
    recorded_groups = _normalized_groups(split.manifest.get("linked_groups", {}), proteins)
    if linked_groups is not None and _normalized_groups(linked_groups, proteins) != recorded_groups:
        raise ValueError("recorded linked groups differ from the source groups supplied to the audit")
    for group, members in recorded_groups.items():
        names = {split.protein_to_split[protein] for protein in members}
        if len(names) != 1:
            raise ValueError(f"linked group {group!r} spans splits {sorted(names)}")
    for cluster, members in hierarchy.members[hierarchy.finest].items():
        names = {split.protein_to_split[protein] for protein in members}
        if len(names) > 1:
            raise ValueError(f"0.9 cluster {cluster!r} spans splits {sorted(names)}")
    if enforced in hierarchy.levels and enforced != hierarchy.finest:
        for cluster, members in hierarchy.members[enforced].items():
            names = {split.protein_to_split[protein] for protein in members}
            if len(names) > 1:
                raise ValueError(f"{enforced} cluster {cluster!r} spans splits {sorted(names)}")
    evaluation = {protein for protein, name in split.protein_to_split.items() if name != "train"}
    for field, recorded in (("identity_to_train", split.identity_to_train), ("held_out_stratum", split.held_out_stratum)):
        if set(recorded) != evaluation:
            raise ValueError(f"{field} must name exactly the evaluation proteins")
    near = sorted(protein for protein, identity in split.identity_to_train.items() if identity >= enforced)
    if near:
        raise ValueError(f"near-duplicates of train (identity {enforced} or more) are held out: {near[:5]}")
    counts = Counter(split.protein_to_split.values())
    recorded_counts = {name: record["proteins"] for name, record in split.manifest["splits"].items()}
    if recorded_counts != {name: counts.get(name, 0) for name in STRATIFIED_SPLIT_NAMES}:
        raise ValueError(f"manifest counts {recorded_counts} do not match the split {dict(counts)}")
    for value in split.identity_to_train.values():
        identity_stratum(value, enforced)
    if set(split.held_out_stratum.values()) - set(strata):
        raise ValueError("unknown held-out identity stratum")
    if split.manifest.get("allocation_policy") == "cluster_count_v1":
        cluster_counts = Counter(
            split.protein_to_split[members[0]] for members in hierarchy.members[hierarchy.finest].values()
        )
        actual = {name: cluster_counts[name] for name in STRATIFIED_SPLIT_NAMES}
        share = split.manifest.get("settings", {}).get("evaluation_share", DEFAULT_EVALUATION_SHARE)
        quotas = _cluster_quotas(len(hierarchy.members[hierarchy.finest]), share)
        recorded = {name: record.get("clusters") for name, record in split.manifest["splits"].items()}
        if actual != quotas or recorded != actual or split.manifest.get("requested_counts") != quotas:
            raise ValueError(f"cluster quotas disagree: expected={quotas}, actual={actual}, recorded={recorded}")
        if share < DEFAULT_EVALUATION_SHARE:
            small = {name: counts.get(name, 0) for name in EVALUATION_SPLITS if counts.get(name, 0) < MINIMUM_REDUCED_EVALUATION_PROTEINS}
            if small:
                raise ValueError(
                    f"an evaluation share of {share} must still leave every evaluation split at least "
                    f"{MINIMUM_REDUCED_EVALUATION_PROTEINS} sequences: {small}"
                )
        if split.manifest.get("allocation_unit") != "cluster" or split.manifest.get("cluster_identity") != hierarchy.finest:
            raise ValueError("cluster allocation metadata disagrees with the hierarchy")


def _normalized_groups(groups: Mapping[str, Sequence[str]], proteins: set[str]) -> dict[str, tuple[str, ...]]:
    """Validate source links before they affect cluster membership or allocation."""
    if not isinstance(groups, Mapping):
        raise TypeError("linked_groups must map source-group identifiers to protein identifiers")
    normalized: dict[str, tuple[str, ...]] = {}
    for group, members in groups.items():
        if not isinstance(group, str) or not group:
            raise ValueError("a linked group must have a nonempty string identifier")
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or not members:
            raise ValueError(f"linked group {group!r} must contain a nonempty sequence of protein identifiers")
        if any(not isinstance(protein, str) or protein not in proteins for protein in members):
            raise ValueError(f"linked group {group!r} contains a protein absent from the hierarchy")
        normalized[group] = tuple(sorted(set(members)))
    return dict(sorted(normalized.items()))


def _linked_components(
    finest_of: Mapping[str, str], groups: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    """The transitive source-link component of each finest-level homology cluster."""
    parent = {cluster: cluster for cluster in finest_of.values()}

    def root(cluster: str) -> str:
        while parent[cluster] != cluster:
            parent[cluster] = parent[parent[cluster]]
            cluster = parent[cluster]
        return cluster

    for members in groups.values():
        first = finest_of[members[0]]
        for protein in members[1:]:
            left, right = root(first), root(finest_of[protein])
            if left != right:
                parent[max(left, right)] = min(left, right)
    components: dict[str, list[str]] = {}
    for cluster in sorted(parent):
        components.setdefault(root(cluster), []).append(cluster)
    linked: dict[str, tuple[str, ...]] = {}
    for members in components.values():
        component = tuple(members)
        for cluster in component:
            linked[cluster] = component
    return linked


def _fill_order(strata: Sequence[str]) -> tuple[str, ...]:
    """Far first, then nearer; what the nearest stratum leaves unfilled returns to the farther ones."""
    return (*strata, *reversed(strata[:-1]))


# One cluster repair returned to train, as the manifest's `moves` records it.
SplitMove = TypedDict(
    "SplitMove",
    {
        "iteration": int, "cluster": str, "proteins": int, "from": str, "held_out_stratum": str, "protein": str,
        "identity": float, "reason": str,
    },
)


class RefilledCluster(TypedDict):
    """A cluster a refill held out in place of one repair returned to train."""

    cluster: str
    split: str
    held_out_stratum: str


class Refill(TypedDict):
    """The clusters one refill held out, after the repair round it follows."""

    after_iteration: int
    held_out: list[RefilledCluster]


class _Assignment:
    """The mutable state of one split: which split each 0.9 cluster is in, and every running count."""

    def __init__(
        self, hierarchy: ClusterHierarchy, labels: Mapping[str, object] | None,
        settings: StratifiedSettings, linked_groups: Mapping[str, Sequence[str]] | None,
        incremental: bool = True, matches: IdentityMatches | None = None, progress: SplitProgress | None = None,
        relatives: Relatives | None = None,
    ) -> None:
        self.hierarchy = hierarchy
        self.settings = settings
        self.incremental = incremental
        self.matches = matches
        self.progress = progress
        self.relatives = relatives
        # How the components allocation went, kept for the manifest; empty when the first allocation was the greedy draw.
        self.component_record: dict[str, object] = {}
        self.components_placed = False
        self.levels = hierarchy.levels
        self.enforced = settings.enforced_identity
        self.strata = strata_below(self.enforced)
        self.fill_order = _fill_order(self.strata)
        finest = hierarchy.members[hierarchy.finest]
        self.total = sum(len(members) for members in finest.values())
        self.count_clusters = settings.allocation_unit == "cluster"
        largest = max(len(members) for members in finest.values())
        if not self.count_clusters and largest > settings.dominant_cluster_share * self.total:
            raise StratificationInfeasible(
                f"one 0.9 cluster holds {largest} of {self.total} proteins; declare another scheme, such as "
                "positional folds or the published split"
            )
        if self.count_clusters:
            self.targets = _cluster_quotas(len(finest), settings.evaluation_share)
            if any(size == 0 for size in self.targets.values()):
                raise StratificationInfeasible(f"{len(finest)} clusters cannot fill four splits: quotas={self.targets}")
            self.n = min(self.targets[split] for split in EVALUATION_SPLITS)
        else:
            self.n = evaluation_split_size(self.total, settings)
            self.targets = {split: self.n for split in EVALUATION_SPLITS}
            self.targets["train"] = self.total - 3 * self.n
        if not self.count_clusters and self.n < len(self.strata):
            raise StratificationInfeasible(f"{self.total} proteins leave evaluation splits of {self.n}; declare another scheme")
        self.quota = stratum_quotas(self.n, self.strata)
        self.quotas = {split: stratum_quotas(self.targets[split], self.strata) for split in EVALUATION_SPLITS}
        self.max_unit = max(1, math.floor(settings.max_unit_share * self.n))

        self.lineage = {cluster: hierarchy.lineage(cluster) for cluster in finest}
        self.finest_of = {protein: cluster for cluster, members in finest.items() for protein in members}
        self.linked_groups = _normalized_groups({} if linked_groups is None else linked_groups, set(self.finest_of))
        self.linked = _linked_components(self.finest_of, self.linked_groups)
        self.finest_under: dict[float, dict[str, list[str]]] = {level: {} for level in self.levels}
        for cluster, chain in self.lineage.items():
            for level, ancestor in zip(self.levels, chain, strict=True):
                self.finest_under[level].setdefault(ancestor, []).append(cluster)
        self.size = {
            level: {
                cluster: len(self.finest_under[level][cluster]) if self.count_clusters else len(members)
                for cluster, members in hierarchy.members[level].items()
            }
            for level in self.levels
        }
        self.train = {level: dict(sizes) for level, sizes in self.size.items()}
        self.linked_size: dict[str, int] = {}
        for component in self.linked.values():
            representative = component[0]
            if representative not in self.linked_size:
                self.linked_size[representative] = sum(self.size[hierarchy.finest][cluster] for cluster in component)

        if settings.balance_labels and not labels:
            raise ValueError("balance_labels needs the labels of the proteins")
        if settings.train_label_floor and not labels:
            raise ValueError("train_label_floor needs the labels of the proteins")
        self.labels = None if labels is None else {protein: _label_tuple(label) for protein, label in labels.items()}
        self.label_counts = {
            cluster: Counter(label for protein in members for label in self.labels[protein]) if self.labels else Counter()
            for cluster, members in finest.items()
        }
        shares = Counter(label for held in self.labels.values() for label in held) if self.labels else Counter()
        # A label's expected count in one evaluation split: its count in the dataset times the share of units the split takes.
        units = len(finest) if self.count_clusters else self.total
        self.label_target = {label: count / units * self.n for label, count in shares.items()}
        # Proteins carrying each label in the dataset and in all evaluation splits together, for `train_label_floor`, and the units
        # the floor refused at least once.
        self.label_total = shares
        self.held_total: Counter[str] = Counter()
        self.floor_refused: set[tuple[float, str]] = set()

        self.split_of = dict.fromkeys(finest, "train")
        self.unit_of: dict[str, tuple[str, float, str]] = {}  # finest cluster to (stratum, level, unit)
        self.held = {split: dict.fromkeys(self.strata, 0) for split in EVALUATION_SPLITS}
        self.held_labels: dict[str, Counter[str]] = {split: Counter() for split in EVALUATION_SPLITS}
        self.shortfall = {stratum: dict.fromkeys(EVALUATION_SPLITS, 0) for stratum in self.strata}
        self.unfilled = dict.fromkeys(EVALUATION_SPLITS, 0)
        # Clusters repair has returned to train: the cluster policy never holds one out again.
        self.pinned: set[str] = set()
        self.moves: list[SplitMove] = []
        self.refills: list[Refill] = []
        self.iteration = 0
        # The split of each cluster when repair last measured, so the next round searches only what changed.
        self.measured: dict[str, str] | None = None
        # Each held-out protein's identity to train from the last measurement of the whole assignment;
        # None once a round that measured only its changes has followed.
        self.identity: dict[str, float] | None = None
        # The held-out proteins at the enforced identity or more to train, each with its identity.
        self.forward: dict[str, float] = {}
        # Per directed pair of splits not ending in train, the queries at the enforced identity or more to the reference split.
        self.cross: dict[tuple[str, str], dict[str, float]] = {}

    def fill(self) -> None:
        """Fill each stratum in `self.fill_order`, passing what one leaves unfilled to the next.

        Run again after repair, it holds out other clusters for what repair returned to train. Under `label_allocation="components"`
        the first run deals out whole components of related sequences instead (`_fill_components`) and the later ones fill as usual.
        """
        self.shortfall = {stratum: dict.fromkeys(EVALUATION_SPLITS, 0) for stratum in self.strata}
        if self.settings.label_allocation == "components" and not self.components_placed:
            self.components_placed = True
            self.unfilled = dict.fromkeys(EVALUATION_SPLITS, 0)
            self._fill_components()
            return
        visited: set[str] = set()
        carry = dict.fromkeys(EVALUATION_SPLITS, 0)
        for stratum in self.fill_order:
            base = {
                split: self.held[split][stratum] if stratum in visited else self.quotas[split][stratum]
                for split in EVALUATION_SPLITS
            }
            target = {split: base[split] + carry[split] for split in EVALUATION_SPLITS}
            for split in EVALUATION_SPLITS:
                self.shortfall[stratum][split] += carry[split]
            self._fill_stratum(stratum, target)
            carry = {split: max(0, target[split] - self.held[split][stratum]) for split in EVALUATION_SPLITS}
            visited.add(stratum)
        self.unfilled = carry

    def _fill_components(self) -> None:
        """Deal out the finest clusters in whole components of related sequences, stratifying their labels (`component_allocation`).

        A component joins the clusters of one enforced-level cluster, the clusters of a linked source group, and the clusters of the
        proteins `relatives` pairs at the enforced identity or more against a reference holding the share of the residues train
        does. A component over the unit cap stays in train. The pairs only a held-out reference counts are not joined, so repair
        can find a few and returns their clusters to train.
        """
        if self.relatives is None:
            raise ValueError("label_allocation 'components' needs `relatives`, which the all-pairs table gives")
        import numpy as np  # loaded only by a split that asks for components, as the pair table is
        import scipy.sparse as sp

        from .component_allocation import allocate_components, connected_groups

        clusters = sorted(self.split_of)
        index = {cluster: place for place, cluster in enumerate(clusters)}
        members = self.hierarchy.members[self.hierarchy.finest]
        train_share = 1.0 - 3.0 * self.settings.evaluation_share
        names, query_rows, target_rows = self.relatives(self.enforced, train_share)
        owner = np.array([index[self.finest_of[name]] for name in names])  # (n,) the cluster of each protein, in the table's order
        left, right = [owner[query_rows]], [owner[target_rows]]  # (m,) each pair's two clusters
        groups = [tuple(group) for group in self.finest_under[self.enforced].values()]
        groups += sorted({tuple(group) for group in self.linked.values()})
        for group in groups:
            if len(group) > 1:
                left.append(np.full(len(group) - 1, index[group[0]]))
                right.append(np.array([index[cluster] for cluster in group[1:]]))
        component_of = connected_groups(len(clusters), np.concatenate(left), np.concatenate(right))  # (c,)
        count = int(component_of.max()) + 1  # K
        weights = np.bincount(component_of, minlength=count)  # (K,) finest clusters of each component
        rows = np.bincount(component_of, weights=[len(members[cluster]) for cluster in clusters], minlength=count).astype(np.int64)  # (K,)
        label_index = {label: place for place, label in enumerate(sorted(self.label_target))}
        entries = [(index[cluster], label_index[label], number) for cluster in clusters for label, number in self.label_counts[cluster].items()]
        carried = sp.csr_matrix(
            ([number for _, _, number in entries], ([place for place, _, _ in entries], [label for _, label, _ in entries])),
            shape=(len(clusters), len(label_index)), dtype=np.float64,
        )  # (c, L) rows of each cluster that carry each label
        membership = sp.csr_matrix((np.ones(len(clusters)), (component_of, np.arange(len(clusters)))), shape=(count, len(clusters)))  # (K, c)
        placed, record = allocate_components(
            (membership @ carried).tocsr(), weights, rows, [self.targets[name] for name in STRATIFIED_SPLIT_NAMES], weights > self.max_unit,
            seed=self.settings.seed,
        )  # (K,)
        self.component_record = {**record, "reference_share_of_residues": train_share, "unit_cap_clusters": self.max_unit}
        stratum = self.strata[0]
        level = UNIT_LEVEL[stratum]
        by_unit: dict[tuple[str, str], list[str]] = {}
        for cluster in clusters:
            split = STRATIFIED_SPLIT_NAMES[int(placed[component_of[index[cluster]]])]
            if split != "train":
                by_unit.setdefault((split, self.lineage[cluster][self.levels.index(level)]), []).append(cluster)
        for (split, unit), held in sorted(by_unit.items()):
            self._hold_out(level, unit, split, stratum, sum((self.label_counts[cluster] for cluster in held), Counter()), tuple(held))
        self.tell("components", components=count, largest_component_clusters=int(weights.max()), forced_to_train=int((weights > self.max_unit).sum()))

    def tell(self, name: str, **facts: object) -> None:
        """Tell `progress` of a step, with the quota of each evaluation split and what it holds now, in the unit of the quota."""
        if self.progress is not None:
            self.progress(
                name, **facts, requested={split: self.targets[split] for split in EVALUATION_SPLITS},
                realized={split: sum(self.held[split].values()) for split in EVALUATION_SPLITS},
            )

    def _fill_stratum(self, stratum: str, target: Mapping[str, int]) -> None:
        level = UNIT_LEVEL[stratum]
        seed = self.settings.seed
        largest_first = sorted(
            self.size[level], key=lambda unit: (-self.size[level][unit], _seed_rank(seed, f"order:{stratum}:{unit}"), unit)
        )
        seeded = self.settings.unit_order == "seeded"
        units = sorted(self.size[level], key=lambda unit: (_seed_rank(seed, f"order:{stratum}:{unit}"), unit)) if seeded else largest_first
        self._draw(stratum, target, units, only_if_better=True)
        # A seeded order, or a unit that labels held back, can leave room that only another unit fits, and the cluster policy has
        # to reach its exact counts: place what still fits, largest first, whatever it does to the labels.
        if self.count_clusters and (seeded or self.settings.balance_labels) and any(
            self.held[split][stratum] < target[split] for split in EVALUATION_SPLITS
        ):
            self._draw(stratum, target, largest_first, only_if_better=False)

    def _draw(self, stratum: str, target: Mapping[str, int], units: Sequence[str], *, only_if_better: bool) -> None:
        """Offer `units` in order to the evaluation splits, holding each out where it moves its split closest to its targets.

        `only_if_better` holds a unit out only when that reduces the squared distance from the targets, which is how the first
        pass decides. The top-up pass holds out any unit that fits.
        """
        level = UNIT_LEVEL[stratum]
        seed = self.settings.seed
        for unit in units:
            if all(self.held[split][stratum] >= target[split] for split in EVALUATION_SPLITS):
                return
            clusters = self._eligible_clusters(level, unit)
            if not clusters:
                continue
            size = sum(self.size[self.hierarchy.finest][cluster] for cluster in clusters)
            label_counts = sum((self.label_counts[cluster] for cluster in clusters), Counter())
            if self._breaks_floor(label_counts):
                self.floor_refused.add((level, unit))
                continue
            candidates = [
                split for split in EVALUATION_SPLITS
                if not self.count_clusters or (
                    self.held[split][stratum] + size <= target[split]
                    and sum(self.held[split].values()) + size <= self.targets[split]
                )
            ]
            if not candidates:
                continue
            change, _, _, split = min(
                (
                    self._change(split, stratum, size, label_counts, target[split]),
                    _seed_rank(seed, f"assign:{stratum}:{unit}:{split}"),
                    index,
                    split,
                )
                for index, split in enumerate(candidates)
            )
            if not only_if_better or change < -1.0e-12:
                self._hold_out(level, unit, split, stratum, label_counts, clusters)

    def _eligible_clusters(self, level: float, unit: str) -> tuple[str, ...]:
        """Expand source links, respect the size cap, and retain half of every affected parent."""
        if self.size[level][unit] > self.max_unit:
            return ()
        clusters: set[str] = set()
        size = 0
        # Source links and enforced-level parents pull in more clusters: a held-out unit holds every cluster of each enforced-level parent it touches.
        pending = list(self.finest_under[level][unit])
        while pending:
            cluster = pending.pop(0)
            if cluster in clusters:
                continue
            component = self.linked[cluster]
            if size + self.linked_size[component[0]] > self.max_unit:
                return ()
            for linked in component:
                if self.split_of[linked] != "train" or linked in self.pinned:
                    return ()
                size += self.size[self.hierarchy.finest][linked]
                if size > self.max_unit:
                    return ()
                clusters.add(linked)
                pending.extend(self._enforced_relatives(linked))
        if level == self.levels[-1]:
            return tuple(sorted(clusters))
        parent_index = self.levels.index(level) + 1
        parent_level = self.levels[parent_index]
        removed: Counter[str] = Counter()
        for cluster in clusters:
            removed[self.lineage[cluster][parent_index]] += self.size[self.hierarchy.finest][cluster]
        if any(2 * (self.train[parent_level][parent] - count) < self.size[parent_level][parent] for parent, count in removed.items()):
            return ()
        return tuple(sorted(clusters))

    def _breaks_floor(self, label_counts: Counter[str]) -> bool:
        """Whether holding out these label counts would leave a label the dataset holds at `train_label_floor` below it in train."""
        floor = self.settings.train_label_floor
        return bool(floor) and any(
            self.label_total[label] >= floor and self.label_total[label] - self.held_total[label] - count < floor
            for label, count in label_counts.items()
        )

    def _change(self, split: str, stratum: str, size: int, label_counts: Counter[str], target: int) -> float:
        """How much holding out `size` proteins in `split` changes its squared relative distance from its targets."""
        change = _squared_change(self.held[split][stratum], size, target)
        if self.count_clusters and not self.settings.balance_labels:
            # Exact cluster counts take precedence over protein-level label balance, and the historical cluster policy stops there.
            # Label support is still reported and must pass the downstream data audit.
            return change
        for label, count in label_counts.items():
            change += _squared_change(self.held_labels[split][label], count, self.label_target[label])
        return change

    def _hold_out(
        self, level: float, unit: str, split: str, stratum: str,
        label_counts: Counter[str], clusters: Sequence[str],
    ) -> None:
        for cluster in clusters:
            self.split_of[cluster] = split
            self.unit_of[cluster] = (stratum, level, unit)
            self._move_train_counts(cluster, -self.size[self.levels[0]][cluster])
        self.held[split][stratum] += sum(self.size[self.hierarchy.finest][cluster] for cluster in clusters)
        self.held_labels[split] += label_counts
        self.held_total += label_counts

    def _move_train_counts(self, cluster: str, change: int) -> None:
        for level, ancestor in zip(self.levels, self.lineage[cluster], strict=True):
            self.train[level][ancestor] += change

    def exact(self) -> bool:
        """Whether the last measurement covered the whole assignment as it stands."""
        return self.identity is not None and self.measured == self.split_of

    def repair(self, sequences: Mapping[str, str], search: IdentitySearch, *, whole: bool = False) -> None:
        """Return held-out 0.9 clusters to train while any is a near-duplicate.

        A held-out protein at 0.9 identity or more to train sends its cluster back. Under the
        cluster policy, so does a near-duplicate in another split, as `_leaks` finds them. The
        first round measures the whole assignment; each later round measures only what the last
        one moved (`_measure_changes`), which finds the flags a whole measurement finds when the
        search's identity for a pair does not depend on the other references. `whole` measures
        everything in every round, as `incremental=False` does for the whole split.
        """
        whole = whole or not self.incremental
        while True:
            if all(split == "train" for split in self.split_of.values()):
                self.forward, self.cross, self.identity, self.measured = {}, {}, {}, dict(self.split_of)
                return
            if not self.exact():
                if whole or self.measured is None:
                    self._measure_whole(sequences, search)
                else:
                    self._measure_changes(sequences, search)
            direct = self._flagged(sequences, search)
            if not direct:
                return
            self._return_to_train(direct)

    def _measure_whole(self, sequences: Mapping[str, str], search: IdentitySearch) -> None:
        """Search every held-out protein against train and, under the cluster policy, every pair of splits as the audit does."""
        members = self.hierarchy.members[self.hierarchy.finest]
        held = {protein: sequences[protein] for cluster, split in self.split_of.items() if split != "train" for protein in members[cluster]}
        train = {protein: sequences[protein] for cluster, split in self.split_of.items() if split == "train" for protein in members[cluster]}
        identity = _measure(search, held, train, "held-out protein")
        self.forward = {protein: value for protein, value in identity.items() if value >= self.enforced}
        self.cross = {}
        if self.count_clusters:
            clusters = {name: sorted(cluster for cluster, split in self.split_of.items() if split == name) for name in STRATIFIED_SPLIT_NAMES}
            by_split = {
                name: {protein: sequences[protein] for protein in sorted(protein for cluster in names for protein in members[cluster])}
                for name, names in clusters.items() if names
            }
            if len(by_split) > 1:
                audit = check_cross_split_homology(
                    by_split, search, search_parameters={}, identity_threshold=self.enforced,
                )
                # Directions into train duplicate the forward measurement.
                self.cross = {
                    (comparison.query_split, comparison.reference_split): dict(comparison.near_duplicates)
                    for comparison in audit.comparisons if comparison.reference_split != "train"
                }
        self.identity, self.measured = identity, dict(self.split_of)

    def _measure_changes(self, sequences: Mapping[str, str], search: IdentitySearch) -> None:
        """Update the flags for the clusters whose split changed since the last measurement, searching only those.

        Under repair, clusters leave evaluation splits and join train; under refill, the reverse.
        Each directed pair of splits is advanced by `_advance` with the proteins that joined or
        left its query split and its reference split.
        """
        members = self.hierarchy.members[self.hierarchy.finest]
        now: dict[str, list[str]] = {name: [] for name in STRATIFIED_SPLIT_NAMES}
        gained: dict[str, list[str]] = {name: [] for name in STRATIFIED_SPLIT_NAMES}
        lost: dict[str, list[str]] = {name: [] for name in STRATIFIED_SPLIT_NAMES}
        for cluster, split in self.split_of.items():
            now[split].extend(members[cluster])
            before = self.measured[cluster]
            if before != split:
                gained[split].extend(members[cluster])
                lost[before].extend(members[cluster])
        for table in (now, gained, lost):
            for proteins in table.values():
                proteins.sort()
        held = sorted(protein for name in EVALUATION_SPLITS for protein in now[name])
        self.forward = self._advance(
            self.forward, held, lost["train"], now["train"], gained["train"], bool(lost["train"]),
            sequences, search, "held-out protein",
        )
        if self.count_clusters:
            names = [name for name in sorted(STRATIFIED_SPLIT_NAMES) if now[name]]
            self.cross = {
                (query, reference): self._advance(
                    self.cross.get((query, reference), {}), now[query], gained[query], now[reference],
                    gained[reference], bool(lost[reference]), sequences, search, "audited protein",
                )
                for query in names for reference in names if query != reference and reference != "train"
            }
        self.identity, self.measured = None, dict(self.split_of)

    def _advance(
        self, flagged: Mapping[str, float], queries: Sequence[str], new_queries: Sequence[str],
        references: Sequence[str], new_references: Sequence[str], references_lost: bool,
        sequences: Mapping[str, str], search: IdentitySearch, subject: str,
    ) -> dict[str, float]:
        """The queries at the enforced identity or more to the references after a change, each with its highest identity.

        `flagged` is the answer before the change; `queries` and `references` are the proteins now,
        `new_queries` and `new_references` those that joined, and `references_lost` says whether any
        reference left. A new query is searched against every reference. So is a flagged query when a
        reference left, because leaving can only lower an identity, and only a flagged query could
        fall below the enforced identity. Every other query is searched against the new references alone, because
        joining can only raise an identity, so an identity at or above it can appear only there, and
        the higher of the two is the highest over all the references.
        """
        current, joined = set(queries), set(new_queries)
        kept = {protein: value for protein, value in flagged.items() if protein in current and protein not in joined}
        again = joined | set(kept) if references_lost else joined
        advanced = {protein: value for protein, value in kept.items() if protein not in again}
        if again and references:
            found = _measure(
                search, {protein: sequences[protein] for protein in sorted(again)},
                {protein: sequences[protein] for protein in references}, subject,
            )
            advanced.update({protein: value for protein, value in found.items() if value >= self.enforced})
        rest = [protein for protein in queries if protein not in again]
        if rest and new_references:
            found = _measure(
                search, {protein: sequences[protein] for protein in rest},
                {protein: sequences[protein] for protein in new_references}, subject,
            )
            for protein, value in found.items():
                if value >= self.enforced:
                    advanced[protein] = max(advanced.get(protein, 0.0), value)
        return advanced

    def _flagged(self, sequences: Mapping[str, str], search: IdentitySearch) -> dict[str, tuple[float, str, str]]:
        """The held-out clusters that a near-duplicate sends back to train, each with its strongest (identity, protein, reason)."""
        direct: dict[str, tuple[float, str, str]] = {}
        for protein, value in self.forward.items():
            cluster = self.finest_of[protein]
            direct[cluster] = max(direct.get(cluster, (0.0, "", "")), (value, protein, "near_duplicate"))
        if self.count_clusters:
            for cluster, evidence in self._leaks(sequences, search).items():
                direct[cluster] = max(direct.get(cluster, (0.0, "", "")), evidence)
        return direct

    def _return_to_train(self, direct: Mapping[str, tuple[float, str, str]]) -> None:
        """Move the flagged clusters and the clusters linked to them back to train, pinned, and record each move."""
        members = self.hierarchy.members[self.hierarchy.finest]
        affected = dict(direct)
        reasons: dict[str, str] = {}
        pending = sorted(direct.items())
        while pending:
            cluster, evidence = pending.pop(0)
            related = [(linked, "linked_source_group") for linked in self.linked[cluster]]
            related += [(relative, "enforced_level_cluster") for relative in self._enforced_relatives(cluster)]
            for other, reason in related:
                if other in affected:
                    affected[other] = max(affected[other], evidence)
                elif self.split_of[other] != "train":
                    affected[other] = evidence
                    reasons[other] = reason
                    pending.append((other, evidence))
        self.iteration += 1
        for cluster in sorted(affected):
            split = self.split_of[cluster]
            stratum = self.unit_of[cluster][0]
            size = self.size[self.hierarchy.finest][cluster]
            self.split_of[cluster] = "train"
            self.pinned.add(cluster)
            self.held[split][stratum] -= size
            self.held_labels[split] -= self.label_counts[cluster]
            self.held_total -= self.label_counts[cluster]
            self._move_train_counts(cluster, size)
            value, protein, _ = affected[cluster]
            self.moves.append({
                "iteration": self.iteration, "cluster": cluster, "proteins": len(members[cluster]), "from": split,
                "held_out_stratum": stratum, "protein": protein, "identity": value,
                "reason": direct[cluster][2] if cluster in direct else reasons[cluster],
            })
        self.tell(
            "repair", iteration=self.iteration, flagged=len(direct), returned=len(affected), pinned=len(self.pinned),
            reasons=dict(sorted(Counter(move["reason"] for move in self.moves[-len(affected):]).items())),
        )

    def _enforced_relatives(self, cluster: str) -> tuple[str, ...]:
        """The other finest-level clusters inside the same enforced-level cluster, none when 0.9 is enforced."""
        if self.enforced == self.hierarchy.finest:
            return ()
        ancestor = self.lineage[cluster][self.levels.index(self.enforced)]
        return tuple(relative for relative in self.finest_under[self.enforced][ancestor] if relative != cluster)

    def _leaks(self, sequences: Mapping[str, str], search: IdentitySearch) -> dict[str, tuple[float, str, str]]:
        """The held-out clusters that a protein of another split is a near-duplicate of, each with its evidence.

        The flags are those of `check_cross_split_homology`, the audit that ends every
        cluster-policy build, so a split that passes here passes it. Directions into train are the
        forward measurement. Any other flagged query is located among the clusters of its
        reference split. A train query is a fragment of, or covered by, a held-out protein, and
        that cluster goes back. An evaluation query and its match sit in two evaluation splits, and
        both clusters go back: the alignment would flag either one against the other from train.
        """
        clusters = {name: sorted(cluster for cluster, split in self.split_of.items() if split == name) for name in STRATIFIED_SPLIT_NAMES}
        leaks: dict[str, tuple[float, str, str]] = {}
        for query_split, reference_split in sorted(self.cross):
            flagged = self.cross[query_split, reference_split]
            if not flagged:
                continue
            reason = "reverse_near_duplicate" if query_split == "train" else "evaluation_near_duplicate"
            located = self._locate(sequences, search, sorted(flagged), clusters[reference_split])
            for protein, targets in located.items():
                evidence = (flagged[protein], protein, reason)
                for target in targets:
                    leaks[target] = max(leaks.get(target, (0.0, "", "")), evidence)
                if query_split != "train":
                    own = self.finest_of[protein]
                    leaks[own] = max(leaks.get(own, (0.0, "", "")), evidence)
        return leaks

    def _locate(
        self, sequences: Mapping[str, str], search: IdentitySearch, flagged: Sequence[str], clusters: Sequence[str],
    ) -> dict[str, tuple[str, ...]]:
        """The clusters of `clusters` that each flagged protein is a near-duplicate of: every one `matches` names, else one found by halving.

        One search of the flagged proteins against the whole reference split names every match. A
        flag it does not reproduce goes to `_halve`, because a hit's e-value follows the size of
        the reference set, and a flag a smaller set raised can fall short against the whole one.
        """
        members = self.hierarchy.members[self.hierarchy.finest]
        located: dict[str, tuple[str, ...]] = {}
        unplaced = tuple(flagged)
        if self.matches is not None:
            references = {protein: sequences[protein] for cluster in clusters for protein in sorted(members[cluster])}
            found = self.matches({protein: sequences[protein] for protein in flagged}, references, self.enforced)
            if set(found) != set(flagged) or any(not set(targets) <= references.keys() for targets in found.values()):
                raise ValueError("the match search must name references of the split for every flagged protein, and only those")
            for protein in flagged:
                if found[protein]:
                    located[protein] = tuple(sorted({self.finest_of[target] for target in found[protein]}))
            unplaced = tuple(protein for protein in flagged if protein not in located)
        if unplaced:
            located.update({protein: (cluster,) for protein, cluster in self._halve(sequences, search, unplaced, clusters).items()})
        return located

    def _halve(
        self, sequences: Mapping[str, str], search: IdentitySearch, flagged: Sequence[str], clusters: Sequence[str],
    ) -> dict[str, str]:
        """One cluster of `clusters` that each flagged protein is a near-duplicate of, found by halving them.

        `search` gives only each query's highest identity over the references it is handed, so it
        cannot name the cluster it matched. Every flagged protein matches one in `clusters`: the
        proteins a search of the first half flags go on with that half, and the rest match in the
        other half. A smaller reference set only lowers an e-value, so a flag never disappears.
        A protein with several matches gets one per call; the next round finds the others.
        """
        members = self.hierarchy.members[self.hierarchy.finest]
        located: dict[str, str] = {}
        pending = [(tuple(flagged), tuple(clusters))]
        while pending:
            active, group = pending.pop()
            if len(group) == 1:
                located.update(dict.fromkeys(active, group[0]))
                continue
            first, second = group[:len(group) // 2], group[len(group) // 2:]
            references = {protein: sequences[protein] for cluster in first for protein in sorted(members[cluster])}
            identity = _measure(search, {protein: sequences[protein] for protein in active}, references, "flagged protein")
            near = tuple(protein for protein in active if identity[protein] >= self.enforced)
            rest = tuple(protein for protein in active if identity[protein] < self.enforced)
            if near:
                pending.append((near, first))
            if rest:
                pending.append((rest, second))
        return located

    def refill(self) -> bool:
        """Hold out other clusters for the quota repair freed; whether any cluster was newly held out.

        Only the cluster policy refills, because only its quotas are exact: protein counts accept
        the shortfall. A pinned cluster is never held out again, so every round pins another
        cluster or ends, and a refilled cluster still has to survive the next repair.
        """
        clusters = Counter(self.split_of.values())
        if not self.count_clusters or all(clusters[split] == self.targets[split] for split in EVALUATION_SPLITS):
            return False
        before = dict(self.split_of)
        self.fill()
        held = {cluster: split for cluster, split in sorted(self.split_of.items()) if split != before[cluster]}
        self.tell("refill", iteration=self.iteration, held_out=len(held))
        if held:
            self.refills.append({
                "after_iteration": self.iteration,
                "held_out": [
                    {"cluster": cluster, "split": split, "held_out_stratum": self.unit_of[cluster][0]}
                    for cluster, split in held.items()
                ],
            })
        return bool(held)

    def finished_split(
        self, identity: Mapping[str, float], moves: list[SplitMove], search_parameters: Mapping[str, object] | None
    ) -> IdentityStratifiedSplit:
        """The split as it stands, with its manifest; `StratificationInfeasible` when it misses its cluster quotas or filled share."""
        members = self.hierarchy.members[self.hierarchy.finest]
        protein_to_split = {protein: self.split_of[cluster] for cluster, proteins in members.items() for protein in proteins}
        held_out_stratum = {
            protein: self.unit_of[cluster][0]
            for cluster, proteins in members.items()
            if self.split_of[cluster] != "train"
            for protein in proteins
        }
        counts = Counter(protein_to_split.values())
        cluster_counts = Counter(self.split_of.values())
        if self.count_clusters and any(cluster_counts[split] != self.targets[split] for split in STRATIFIED_SPLIT_NAMES):
            raise StratificationInfeasible(
                f"the allocator did not reach exact cluster quotas: requested={self.targets}, "
                f"realized_clusters={dict(cluster_counts)}, realized_proteins={dict(counts)}; "
                "do not substitute protein-count quotas; this greedy result is not proof that every allocation is infeasible"
            )
        short = {
            split: counts.get(split, 0) for split in EVALUATION_SPLITS
            if not self.count_clusters and counts.get(split, 0) < self.settings.min_filled_share * self.n
        }
        if short:
            raise StratificationInfeasible(
                f"evaluation splits {short} fall below {self.settings.min_filled_share} of n={self.n}; declare another scheme"
            )

        splits: dict[str, dict[str, object]] = {"train": {"proteins": counts.get("train", 0)}}
        agreement: dict[str, float] = {}
        for split in EVALUATION_SPLITS:
            in_split = [protein for protein, name in protein_to_split.items() if name == split]
            measured = {protein: identity_stratum(identity[protein], self.enforced) for protein in in_split}
            units: dict[str, set[tuple[float, str]]] = {stratum: set() for stratum in self.strata}
            for cluster, name in self.split_of.items():
                if name == split:
                    stratum, level, unit = self.unit_of[cluster]
                    units[stratum].add((level, unit))
            strata: dict[str, object] = {}
            for stratum in self.strata:
                held_here = [protein for protein in in_split if held_out_stratum[protein] == stratum]
                measured_here = [protein for protein in in_split if measured[protein] == stratum]
                strata[stratum] = {
                    "quota": self.quotas[split][stratum],
                    "held_out": {
                        "proteins": len(held_here), "units": len(units[stratum]),
                        "clusters": len({self.finest_of[protein] for protein in held_here}),
                    },
                    "measured": {"proteins": len(measured_here), "label_shares": self._shares(measured_here)},
                }
            splits[split] = {"proteins": len(in_split), "label_shares": self._shares(in_split), "strata": strata}
            agreement[split] = sum(measured[protein] == held_out_stratum[protein] for protein in in_split) / len(in_split)
        calibration_counts = {stratum: splits["calibration"]["strata"][stratum]["measured"]["proteins"] for stratum in self.strata}
        for split in STRATIFIED_SPLIT_NAMES:
            splits[split]["clusters"] = cluster_counts[split]

        origin = self.hierarchy.origin
        manifest: dict[str, object] = {
            "method": "identity_stratified",
            "allocation_policy": "cluster_count_v1" if self.count_clusters else "protein_count_v1",
            "allocation_unit": self.settings.allocation_unit,
            "cluster_identity": self.hierarchy.finest,
            "requested_counts": dict(self.targets),
            "rounding": "largest_remainder_split_order" if self.count_clusters else "legacy_minimum_cap",
            "linked_groups": {group: list(members) for group, members in self.linked_groups.items()},
            "linked_allocation_units": len(self.linked_size),
            "settings": recorded_settings(self.settings),
            "levels": list(self.levels),
            "near_duplicate_identity": self.enforced,
            "strata": list(self.strata),
            "mmseqs_version": origin[0].get("version") if origin else None,
            "clustering": [dict(record) for record in origin],
            "search": dict(search_parameters) if search_parameters is not None else None,
            "proteins": self.total,
            "evaluation_split_size": self.n,
            "quota_per_stratum": dict(self.quota),
            "quota_per_split_stratum": {split: dict(quotas) for split, quotas in self.quotas.items()},
            "label_shares": self._shares(list(protein_to_split)),
            "splits": splits,
            "agreement": agreement,
            "shortfall_passed": {stratum: dict(values) for stratum, values in self.shortfall.items()},
            "unfilled": dict(self.unfilled),
            "moves": moves,
            "calibration_groups": calibration_groups(calibration_counts, strata=self.strata),
        }
        if self.count_clusters:
            manifest["refills"] = self.refills
        if self.component_record:
            manifest["components"] = self.component_record
        if self.settings.train_label_floor:
            floor = self.settings.train_label_floor
            reachable = [label for label, count in self.label_total.items() if count >= floor]
            manifest["train_label_floor"] = {
                "floor": floor, "labels_reaching_it": len(reachable),
                "labels_below_it_in_train": sum(self.label_total[label] - self.held_total[label] < floor for label in reachable),
                "units_refused": len(self.floor_refused),
            }
        return IdentityStratifiedSplit(
            protein_to_split=dict(sorted(protein_to_split.items())),
            held_out_stratum=dict(sorted(held_out_stratum.items())),
            identity_to_train=dict(sorted(identity.items())),
            manifest=manifest,
        )

    def _shares(self, proteins: list[str]) -> dict[str, float]:
        if not self.labels or not proteins:
            return {}
        counts = Counter(label for protein in proteins for label in self.labels[protein])
        return {label: counts[label] / len(proteins) for label in sorted(counts)}


def _squared_change(current: float, add: float, target: float) -> float:
    """The change in squared distance from `target`, relative to it, when `add` joins `current`."""
    return ((current + add - target) ** 2 - (current - target) ** 2) / max(1.0, target)


def _measure(
    search: IdentitySearch, queries: Mapping[str, str], references: Mapping[str, str], subject: str,
) -> dict[str, float]:
    """`search` of the queries against the references: one identity in [0, 1] for each query, and only those."""
    measured = search(queries, references)
    if set(measured) != set(queries) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0
        for value in measured.values()
    ):
        raise ValueError(f"the search must give every {subject} one identity in [0, 1]")
    return {protein: float(value) for protein, value in measured.items()}
