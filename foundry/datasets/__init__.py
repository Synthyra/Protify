"""Protein datasets: homology clusters and search, disjoint and identity-stratified splits, restartable stages, and Hub publishing.

The publishing names load `huggingface_hub` on first use. A project that only clusters, splits,
or checks leakage, such as embedding_translation's independent split audit, runs without it.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .hierarchy import HIERARCHY_LEVELS, ClusterHierarchy, cluster_hierarchy, nest_clusterings
from .homology import (
    SearchConfig,
    SearchHit,
    matching_references,
    max_identity_to_reference,
    parse_search_hits,
    search_command,
    search_hits,
)
from .identity_survey import IdentitySurvey, LevelSurvey, SurveySettings, survey_identity_levels
from .label_balance import LabelBalance, SplitLabelBalance, measure_label_balance
from .leakage import (
    HomologyComparison,
    HomologyLeakageReport,
    LeakageError,
    LeakageReport,
    check_cross_split_homology,
    check_split_disjointness,
)
from .mmseqs import (
    Clustering,
    MMseqsConfig,
    cluster_command,
    cluster_sequences,
    createdb_command,
    createtsv_command,
    mmseqs_version,
    parse_memberships,
)
from .splits import SPLIT_NAMES, SplitAssignment, SplitRatios, assign_clusters_to_splits, validate_split_assignment
from .stage import run_stage
from .stratified import (
    ENFORCEABLE_IDENTITIES,
    EVALUATION_SPLITS,
    INTERACTION_DATASET_PREFIXES,
    INTERACTION_ENFORCED_IDENTITY_LIMIT,
    INTERACTION_TASKS,
    MEASURE_SEARCH,
    STRATA,
    STRATIFIED_SPLIT_NAMES,
    UNIT_ORDERS,
    IdentityStratifiedSplit,
    StratificationInfeasible,
    StratifiedSettings,
    build_identity_stratified_split,
    calibration_groups,
    enforced_identity_ceiling,
    evaluation_split_size,
    identity_stratum,
    quintile_labels,
    recorded_settings,
    require_interaction_identity,
    split_by_identity_strata,
    strata_below,
    stratum_quotas,
    validate_identity_stratified_split,
)


if TYPE_CHECKING:
    from .hub import IntegrityError, PublicationError, PublishedRevision, RevisionCheck, VerifiedFile, publish_dataset


HUB_NAMES = frozenset(
    {"IntegrityError", "PublicationError", "PublishedRevision", "RevisionCheck", "VerifiedFile", "publish_dataset"}
)


def __getattr__(name: str) -> object:
    if name in HUB_NAMES:
        return getattr(import_module(".hub", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ENFORCEABLE_IDENTITIES",
    "EVALUATION_SPLITS",
    "HIERARCHY_LEVELS",
    "INTERACTION_DATASET_PREFIXES",
    "INTERACTION_ENFORCED_IDENTITY_LIMIT",
    "INTERACTION_TASKS",
    "MEASURE_SEARCH",
    "SPLIT_NAMES",
    "STRATA",
    "STRATIFIED_SPLIT_NAMES",
    "UNIT_ORDERS",
    "ClusterHierarchy",
    "Clustering",
    "HomologyComparison",
    "HomologyLeakageReport",
    "IdentityStratifiedSplit",
    "IdentitySurvey",
    "IntegrityError",
    "LabelBalance",
    "LeakageError",
    "LeakageReport",
    "LevelSurvey",
    "MMseqsConfig",
    "PublicationError",
    "PublishedRevision",
    "RevisionCheck",
    "SearchConfig",
    "SearchHit",
    "SplitAssignment",
    "SplitLabelBalance",
    "SplitRatios",
    "StratificationInfeasible",
    "StratifiedSettings",
    "SurveySettings",
    "VerifiedFile",
    "assign_clusters_to_splits",
    "build_identity_stratified_split",
    "calibration_groups",
    "check_split_disjointness",
    "check_cross_split_homology",
    "cluster_command",
    "cluster_hierarchy",
    "cluster_sequences",
    "createdb_command",
    "createtsv_command",
    "enforced_identity_ceiling",
    "evaluation_split_size",
    "identity_stratum",
    "matching_references",
    "max_identity_to_reference",
    "measure_label_balance",
    "mmseqs_version",
    "nest_clusterings",
    "parse_memberships",
    "parse_search_hits",
    "publish_dataset",
    "quintile_labels",
    "recorded_settings",
    "require_interaction_identity",
    "run_stage",
    "search_command",
    "search_hits",
    "split_by_identity_strata",
    "strata_below",
    "stratum_quotas",
    "survey_identity_levels",
    "validate_identity_stratified_split",
    "validate_split_assignment",
]
