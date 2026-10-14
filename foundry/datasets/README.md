---
id: foundry.datasets
kind: library_module
status: active
aliases: [foundry.datasets]
---
# foundry.datasets

How a protein dataset is built and reaches the Hugging Face Hub: homology clusters and
searches, identity-stratified splits, stages that survive a crash, interaction tables cleaned of
incomplete rows and repeated pairs, splits checked for leakage on the rows that ship, and one
verified revision that pins the result.

## Paired training

`pair_sampling.ClusterPatchSampler` owns production Atlas's sampler for paired datasets.
Atlas and Dyad import this implementation. It draws an ordered taxonomy-pair stratum with
square-root incidence-count weighting, a cluster uniformly from either endpoint, then an
incident measured pair. Rows are distinct within each sampling patch; patch size defaults
to 64 and `stratum_exponent` to 0.5. Explicit historical exponents retain their seeded draws.
See the [paired sampling convention](../../docs/conventions/paired_sampling.md) for adapters,
small strata, input orientation and the separation from published dataset splits.

## Clusters and stages

| Name | Does |
|---|---|
| `cluster_sequences(sequences, directory, config)` | Clusters a mapping of identifier to sequence with MMseqs2 and returns `Clustering`: representative to members, member to representative, and an `origin` record of the version, commands, and input and output digests. The run lives in a stage named by a hash of the sequences, the MMseqs version, and every setting that changes the result, so equal inputs reuse it and changed inputs never do |
| `MMseqsConfig(min_sequence_identity, min_coverage, ...)` | One run's settings. Identity and coverage are required; the rest default to what annotation_vocabulary and latent_sep share. `expected_version` refuses any other MMseqs build |
| `createdb_command`, `cluster_command`, `createtsv_command` | The argument lists, for projects that run their own pipeline |
| `MMseqsConfig(linear=True)`, `cluster_hierarchy(..., linear_from=0.9)` | Runs `linclust` (linear time, identity 0.8 or more) for the finest levels of a set with thousands of close variants of one scaffold, where `cluster` never finishes; coarser levels keep the full search. Off by default, and a stage made without it keeps its name. See [the note](../../archive/memory/2026-09-29_mmseqs_cluster_on_dense_libraries.md) |
| `parse_memberships(lines, expected)` | `createtsv` output as clusters, refused if any expected protein is missing, repeated, or foreign |
| `run_stage(directory, outputs, build)` | Builds into a temporary sibling and renames it into place once every output exists. A finished stage is reused; a crashed one is refused rather than guessed about |

Promoted from annotation_vocabulary's `mmseqs.py` (commands, parser, version) and latent_sep's
`mmseqs.py` (commands, stages), which both now delegate here. Rewiring latent_sep fixed a defect
the survey found: its command hard-coded 0.30 identity and 0.50 coverage while its build
manifest recorded `BuildConfig`'s values; the configured values now reach MMseqs.

## Docker-run clustering

`docker_clustering.py` is the one copy of the clusterer that DatasetDev's builders, confounders,
Atlas, and synth each held (`data_clustering.py`, four copies). It runs MMseqs2 `easy-cluster` or
CD-HIT, on the machine when the binary is on the PATH and in Docker otherwise, and it is not a
second general clustering API: **new work uses `cluster_sequences` above**, which is staged,
version-pinned, checked against the input, and keyed by the content it clustered. This module
exists because the splits those builders published came from these exact commands, and a cluster
membership must not change behind a published split.

| Use | When |
|---|---|
| `foundry.datasets.mmseqs` (`cluster_sequences`, `cluster_hierarchy`) | Any new build, and any dataset that is rebuilt from scratch |
| `docker_clustering` (`ProteinSequenceClusterer`, `cluster_sequences`) | A builder whose published split must rebuild with the same clusters, until its clusters are compared with the staged ones and Logan agrees to move it |

| Name | Does |
|---|---|
| `ClusteringConfig(method="mmseqs2", cluster_percentage=0.5, coverage=0.8, n=3, memory_percentage=0.5, cache_suffix="", base_path="data", identifier="unknown", options=ClusterRunOptions())` | One run: `mmseqs2`, `cd-hit`, or `none` (every sequence alone), the identity, the coverage (`-c` and `-aL`), CD-HIT's word size, the share of memory, and where the FASTA and the reports go |
| `ProteinSequenceClusterer(config, runner=None, which=None)(sequences)` | A `defaultdict` of each cluster's key to its members. `runner` and `which` stand in for `subprocess.run` and `shutil.which`, so a test records the commands and writes the reports itself |
| `cluster_sequences(sequences, method=..., ..., options=None)` | The same from keyword settings |
| `cluster_sequences_mmseqs(id_to_seq, work_dir, identity, coverage, options=None)`, `cluster_sequences_cdhit(id_to_seq, work_dir, identity, n, coverage=0.0, min_length=11, options=None)` | A flat member to cluster key map from a work directory, which is its parent (the base path) and its last component (the identifier). CD-HIT drops sequences of 10 or fewer residues without a word, so shorter ones never reach it and become their own clusters |
| `ClusterRunOptions` | The choices the copies disagreed on, below |
| `StaleFastaError` | A cached FASTA of another size than the sequences. A `ValueError` and an `AssertionError`, because the Atlas copy raised the second |

Every path lies under the working directory, which Docker mounts. Results are cached by name, so a
finished report is read again whatever the sequences now are; with `check_cached_fasta` a FASTA of
another size is refused first. `psutil` sizes the run (all but four cores, `memory_percentage` of
memory). Torch is optional: it is asked only whether CUDA is visible, while `gpu` is left unset.

The copies were identical in `ProteinSequenceClusterer` except for four places, each a field of
`ClusterRunOptions`. The defaults reproduce DatasetDev, confounders, and synth, whose published
splits and caches came from them; Atlas sets every field.

| Field | Default (DatasetDev, confounders, synth) | Atlas passes |
|---|---|---|
| `gpu` | `None`: `--gpu 1` (and Docker `--gpus all`) when torch sees CUDA | `False`: never |
| `check_cached_fasta`, `stale_fasta_remedy` | `False`: a cached FASTA is reused whatever it holds | `True`, with the `--clear_work_dir / --clear_mmseqs_cache / --clear_cdhit_cache` advice |
| `cdhit_on_path` | `False`: CD-HIT always runs in Docker | `True`: a `cd-hit` on the PATH first |
| `cdhit_cluster_key` | `"cluster_number"`: a cluster's key is its number in the report, `"0"`, `"1"` | `"representative"`: its representative's identifier, the member marked `*` (and a cluster without one is an error) |
| `disable_mmseqs_shuffle` | `False` | `True`: `--shuffle 0`, which easy-cluster's default cascade needs under the default createdb mode |

Atlas's call is `ClusterRunOptions(gpu=False, check_cached_fasta=True, stale_fasta_remedy="re-run
build_up.py with --clear_work_dir / --clear_mmseqs_cache / --clear_cdhit_cache", cdhit_on_path=True,
cdhit_cluster_key="representative", disable_mmseqs_shuffle=True)`. synth's flat `cluster_sequences_cdhit`
keyed its clusters by number, so its callers got cluster numbers where Atlas's got representatives; both
survive as the two keys.

DatasetDev's `dataset_dev_utils.data_clustering` and confounders' `data.data_clustering` re-export
`ClusteringConfig`, `ProteinSequenceClusterer`, and `cluster_sequences` from here, so every builder import
is unchanged. `tests/tier1_unit/test_datasets_docker_clustering.py` pins each copy's command lines, files
written, and parsing against a fake runner. Before this module replaced the copies, a throwaway comparison
ran all four against one fake `subprocess.run` over 136 scenarios (MMseqs2 and CD-HIT, each tool on and off
the PATH, GPU visible and not, a failed first command, the flat functions) and found the same commands,
keyword arguments, printed text, files, and clusters.

**Cluster memberships are not yet compared with `foundry.datasets.mmseqs`.** It needs MMseqs2 `18-8cc5c` and
CD-HIT 4.8.1 on a Linux machine (or a running Docker daemon with the `ghcr.io/soedinglab/mmseqs2` and local
`cd-hit` images) and a small fixture, run through both with matching settings: `easy-cluster`'s defaults are a
cascaded run, so the staged side needs `single_step_clustering=False`, `cluster_reassign=False`, and
`sensitivity=7.5`, and its fixed `--alignment-mode 3` must be checked against easy-cluster's default. Until
the memberships agree, or Logan accepts a difference, no builder moves to `cluster_sequences` above.

## Homology search

| Name | Does |
|---|---|
| `max_identity_to_reference(queries, references, directory, config=SearchConfig())` | Each query's highest MMseqs2 identity (`fident`) to any reference among hits with an e-value at most `evalue_max`, and 0.0 for a query with none, in input order. Runs `easy-search` in a stage named by a hash of both sequence sets, the MMseqs version, and every setting that changes the hits, so a rerun reads the finished hits. A hit naming a sequence outside the search is refused |
| `SearchConfig(evalue_max=1e-3, sensitivity=7.5, min_coverage=0.0, ...)` | One search's settings. Sensitivity, the cap of 2,000 targets per query, and the e-value follow sae_xgboost_ppi; alignment mode 3 follows the clustering above, and `alignment_mode=None` leaves the flag to MMseqs, as sae_xgboost_ppi's search does. `expected_version` refuses any other MMseqs build |
| `search_hits(queries, references, directory, config=SearchConfig())` | The staged search itself: every hit MMseqs reports, self-hits included, for a consumer that needs more than the best identity. sae_xgboost_ppi's characterization uses it, one search since 2026-09-26 ([the decision](../../docs/decisions/foundry/datasets/2026-09-24_homology_search.md)) |
| `search_command`, `parse_search_hits(lines)` | The argument list, and its TSV output as `SearchHit` rows, refused at the first malformed row |
| `search_file(queries, references, directory, config)` | The staged search's TSV of hits, for a consumer that reads millions of them as a stream |
| `pair_table.PairTable.from_search(sequences, directory, config)` | One search of every sequence against every other, answering `max_identity` and `matching` for any subsets by lookup, with MMseqs2's e-value scaling by target residues applied to the stored e-values (hits kept to 25 times the limit, `DEFAULT_REACH`). Imports numpy on first use; 10 million hits parse in 12 s and a query over 79,000 references takes 0.03 s ([lean builds](../../docs/conventions/splits.md#lean-builds)) |

It bins test proteins by their identity to training, as sae_xgboost_ppi's `gp_analysis.py` did
with its own search and as Protify's uncertainty workflow does through this function. Coverage is
off by default, as sae_xgboost_ppi ran it; contact_esmc6b's split audits require half of both
sequences to align, `min_coverage=0.5` with `coverage_mode=0`. Neither project calls this yet:
sae_xgboost_ppi searched at e-value 10, filtered at 0.001 afterwards, and left the alignment mode
at MMseqs's default, and contact_esmc6b also filters at 0.30 identity, so moving either changes
numbers it has reported. That move waits for Logan's decision.

## Splits

| Name | Does |
|---|---|
| `assign_clusters_to_splits(clusters, ratios, seed, *, expected_accessions=None)` | `SplitAssignment`: each whole cluster, and so each protein, in `train`, `validation`, or `test`. Targets are counted in proteins. Ties break on a SHA-256 rank of the seed, so input order and global random state never matter |
| `validate_split_assignment(clusters, assignment)` | Raises unless every protein is assigned, no cluster spans splits, and the recorded counts hold; runs on an assignment read back from disk |
| `SplitRatios(train, validation, test)` | Fractions of proteins; they must sum to 1 |

`Clustering.clusters` is the input `assign_clusters_to_splits` takes.

## Identity-stratified splits

`build_identity_stratified_split` accepts an optional `checkpoint(name, operation)` callback.
It wraps clustering, assignment, and (for cluster allocation) the all-pairs split audit.
The callback must return the corresponding stage object, whether freshly computed or read
from its verified journal. The allocator and validators remain shared; omitting the callback
preserves ordinary execution. The production suite uses this to retain separate stage receipts.

The workspace default since 2026-09-25, under principles 2 and 3; how a builder uses it is
[docs/conventions/splits.md](../../docs/conventions/splits.md), and why is
[the accepted decision](../../docs/decisions/foundry/datasets/2026-09-24_identity_stratified_splits.md).

Canonical suite successors explicitly select `StratifiedSettings(allocation_unit="cluster")`:
85/5/5/5 of finest-level clusters, with largest-remainder rounding and no protein-count floor
or cap; `evaluation_share` changes the 5 (in whole thousandths, exact) for a paired dataset that
cannot fill it at its enforced identity, and the policy must say why. The manifest names
`cluster_count_v1`. A cluster that repair returns to train is
replaced by another, so quotas stay exact, and only a shortfall no eligible cluster can fill
reports realized counts. Its repair runs the audit's twelve directed searches on the first
round and on the finished split, and in between searches only the clusters that moved, so a
train protein or another evaluation split's protein that duplicates a held-out one also sends
that cluster back ([the decision](../../docs/decisions/foundry/datasets/2026-09-28_refill_after_repair.md)).
`split_by_identity_strata(..., incremental=False)` measures everything every round, for a comparison.
The API default retains `protein_count_v1` for existing studies. The
[split convention](../../docs/conventions/splits.md#canonical-suite-release) defines the
grouping and release checks beyond the basic allocator. Logan authorized source/data transfer
and direct workstation downloads on 2026-09-26. The cluster-count and leakage regression lane
passed 96 tests on `suite_gh200` in
`ws-runs/production_suite/20260926-083820-51e445c8cfd7`; its source/runtime/JUnit receipts are
imported through `ws remote import-tests`. This validates the fixtures, not a canonical dataset.

`linked_groups` maps source-group ids to protein ids. Assignment and near-duplicate repair
preserve their transitive components; quotas still count homology clusters. To audit a saved
canonical split, call `validate_identity_stratified_split(..., linked_groups=original_groups,
expected_allocation_unit="cluster")` so altering its recorded policy or dropping source links
cannot make it pass. Canonical builds additionally call `check_cross_split_homology` on every
ordered split pair and reject leakage between held-out splits as well as leakage against train.
Those regression fixtures passed in the same GH200 lane. The suite's
[canonical adapter](../../projects/serving/production_suite/backlog/gate_01_dataset_machinery.md#completed-identity-and-split-machinery)
now enforces a dataset-specific policy and integrates sequence/measurement identities. Its
readback audit reclusters and searches original observations in fresh stages; a real-binary
40-sequence software fixture passed on GH200. Source adapters and actual dataset release
audits remain open. The adapter stays in its first consumer, production_suite.

| Name | Does |
|---|---|
| `cluster_hierarchy(sequences, directory, levels=(0.9, 0.7, 0.5, 0.3), min_coverage=0.8, **settings)` | `ClusterHierarchy`: MMseqs2 at each level, finest first, each level clustering the representatives of the one before, so each 0.9 cluster lies inside one cluster at every lower level. Each level is a `cluster_sequences` stage; `settings` go to `MMseqsConfig` |
| `nest_clusterings(levels, clusterings, origin=())` | The same hierarchy from memberships already in hand, refused unless each level covers the representatives of the one before exactly |
| `split_by_identity_strata(hierarchy, sequences, search, labels=None, settings=StratifiedSettings())` | `IdentityStratifiedSplit`: every protein in `train`, `validation`, `calibration`, or `test`. Each evaluation split holds n/4 proteins per stratum (far, 0.3 to 0.5, 0.5 to 0.7, 0.7 to 0.9), held out as whole clusters whose parent keeps half its proteins in train, aiming also for the label shares; shortfalls pass to the nearer strata and back. `search` then measures every held-out protein against train, and each near-duplicate at 0.9 or more returns to train with its 0.9 cluster until none remains; under the cluster policy another cluster takes each freed place and is measured in turn |
| `build_identity_stratified_split(sequences, directory, labels=None, ...)` | Both of the above with MMseqs2, searching with `MEASURE_SEARCH`: e-value 1e-3 and half the query covered. `answer_from="all_pairs"` answers every repair and audit search from one all-against-all search (`PairTable`) instead of running one each, and records `search_answered_by` in the manifest |
| `validate_identity_stratified_split(hierarchy, split)` | Raises unless every protein has one split, no 0.9 cluster spans splits, no near-duplicate is held out, and the manifest's counts hold |
| `StratifiedSettings`, `evaluation_split_size`, `stratum_quotas` | n, 5% of proteins raised to 500 and capped at 10%; the quotas; the 10% unit cap; the seed |
| `StratifiedSettings.enforced_identity`, `strata_below`, `require_interaction_identity`, `enforced_identity_ceiling` | The identity enforced between splits, one of 0.9 (the default), 0.7, 0.5 or 0.3: the strata left are the levels below it, units are whole clusters at that level or coarser, and repair returns a flagged protein's whole enforced-level cluster to train. `require_interaction_identity(enforced, dataset=..., task=...)` raises for a protein-protein, protein-ligand or CAMP dataset that enforces more than `INTERACTION_ENFORCED_IDENTITY_LIMIT` (0.5), and `enforced_identity_ceiling` returns that limit or 0.9; [the reasons](../../docs/decisions/foundry/datasets/2026-09-29_interaction_datasets_enforce_half_identity.md) |
| `survey_identity_levels(hierarchy, label_sets=None, *, settings=SurveySettings(), ceiling=0.9)` | `IdentitySurvey`: for each of the four levels, before any split exists, the units it leaves, the proteins locked in units too large to hold out, the units a held-out split gets, and the labels it takes out of reach (`LevelSurvey`), and `recommended`, the lowest level that passes all three checks (0.9 when none does, and never above `ceiling`). `to_dict()` reads back from JSON unchanged and `table()` is the Markdown for a record. The vocabulary pilot runs it, and [the rule](../../docs/conventions/splits.md#choosing-the-enforced-identity) says how to read it |
| `StratifiedSettings.unit_order`, `StratifiedSettings.balance_labels`, `recorded_settings(settings)` | How evenly the evaluation splits sample the dataset. `unit_order="seeded"` draws the units of a stratum in a seeded pseudo-random order in place of largest first, and `balance_labels=True` adds each label's share to the allocator's loss (needs `labels`, each a class, a set of classes, or `quintile_labels`). Both are left out of a manifest at their historical values (`recorded_settings`), so a recorded split still validates |
| `StratifiedSettings.label_allocation`, `component_allocation.allocate_components`, `PairTable.counted_hits` | `label_allocation="components"` (needs the cluster policy, `balance_labels`, an enforced 0.3, and `answer_from="all_pairs"`) deals the first allocation as whole components of related sequences by iterative stratification, rarest term first (Sechidis et al. 2011), then swaps equal-size components to lower each split's per-term chi-square; repair and refill still follow. Imports scipy on first use. Left out of a manifest at the default `"greedy"` ([the rule](../../docs/conventions/splits.md#components-allocation), [the record](../../docs/decisions/foundry/datasets/2026-10-01_component_label_allocation.md)) |
| `measure_label_balance(protein_to_split, label_sets, *, max_excess=2.0)` | `LabelBalance`: per evaluation split, the total variation between its label shares and train's, the same figure for a random sample of its size, their ratio (`excess`), the 5th, 50th and 95th percentile prevalence ratio, and the labels it lacks; `passed` at an excess of `max_excess` or less. [Labels alike in every split](../../docs/conventions/splits.md#labels-alike-in-every-split) |
| `identity_stratum`, `quintile_labels`, `calibration_groups` | An identity's stratum; a regression target's quintiles as classes; adjacent strata merged until each holds 100 calibration proteins |
| `StratificationInfeasible` | The dataset declares another scheme: one 0.9 cluster holds most proteins, or an evaluation split falls below half of n |

`split.to_dict()` is the manifest: settings, MMseqs version and each level's clustering origin,
search settings, per split and stratum the proteins, units, and label shares, agreement between
held-out and measured strata, shortfalls, and every move. Ties break on the seeded hash
`assign_clusters_to_splits` uses, so input order never matters; 420,000 proteins split in about
four seconds, before the searches. Pairs, for Atlas, come in the Atlas phase from these pieces.

**An exception to the promotion rule.** This came from annotation_vocabulary's `splitting.py`,
which now re-exports it, and it has no second consumer yet. Logan's plan names splits as a
step of `foundry.datasets`, and every other split in the workspace is a different algorithm:
embedding_translation shuffles cluster identifiers and cuts by cluster count, and the DatasetDev
builders each have their own. Moving any of those here would change splits they have already
published, so each move is a decision for Logan, together with the split name: this module
says `validation`, as Hugging Face does, while embedding_translation and DatasetDev say `valid`.

## Leakage

| Name | Does |
|---|---|
| `check_split_disjointness(rows_by_split, fields, *, unique_within_split=())` | `LeakageReport`: every value of each field that rows of two or more splits hold, and every value of the `unique_within_split` fields that repeats inside one split. Reads each split's rows once. Values must be strings, so `1` and `"1"` cannot pass as different proteins |
| `LeakageReport` | `clean`, `summary()` for an error or a log, `to_dict()` for a card or manifest, and `raise_if_leaking()`, which raises `LeakageError` |
| `check_cross_split_homology(sequences_by_split, search, search_parameters=..., identity_threshold=0.9)` | Independently search every ordered split pair through the caller's existing homology engine; fail on incomplete or invalid measurements |
| `HomologyLeakageReport` | One `HomologyComparison` per directed pair, with query/reference counts, maximum identity, every violating query, and the declared search parameters; `raise_if_leaking()` rejects any cross-split hit at the threshold |

It checks the rows a builder is about to publish, independently of the code that split them.
Split names are the caller's, so `valid` and `validation` both work. Repeats inside a split are
opt-in per field, because the proteins of one cluster share its identifier by design.

Promoted from latent_sep's `validation.py`, which allows one row per accession, sequence, and
cluster in the whole benchmark, and embedding_translation's `split_provenance.py`, which audits
cluster lists. Both now delegate, keep their own error messages, and have new tests pinning the
cases their suites had not covered. embedding_translation's audit record, which its split
manifests store and hash, is unchanged. The other eight copies the survey found, in seven
DatasetDev builders and contact_esmc6b, keep their own checks for now.

## Interaction tables

In `foundry.datasets.ppi`, which the package does not import, so the rest of it still runs
without pandas.

| Name | Does |
|---|---|
| `clean_ppi_df(df)` | A new table without incomplete rows or repeated pairs, on a fresh `RangeIndex`, with the columns in their order and types. A row is incomplete when it lacks a value in any of `IdA`, `IdB`, `OrgA`, and `OrgB` the table has. A pair counts once whichever way round it is written, and its first row is kept. Identifiers compare as strings, so `1` and `"1"` are one protein. Prints the table's size before, between, and after |
| `print_ppi_df_info(train_df, test_df, valid_df)` | Prints each split's size, then its first five rows, in the order train, test, valid |
| `check_pair_split_disjointness(splits, seq_dict)` | A `LeakageReport` of the proteins (`IdA` and `IdB`), sequences, and clusters (`cluster_a` and `cluster_b`, when every split has them) that two splits share, from `check_split_disjointness`. Two identifiers with one sequence count as shared |
| `push_split_config_to_hub(*, hf_repo, prefix, train_df, test_df, valid_df, extra_splits=None)` | Uploads the frames as one private config named `prefix` of a dataset repository, through `DatasetDict.push_to_hub`: splits `train`, `valid`, `test`, then `extra_splits` in its order, on one schema from `align_split_columns`. Nothing is read back and no token is checked, so it is an unchecked upload (see below). `datasets` is imported on first use |
| `align_split_columns(*frames)`, `frame_to_dataset(frame)` | The frames on one column set in order of first appearance, a missing column filled with -1 where the first holder types it as an integer, NaN as a float, and an empty string otherwise; and a `datasets.Dataset` with `int8` columns widened to `int32` and object columns made strings |

Promoted from Atlas's and DatasetDev's `paired_data` modules, which held the same code and now
import it under their own names; DatasetDev's `apply_seq_dedup` calls `clean_ppi_df` after it
remaps identifiers. Nothing changed. Their `assert_split_disjointness`, and DatasetDev's copy in
`ppa_data/ppa.py`, raise `LeakageError` on a report that is not clean, naming the stage and the
first shared values, where each once stopped at the first overlapping pair of splits; synth's
unimported copy waits under item 27 of the approvals record below. `_IdMin` and `_IdMax` serve as scratch columns, so a table's
own columns of those names are dropped, as they always were, and a table without both `IdA` and
`IdB` raises pandas' `KeyError`.

`push_split_config_to_hub` is the one copy of `push_datasets_to_hub` and its two helpers, which Atlas's
`paired_data.py` and DatasetDev's held (Atlas's with `extra_splits`, DatasetDev's without; the rest
identical). DatasetDev's `paired_data` imports it under the old name, so its PPI builders are unchanged;
Atlas's builders import it as `from foundry.datasets.ppi import push_split_config_to_hub` and pass the
same keywords. It keeps the publishing the builders had, `DatasetDict.push_to_hub` at its default shard
size and `private=True`, so it is not `uploads.push_dataset_to_hub`, which takes no config name and sets
`max_shard_size="1GB"`. Like the helpers under Unchecked uploads it is one copy of a way to publish that
already existed twice, not a new one: each builder moves to `publish_dataset` at its next release, with
Logan's go, by the plan that
[the foundry unification](../../docs/decisions/workspace/2026-09-24_foundry_unification.md) records.

The rest of `paired_data` stays in the projects. The split, negative-sampling, matching, and
de-duplication functions have one real consumer, DatasetDev's PPI builders, and DatasetDev's
`paired_data.py` is their one copy. Atlas's copies, which nothing in Atlas calls but the module itself
and its tests, wait under item 27 of
[the approvals record](../../docs/decisions/workspace/approvals_waiting_on_logan.md).

## Unchecked uploads

`uploads.py` holds the one copy of the upload helpers that DatasetDev's, Atlas's, and synth's
`huggingface.py` each held: `require_hub_token`, `push_dataset_to_hub`, `upload_folder_to_hub`,
`upload_embeddings_to_hf`, and `build_basic_dataset_card`. They commit and return, verifying
nothing, which is why they are not under Publishing. Gathering them on 2026-09-26, when Logan asked
for the Hugging Face code to be consolidated, added no way to publish that did not already exist
three times. It also ended synth's `hf_login`, which called `huggingface_hub.login` with `HF_TOKEN`:
that writes the token to disk, and required `HF_TOKEN` even where a stored login would do. The plan
above still holds: each builder moves to `publish_dataset` at its next release, with Logan's go.
The projects' modules import these names. synth's keeps its own proteome index writer and
checkpoint resolvers; DatasetDev's copies of them, which nothing imported, went under item 27.

Whether to keep these beside `publish_dataset` or retire them now was left to the agent (Logan,
2026-09-26: "Not sure, your call"). They stay: every builder that uploads calls them today, and
retiring them before each builder moves to `publish_dataset` would break those builders or bring
back three copies. They go when the last caller has moved.

Tests: `tests/tier1_unit/test_datasets_uploads.py`, against a stand-in Hub.

## Publishing

| Name | Does |
|---|---|
| `publish_dataset(folder, repo_id, *, commit_message, private, check=None)` | Makes the repository hold exactly `folder` at one new revision and returns it as `PublishedRevision`. One commit, which deletes remote files the folder no longer holds. A new repository is created private and deleted again if it fails to verify. An existing one is verified on a staging branch before `main` moves. Every file is downloaded back and compared byte for byte. Visibility is set last, and `private` has no default |
| `check(revision, downloaded)` | Project-specific verification run on each candidate revision before `main` moves, such as loading every dataset config. Raising rejects the revision |
| `PublishedRevision`, `VerifiedFile` | The repository, the immutable revision, the commit URL, and each file's local and published digest |
| `PublicationError`, `IntegrityError` | A publish stopped; the second means the revision on the Hub differs from the folder |

Publishing needs `huggingface_hub`, and nothing else here does: the package imports it on first
use of a publishing name, so embedding_translation's split audit still runs without Hugging Face
libraries, as its module promises. The folder is whatever the builder wrote, Parquet or
otherwise, so `datasets` and `pyarrow` stay in the projects that build and load configs.

Authentication belongs to `huggingface_hub`, through `HF_TOKEN` or a cached login. No function
here accepts a token. Before anything is uploaded, every file is searched for the value of each
credential-named environment variable, using `foundry.logging.is_credential`; a builder that
writes `vars(args)` into a manifest would otherwise publish its `--hf_token`. `ws publish`
vendors `foundry.logging` alongside this module for that reason.

Promoted from annotation_vocabulary's `hub_release.publish_release`, which now validates its
release and delegates the publication here, with its 23 tests as the guard. gem_integration's
`scripts/publish_hf_dataset.py` is the second consumer.

## Not here yet

In the order the dataset survey recommends:

- **The dataset card**, which would print a `LeakageReport` beside the configs and the origin.
  It has no second consumer yet: annotation_vocabulary's `card.py` generates its `configs:`
  frontmatter from the release files, while latent_sep's card has static frontmatter and lets
  `push_to_hub` write the configs. The card moves here when a second builder writes configs.
- **UniProt retrieval and evidence tiers**, from serving's `curated_client.py`.
- **A cross-split homology audit**, such as contact_esmc6b's `verify_post_split_homology`, which
  rejects any search hit that joins two splits. The search and its parser are here now, under
  Homology search; the audit built on them has one consumer.

Decisions UniProt retrieval and a full build pipeline wait on, all Logan's: the split name
(`valid` or `validation`), which evidence counts as experimental, whether a publish writes its
revision into the dataset manifest, how MMseqs2 runs on Windows and which version is pinned,
and whether publishing defaults to private.

The DatasetDev builders cluster through `dataset_dev_utils.cluster_sequences`, which is
[`docker_clustering`](#docker-run-clustering) since 2026-10-05: `easy-cluster`, 0.8 coverage, and a cache keyed by
name rather than content. They are the natural next consumers of `cluster_sequences`, but moving
them changes their clusters, so their published splits would not rebuild. That move waits for the
comparison of memberships described there, and then for Logan.

Tests: `tests/tier1_unit/test_datasets_hub.py` against a stand-in Hub,
`tests/tier1_unit/test_datasets_mmseqs.py`, `tests/tier1_unit/test_datasets_homology.py`, and
`tests/tier1_unit/test_datasets_stratified.py` against a stand-in MMseqs,
`tests/tier1_unit/test_datasets_identity_survey.py`, `tests/tier1_unit/test_datasets_label_balance.py`,
`tests/tier1_unit/test_datasets_splits.py`,
`tests/tier1_unit/test_datasets_leakage.py`, `tests/tier1_unit/test_datasets_ppi.py` (the interaction tables and the
split push, against a stand-in `datasets`), and `tests/tier1_unit/test_datasets_docker_clustering.py` (against a fake command
runner). None
needs the network or the binary.
