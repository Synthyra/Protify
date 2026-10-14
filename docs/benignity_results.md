# Benignity results and evidence

Start with [result:2026-10-05_benignity_balanced_v2](../../../../results/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/README.md).
It compares the original and balanced-v2 h512 heads on identical retained test rows, with seed-level
scores, plots and independent archive checks. Exact balance did not rescue training: all 24 v2 fits
selected initialization. The linked records retain the evidence and limits of each comparison.

## Dataset versions

| Dataset record | Cohort used by these experiments | Release status |
|---|---|---|
| [Original benignity](../../../../datasets/benignity/README.md) | Strict adopted binary export: 893,735 rows; test 24,516 rows | Private v1; the source release has 894,135 rows before strict supervision |
| [Balanced benignity v2](../../../../datasets/benignity_v2/README.md) | Deterministic majority downsampling within original partitions: 676,096 rows; test 16,444 rows; every split 50/50 | Private diagnostic descendant; original labels and partition boundaries retained |

The endpoint is source-proteome membership under a dated inventory, not measured protein function
or safety. Both records restrict new-source-domain claims. The v2 holdouts were already examined
in v1 work. Average precision on the balanced test cohort has a different prevalence reference
from v1; use the matched-row comparison in the latest result when comparing dataset versions.

## Related comparisons

| Question | Result or figure record | Scope to retain when citing |
|---|---|---|
| Were the original overview values inflated? | [Corrected Oracle ablation figure](../../../../figures/production_suite/2026-10-02_oracle_ablation_summary/README.md) | Actual metrics replace per-row rescaling; undefined random-vector MCC remains missing |
| How did the original model-size matrix behave? | [result:2026-10-02_registry_matrix_benignity](../../../../results/serving/production_suite/2026-10-02_registry_matrix_benignity/README.md) | Original 24,516-row test cohort; one seed per cell |
| Did Atlas profiles help the original head? | [result:2026-10-02_atlas_profile_probe](../../../../results/tooling/property_lowcode/2026-10-02_atlas_profile_probe/README.md) | One-seed profile and wider-trunk controls; class-probability ranking metrics |
| Why was the random-vector MCC absent? | [result:2026-10-05_suite_randomvectors_matrix](../../../../results/serving/production_suite/2026-10-05_suite_randomvectors_matrix/README.md) | Constant class-1 predictions; undefined canonical MCC; raw-logit scores; incomplete full saved-ID audit |
| Did a 512-wide hidden layer and regularization help? | [result:2026-10-05_esmc300_profile_h512_tuning](../../../../results/tooling/property_lowcode/2026-10-05_esmc300_profile_h512_tuning/README.md) | Original cohort; ten recipes per arm and three final seeds; four of six selected heads are initializations |
| Did exact dataset and batch balance help? | [result:2026-10-05_benignity_balanced_v2](../../../../results/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/README.md) | Balanced cohort; ten recipes per arm and three final seeds; all six selected heads are initializations; existing v1 logits scored on the same rows |

The later h512 comparisons use raw-logit ranking scores and different selection recipes from the
earlier profile probe. The v1 random-vector fit is a separate control, not a matched v2 h512 fit.
The profile arm retains more parameters than the base arm. These records do not isolate feature
count or establish a learned Atlas gain.

## Scores, plots and execution records

| Experiment | Protocol and execution receipts | Scores and plots |
|---|---|---|
| Original h512 | [experiment:2026-10-05_esmc300_profile_h512_tuning](../../../../experiments/tooling/property_lowcode/2026-10-05_esmc300_profile_h512_tuning/README.md) | [Result and captions](../../../../results/tooling/property_lowcode/2026-10-05_esmc300_profile_h512_tuning/README.md), [unrounded summary](../../../../results/tooling/property_lowcode/2026-10-05_esmc300_profile_h512_tuning/summary.json), [frozen evaluation](../../../../results/tooling/property_lowcode/2026-10-05_esmc300_profile_h512_tuning/result.json) |
| Balanced v2 h512 | [experiment:2026-10-05_benignity_balanced_v2](../../../../experiments/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/README.md) | [Result and captions](../../../../results/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/README.md), [unrounded summary](../../../../results/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/summary.json), [frozen evaluation](../../../../results/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/result.json), [independent metric audit](../../../../results/tooling/property_lowcode/2026-10-05_benignity_balanced_v2/audit.json) |

Each dated experiment folder holds recipes, source archives, pilot and launch receipts. Its dated
result folder holds the evaluated scores, publication-sized PNGs and captions. The result's
`result.json` is the frozen evaluation; `summary.json` is its checked compact interpretation.
Retained phase echoes such as the v2 experiment's `tune.json` are not separate evaluations.

Both h512 experiments reuse saved ESMC-small and Atlas features. Large datasets, head checkpoints,
full sampling receipts and raw prediction archives remain on registered Modal volumes. On
`oracle-suite-matrix`, their mounted run roots are:

- `/vol/suite/runs/2026-10-05_esmc300_profile_h512_tuning`
- `/vol/suite/runs/2026-10-05_benignity_balanced_v2`

The frozen evaluation records bind exact archive paths and hashes. The dataset records bind private
Hub revisions, manifests and selection receipts. Canonical ESMC inputs remain on `embeddings-esmc-300`;
the [Atlas profile-store result](../../../../results/serving/production_suite/2026-10-02_atlas_proteome_profile_store/README.md)
records the existing profile store. The per-experiment `analyze.py` files reproduce local summaries
and PNGs from saved receipts without retraining. All evidence remains in its original dated location.
