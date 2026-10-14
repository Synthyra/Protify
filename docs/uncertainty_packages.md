# Uncertainty packages

`python -m protify.uncertainty package` turns one finished uncertainty cell into a directory a
service can load: the trained heads, the embedding standardizer, and the calibration and
conformal quantiles the cell measured. production_suite trains each Oracle task as one final
cell and packages it with this command. The method is rebuilt from the cell's saved per-seed
predictions by `methods.build_methods`, the function its `report.json` was scored with, so a
package ships the calibration the cell measured and never fits one again. For the cells
themselves, see [Configuration](cli_and_config.md#uncertainty-cells); for comparing them, see
[Uncertainty analysis](uncertainty_analysis.md).

---

## Keeping the heads

A cell keeps its trained heads only when asked:

```yaml
uncertainty:
  save_heads: true
```

With `save_heads: true`, the run also writes, in its `output_dir`:

| File | What it holds |
|---|---|
| `heads/seed_<seed>/` | The seed's head as `save_pretrained` writes it: `config.json` and `model.safetensors`. These are the weights early stopping restored, with the SNGP posterior `fit_posterior` fitted to them. |
| `standardizer.npz` | `fitted`, then the embedding standardizer's `mean` and `scale`, `(d_in,)` float32 each, when one was fitted, and `train_label_std`, the training labels' standard deviation, which sets the regression conformal floor and which `report.json` does not record. It is written before the first seed trains. |

A head is saved before its seed's `.npz` and `.json`, and the `.json` is written last and marks
the seed finished, so a crash in between retrains the seed rather than leaving it finished
without a head. Saving a head draws no random number and moves no prediction: with `save_heads`
off, the default, a cell writes the same seed files and report as before, and its `cell.json`
records `save_heads: false`. An older `cell.json` without the key reads as false.

`save_heads` is a training setting, so resuming an `output_dir` under the other value is refused
as a change of training settings. A resume with `save_heads` on is also refused:

- before any data loads, when a finished seed has no saved head, naming those seeds. A finished
  seed never trains again, so its head cannot be recovered. This happens when seed files are
  copied into a directory without their heads, as when a final cell resumes from a tuning
  cell's first seed: copy `heads/seed_<seed>` along with that seed's files, or move the seed's
  `.npz` and `.json` aside so it trains again;
- when the standardizer it fits differs from the recorded one beyond float32 rounding, since the
  saved heads read features through the recorded one. A cache or the training split changed;
  use a new `output_dir`.

---

## Packaging a cell

```bash
python -m protify.uncertainty package <cell output_dir> \
  --method temperature \
  --output <package dir> \
  --levels 0.8 0.9 \
  --revisions revisions.json
```

From the workspace, put `ws run property_lowcode --` in front of it.

| Option | Default | Meaning |
|---|---|---|
| `cell_dir` | required | The finished cell's `output_dir`. It must have run with `save_heads: true`, and every seed must be finished and have its head. |
| `--method` | required | The method to ship. |
| `--output` | required | The package directory. It must not exist; a package is written once. |
| `--levels` | the levels `report.json` scored | Conformal coverage levels. V_cal's `n` rows can calibrate levels up to `n / (n + 1)`, and a higher one is refused. |
| `--revisions` | none | A JSON object mapping each repository the features were embedded with to its revision. It must name every repository `features.json` lists. Without it, each revision is recorded as unknown and the model card warns. |
| `--crossfit` | none | The cell's `crossfit --correctness-only` output. The package then ships the correctness model fitted on it; see [The correctness model](#the-correctness-model). Every Oracle classification package passes it. |

From Python, `package.build_package(cell_directory, method_name, output, *, levels=None,
revisions_path=None, crossfit_dir=None)` does the same and returns the package path.

A method can ship when it predicts, not only ranks:

| Head | Methods |
|---|---|
| `projected` | `plain`, `temperature`, `mc_dropout`, `mc_dropout_temperature`, `ensemble`, `ensemble_temperature` |
| `sngp` | `sngp`, `sngp_temperature`, `sngp_ensemble`, `sngp_ensemble_temperature` |

The methods ending in `_entropy`, `_nn_distance`, or `_variance` rank rows by another score but
predict as their head, and are refused; package the head's own method instead.

**Not built yet: the single-probe run's heads and methods.** Packaging the `linear`,
`faithful`, `beta_nll`, and `projected_qclip` heads, and the `laplace_temperature`, `loss_head`,
and `temperature_pnorm` methods of any head, is not built, and `package` refuses each with a
message that says so. A package rebuilds a head's features with a plain standardizer and the
probe's own clamp, and ships one head's outputs and one fitted map. The quantile clip's bounds,
a variance output, the Laplace posterior and its chosen prior precision, the loss predictor, and
the chosen norm order would each need a place in the package and in its loader. They are built
once the single-probe run adopts one of them.

An ensemble ships every seed. Any other method ships one seed: the one whose head scored the
best V_fit primary metric, `report.json` `per_seed[].v_fit_primary`, with ties going to the
lowest seed and seeds with an undefined metric passed over. Test is never read to choose.

Before writing, the package is checked three ways, and a disagreement is refused:

1. Its calibration, applied to the saved per-seed predictions by the same code that applies it
   to new rows, must reproduce `build_methods`' probabilities, mean, variance, and uncertainty
   exactly on V_fit, V_cal, and test.
2. Its quantiles must reproduce `conformal_sets`' test coverage and set size or interval width
   exactly.
3. Its fitted parameters and test scores must match the shipped run's entry in `report.json`
   to a relative 1e-6, the room a different summation order needs. A report scored by an older
   method or scorer fails here; running the cell again rewrites `report.json` from its seed
   files without training.

The package is built in a temporary sibling of `--output` and renamed into place, so a reader
never sees a partial one, and a refused package leaves nothing behind.

## Partially observed targets

An explicit mask distinguishes an unknown target from a measured zero. The package records
`observation_contract` in both `calibration.json` and `origin.json`, and the cold loader validates
their agreement, counts, per-label class support, and whole-answer capability. Predictions carry
the same contract. Legacy packages without explicit masks retain their existing behavior.

Temperature fitting uses observed V_fit targets. Scalar conformal quantiles use observed V_cal
targets, including the finite-sample limit based on that observed count. Reported coverage applies
to the measured population under the conformal method's exchangeability assumptions. Coverage
of unobserved truth or distant protein families is not established. Multilabel predictions still
have no conformal sets. Per-label support records explicitly flag positive-only and unsupported
labels; temperature scaling cannot establish discrimination with no measured negatives.

A cell with `unlisted_labels: negative` reads a declared multilabel release as a closed world: every
label a row does not carry is a negative, so every label counts as observed and the package is
fully observed (`whole_answer: complete_calibration_truth`). It scores `fmax`, `scored_micro_aupr`,
`scored_macro_aupr` and `scored_terms` beside the primary `f1`, over the terms with at least 10
observed positives in the test split. A precision under it is a lower bound on one against complete
annotations. The Gene Ontology Oracle is the one user ([probes](../../../../docs/conventions/probes.md)).

If V_fit or V_cal is partially observed for classification, the development package returns
point `answers` and observed-target `probs`, but `confidence`, `reliability` (P(right)), and
`p_values` are `None`, meaning unavailable rather than zero. `head.select(prediction)` returns
all false, including for an explicitly supplied valid target. A caller must not substitute raw
scores for the missing whole-answer probability. This is not a qualified public Oracle package.
The existing M8 requirement remains, and `--crossfit` refuses partial truth.

Test labels do not choose this contract or the answer rule. With complete calibration truth,
partially observed test rows retain the same inference rule. An observed disagreement proves an
error; a complete observed match proves correctness; partial agreement remains unknown.
`metrics.json` reports these three counts and bounds on whole-answer accuracy and answered
error. Point estimates are `null` when not identifiable, and `test_label_support` records the
observed denominators. Unknown fillers never enter calibration, class indexing, or metrics.

The [decision](../../../../docs/decisions/serving/production_suite/2026-09-28_partial_label_packages.md)
records why this contract does not assume random missingness or extend M8 to incomplete truth.
Optional balanced regression, Laplace, loss-head, and pnorm paths remain explicitly unsupported
for partial labels. The separate frozen calibration stage still rejects partial labels; its
integration is tracked by `suite_calibration_execution`. Development package support does not
bypass the public correctness gate or unlock test data in a validation-only fitting run.

---

## What a package holds

```
<package dir>/
  README.md            the model card
  calibration.json     the method, its seeds, its fitted calibration, its conformal quantiles
  features.json        how to build the rows the heads read
  metrics.json         the shipped predictor's test scores
  origin.json          the cell, dataset, split, seeds, and Protify revision
  standardizer.npz     mean and scale, (d_in,) each; absent when the cell fit none
  heads/seed_<seed>/   one head per shipped seed, as the cell saved it
  correctness/         with --crossfit: reference.npz and correctness.json
```

### calibration.json

| Field | Meaning |
|---|---|
| `scheme` | `validation_carve`: calibration fitted on V_fit and conformal quantiles on V_cal, both carved from validation. |
| `method`, `head`, `task_type`, `label_type`, `num_labels` | The method and the task, as `report.json` names them. |
| `predictive` | What the method starts from: `head`, `mc_dropout`, or `ensemble`. |
| `calibration` | `null`, `logit_temperature` (the head's logits divided by `T`), `probability_temperature` (the log of the MC dropout or ensemble mean probabilities, or their log odds for multilabel, divided by `T`), or `variance_map` (regression, `sigma^2 = a v + b`). |
| `fitted` | The fitted parameters, as the method's run in `report.json` records them: `{}`, `{"temperature": T}`, or `{"a": a, "b": b}`. |
| `seeds` | The shipped seeds. |
| `seed_choice` | `{"rule": "every_seed"}`, or `{"rule": "best_v_fit_primary", "primary_metric": ..., "per_seed": [{"seed": ..., "v_fit_primary": ...}]}`. |
| `mc_dropout_samples` | Passes per row for an MC dropout method, else `null`. |
| `rows` | V_fit and V_cal row counts. |
| `split_sha256` | SHA-256 of the cell's `split.json`, which names V_fit and V_cal by row digest. |
| `conformal` | Binary and multiclass: `levels`, `set_scores` (`lac`), and `quantiles`, one `{"level", "lac"}` threshold per level. Regression: `levels`, `spread` (`sigma` or `none`), `spread_floor`, `train_label_std`, and `quantiles`, one `{"level", "quantile"}` per level. Multilabel: `null`. |
| `correctness` | With `--crossfit`: `directory` (`correctness`) and `scheme` (`latent_correctness_v1`). Absent otherwise. |
| `answer_rule` | Classification: `scheme` (`conformal_selection`), `score` (`confidence`, or `correctness` when the package ships a correctness model, whose reliabilities `wrong_confidences` then holds), `answer` (`top_class` or `label_set`), `at_least_one_label` (multilabel: whether every V_fit and V_cal row carries a label, else `null`), `calibration_rows`, `calibration_error` (V_cal's share answered wrongly), `default_target` (half of it), and `wrong_confidences`, the confidence of every V_cal row answered wrongly, ascending. Regression: `null`, since it claims no per-row confidence. A package written before the rule has no key. |
| `applicability` | `latent_distance_applicability_v1`. `status` `calibrated` holds the `signal` (`inpca_d1`), its `meaning`, the conformal quantile `level` and `cutoff` over V_cal, `calibration_rows`, `novel_share`, the `reference_sha256` it was measured against, and `strata`: the score it ranks by, its band `edges`, `minimum_rows`, and one cell per novelty status and band. `status` `unavailable` holds a `reason_code` in place of all of it. See [Applicability and the decision policy](#applicability-and-the-decision-policy). A package written before it has no key. |
| `decision_policy` | `answer_or_abstain_v1`: the `selection` it runs (the rule's score, `target`, the count of V_cal's wrong answers, and the digest of the answer rule), the digest of the applicability record, what a `novel` row and a row in an `insufficient_stratum` do, `escalation` (`none`), and its own `sha256`. `null` where there is no answer rule. |

A class joins a row's set when its LAC score, `1 - p_k` for class `k`, is at most the level's
threshold. The report also measures APS sets, and a package never ships them
(`SHIPPED_SET_SCORES`). On a binary task at high levels, the APS threshold sits at a cumulative
probability of 1.0, so floating-point rounding, which differs between machines, decides whether a
set holds one class or two. Its coverage then moves by up to about 0.01 from one machine to the
next, and a package built on another machine than its cell would fail its check against the
report. A regression interval is `mu +/- q (sigma + floor)`. The floor is 1e-3 of `train_label_std` for a method with
a sigma; a method without one takes a zero sigma and a floor of 1, so it scores the absolute
residual. This is `regression_conformal_rows`' rule: the floor is added to sigma, not taken as
its minimum.

### features.json

| Field | Meaning |
|---|---|
| `branches` | One entry per branch, in join order: its `spec`, `model_name` (the resolved name, which carries an SAE's layer, sparsity, and codebook width), `pooling_types`, `hidden_state_index`, `width`, and whether it is `sparse`. |
| `width` | `W`, one protein's joined width, the sum of the branch widths. |
| `pair`, `pair_feature_mode` | Whether rows are protein pairs, and the rule that joins two proteins' rows. |
| `input_size` | `d_in`, what the heads read: `W`, or `2W` for pairs. |
| `max_length`, `embedding_dtype` | The embedding settings the caches were written under. |
| `standardizer` | `standardizer.npz`, or `null`. |
| `input_clip`, `projection_size`, `head` | The head's clip on standardized values, its projection width, and its kind. |
| `backbones` | Each repository a branch was embedded with: `model_name`, `component` (`backbone`, or `sparse_autoencoder` for an SAE's Biohub codebook), `repository`, and `revision`, `null` when unknown. A model that loads nothing, such as a one-hot encoding, has repository `null`. |

### The answer rule

`answering.py` defines what a classification package answers and when:

- **The answer:** the top class, or for multilabel every label at 0.5 or above. When
  `at_least_one_label` holds, a row whose every label falls below 0.5 answers its single most
  probable label, the most probable non-empty set under independent labels.
- **The confidence:** the calibrated probability that the answer is right: the top probability,
  or the product of each chosen side's probability, divided under `at_least_one_label` by one
  minus the empty set's probability. Right means the class, or the whole label set.
- **The p-value** against "this answer is wrong": `(1 + #{wrong V_cal answers with confidence >=
  c}) / (n + 1)`.
- **Selection:** Benjamini-Hochberg at a target over the rows asked about together (Jin and
  Candes, arXiv 2210.01408). When V_cal and those rows are exchangeable, the expected share of
  wrong answers among those answered stays within the target.

Two results justify it: result `2026-09-25_calibration_neighbor_uncertainty` and result
`2026-09-25_calibration_logistic_uncertainty` in the workspace. On the published splits, with a
validation carve that is not test-like, selection kept its target on 5 of 9 datasets. EC's
forced top label was right on about 30 percent of V_fit and V_cal rows but on none of test's, whose
empty-set rows carry more labels.

The exchangeability that holds is between protein families, not rows. On identity-stratified
splits (result `2026-09-25_identity_stratified_calibration`), the rule kept its target on 4 of 6
classification datasets. MB's test error among answered rows was 0.255 against a 0.103 target,
because one family of 44 rows that the head gets confidently wrong fell in test. Redrawn by family,
the error averages near the target, but a single package can miss it by 0.15. Read the target as
an average over which families a calibration set and a query stream hold. The decision
`2026-09-25_answer_rule_family_dependence` asks what packages should state about it.

### The correctness model

A package built with `--crossfit` also ships `correctness/`, a logistic regression of whether
each answer is right (`uncertainty/correctness.py`). Logan made it the default for every Oracle
on 2026-09-26, in place of identity to training by MMseqs
(`docs/decisions/tooling/property_lowcode/2026-09-26_m8_without_mmseqs.md` in the workspace).
Its columns are the logit of the confidence and signals of where the protein sits among the
training proteins the head read:

- **Gaussian**, in the head's trunk: `maha_global`; for single-label also `maha_answer`, `rmd`,
  and `ddu`.
- **Neighbors**, among up to 256 nearest reference rows in three spaces (the trunk, the input
  whitened on its top 256 principal components, and which SAE features are active, by Tanimoto):
  `d1`, `dk` (mean of the 10 nearest), `agree` (share of the 10 nearest labeled as answered),
  `trust`, and for multilabel `jacc`.

The reference, `reference.npz`, holds at most 20,000 training rows, sampled to keep every class,
with their trunk, whitened input, active SAE bits, and labels. Distances are divided by V_fit's
median nearest distance, so a fold head reads on the served head's scale.

**Where it is fitted.** On out-of-fold rows at cross-fit level 0.3: each fold head, trained with
whole identity clusters held out, reads a sample of up to 50,000 held-out rows a level against a
reference from its own training rows. The model therefore learns how signals relate to errors on
families the head never trained on, which the head's own V_fit cannot teach it.

```bash
python -m protify.uncertainty crossfit <cell output_dir> --plan <plan.json> --output <crossfit dir> --correctness-only
python -m protify.uncertainty package <cell output_dir> --method temperature --output <package dir> --crossfit <crossfit dir>
```

`package` refuses a cross-fit whose served seed is not the shipped seed, a method other than one
calibrated head, and answers that differ from those the cross-fit read its signals for. After
writing, it predicts 64 test rows through the written package and refuses one whose reliability
differs from the cross-fit's by a median above 1e-3 or at most above 0.1: the cross-fit reads the
head on its training device, in TF32 on CUDA. `metrics.json` `correctness` holds, for V_cal and
test, the Brier score, NLL, equal-mass ECE, and AURC of the reliability beside the confidence's,
and `check`, the drift measured.

For declared `validation_only` fits, add `--calibration-only` to the correctness-only
cross-fit command. This path reads train, V_fit, and V_cal with test inaccessible; its
five-fold plan uses the release's full protein digests. Its reload-check rows come from
V_cal. `package_correctness(..., calibration_only=True)` fits the same compact model on
OOF rows and the same answer rule on V_cal, with no test metrics. This is a calibration
primitive, not a package accepted by the full release builder. The
[calibration-only contract](../README.md#calibration-only-compact-cross-fitting) gives the
input schema, artifact/resume rules, checks, and remaining suite-stage work.

### Applicability and the decision policy

The answer rule and the correctness model speak for proteins that look like V_cal. `applicability.py`
records where that boundary is and turns it, the answer rule, and the support behind each claim
into one decision per row. Only a package with a correctness model has a latent reference, so only
it can tell a novel row from a familiar one; any other classification package records why it
cannot (`no_latent_reference`, `partial_calibration_truth`, `insufficient_calibration_rows`) and
its policy answers from selection alone, marking each answer `applicability_unassessed`.

- **Novelty** is `inpca_d1`: the distance from a row's clamped, standardized input, whitened on
  the reference's top principal components, to its nearest reference row, in units of the median
  such distance among V_fit rows. Nearest-neighbor distance in a normalized representation is a
  strong out-of-distribution score (Sun et al., ICML 2022). The cutoff is the conformal quantile of
  V_cal's distances at 0.95 (Bates et al., Annals of Statistics 2023), so a new row exchangeable
  with V_cal lies past it with probability at most 0.05. It is a distance in a learned space and
  never a percent sequence identity, and a row inside it is not thereby shown to be in
  distribution. V_cal's top 5 percent must hold at least five rows, or the package states no
  cutoff.
- **Support** tabulates V_cal's rows by novelty and by band of the score the answer rule ranks by
  (edges 0.5, 0.8, and 0.95), each cell with its right answers and a Clopper-Pearson upper bound on
  its error. A cell of fewer than 30 rows is `insufficient` and states no rate: with no wrong
  answer observed, the 95 percent upper bound on the error is 3 / n, so thirty rows can at best
  bound it near 10 percent.
- **The decision** runs in this order. A row whose output is not finite is `not_assessed`
  (`non_finite_output`). A novel row (`novel_input`), or a row in an insufficient cell
  (`insufficient_stratum_support`), abstains. Benjamini-Hochberg selection at the answer rule's
  target then runs over the remaining rows of one request together, and a row it does not select
  abstains (`selection_not_passed`); one it selects answers (`conformal_selection_passed`). A
  package whose V_cal holds fewer than 10 wrong answers cannot support an error target, so it
  assesses nothing (`insufficient_calibration_errors`); neither does regression, a package with no
  answer rule, or one whose V_cal truth is too partial to calibrate one.

What this does not say:

- **The guarantee is not restated.** Conformal selection's false-answer-rate statement holds for
  rows exchangeable with V_cal. The two screens make the answered rows a subset chosen from
  features alone, and the p-values still count all of V_cal's wrong answers, so the statement is
  not made for that subset. `metrics.json` `decision_policy` reports the error of the rows the
  policy answers on test, and the support table reports V_cal's error by stratum; those are the
  evidence.
- **A decision belongs to its request.** Selection reads the rows of one request together: a row
  is answered when its p-value is at most `target * k / m`, for `k` rows answered among the `m`
  tested, so a request that holds many rows the package is unsure of is stricter for every row in
  it. A row answered in a request is answered alone, because that threshold never exceeds the
  target and one row is answered when its p-value is at most the target; a row answered alone can
  abstain in a request, and never the reverse. The target bounds the expected share of wrong
  answers among one request's answers, a statement a request of one row does not carry: a lone
  row is bound only by the chance that it is answered and wrong. `metrics.json` therefore reports
  the test rows both ways, decided as one request and decided one row per request
  (`decide_alone`), and the second answers every row the first does, and more.
- **Abstaining is not a verdict.** It says the package will not vouch for the answer, not that
  the answer is wrong. No policy here escalates a row to a larger base model; that is a separate
  routing policy with its own validated release. No percentage or confidence band is defined,
  and a display policy must read the support table.
- **Regression has no novelty indicator and no answer rule.** Its conformal interval holds the
  truth for the stated share of proteins drawn like V_cal, on average over them. The Synthyra API
  returns an interval only for a package whose interval has one width for every row
  (`head.fixed_width_interval`: a plain or temperature-scaled head, since the temperature's
  variance map is a constant); SNGP, MC dropout, and ensembles carry a spread that varies per
  row.

### metrics.json, origin.json, and the model card

`metrics.json` holds `method`, `seeds`, `split` (`test`), `levels`, `primary_metric`, `primary`,
and `test`, the shipped run's test scores computed by `report.py`'s scorers. They equal the run's
entry in `report.json`, except that only the package's levels are scored. `answer_rule` holds the
answer rule on test at its default target: `target`, `accuracy` of the answers, the share
`answered`, and `answered_error`, or `null` for regression. A package with a correctness model
also holds `correctness` (see above) and `decision_policy`: the share of test rows called novel
and each support cell on test (`applicability`), and what the policy did with the test rows
decided as one request (`decisions`) and decided one row per request (`single_row_decisions`):
rows answered, abstained from, and left unassessed, the error among the answered and among the
abstained, and a count of each reason code.

`origin.json` holds `cell_id`, `experiment_id`, `tags`, `dataset` (`name`, and the `revision`
the cell pinned, else `null`), `split` as `report.json` records it, `data_seed`, `seeds`,
`shipped_seeds`, and `protify_revision`, from `foundry.logging.revision.source_revision` over
Protify's `src`, as the Modal grid computes it. That is the revision of the code that packaged
the cell, which needs a source checkout: from an installed wheel, the command stops. The
revision the cell trained with is in its W&B run.

`README.md` is the model card, built from these records: inputs and backbones, calibration and
quantiles, test metrics, origin, and a usage example.

---

## Predicting with a package

```python
from protify.uncertainty.packaged import load_package

head = load_package("<package dir>")
prediction = head.predict(features)                        # features: (n, W) float32
prediction = head.predict(features_a, features_b)          # a pair task: each (n, W)
prediction = head.predict(features, batch_size=256, device="cuda")
```

`features` holds each protein's pooled branch vectors, as the branch caches store them,
concatenated in `features.json`'s branch order, before standardization. The package joins a
pair by the cell's rule, standardizes, and hands the rows to its heads, which clip and project
them as in training. `load_package` refuses a package whose method and calibration this Protify
would not predict that way, or whose heads do not read `input_size` columns into `num_labels`
outputs.

A `Prediction` holds, per row:

| Field | Shape | Filled for |
|---|---|---|
| `probs` | `(n, c)` | binary, multiclass, and multilabel, after calibration |
| `mean`, `sigma` | `(n,)` each | regression; `sigma` only for a method with a spread |
| `uncertainty` | `(n,)` | the method's score, higher for less certain rows |
| `sets` | score, then level, to `(n, c)` bool | binary and multiclass |
| `intervals` | level to an `Interval` with `(n,)` bounds | regression |
| `answers` | `(n,)` class, or `(n, c)` bool label set | classification, from a package with an answer rule |
| `confidence` | `(n,)` | the same: the probability that the answer is right |
| `p_values` | `(n,)` | the same: each answer's conformal p-value against being wrong, from `reliability` when the package ships a correctness model |
| `reliability` | `(n,)` | a package with a correctness model: its probability that the answer is right |
| `signals` | name to `(n,)` | the same: the latent signals it read |

`head.select(prediction, target)` returns which rows to answer, `(n,)` bool, by
Benjamini-Hochberg at `target`, by default the package's `default_target`. It selects the rows
together, so pass every row to be decided at once. `head.novelty(prediction)` marks each row
`supported` or `novel` against the package's cutoff, or returns `None` when it has none, and
`head.decide(prediction)` returns a `Decisions` of `answer`, `abstain`, or `not_assessed` per row
with its reason codes and the policy's digest, under the policy's own target, which a caller
cannot change because the digest names it. A package that cannot vouch for an error target, or
predates its policy, leaves every row `not_assessed` and never turns that into an answer.
`head.fixed_width_interval` says whether a regression package's interval has one width for every
row.

Prediction runs three phases with the cell's own code. `PackagedHead.seed_predictions` runs each
shipped head through `collect_predictions`, as the cell did for its seed files: SNGP reports
mean-field logits and predictive variance, and the plain head its MC dropout mean.
`calibrated_predictive` turns those into the method's prediction with the functions
`build_methods` uses, applying the stored parameters. `conformal_prediction` applies the stored
quantiles.

Two things a package cannot repeat exactly:

- **The device.** A package gives back its cell's head outputs to float rounding on the device
  type the cell predicted on. The workflow predicts on CUDA when one is visible, and Protify
  runs CUDA matmuls in TF32, so a cell trained on a GPU and a package run on a CPU differ at that
  precision: by up to about 1e-3 in the logits of the tests' toy heads.
- **MC dropout passes.** A package draws its passes from torch's generator seeded with the
  shipped seed, so a call repeats itself, but not the passes the cell drew; `metrics.json`
  scores the cell's.

---

## Tests

`src/protify/testing_suite/test_uncertainty_package.py` runs four toy cells through the
workflow on CPU, binary and regression with each head, and packages every shippable method of
each. The package's predictions of the test rows must give back `metrics.json` and the method's
`report.json` scores. `test_crossfit.py` cross-fits a toy cell and packages it with its
correctness model, and `test_correctness.py` and `test_latent_geometry.py` check the model's
pieces against direct computations. See [Testing](testing.md).
