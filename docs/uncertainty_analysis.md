# Uncertainty analysis

`python -m protify.uncertainty compare` reads finished uncertainty cells and answers the two
questions of the ESMC-600M ablation, `2026-09-24_esmc600_representation_uncertainty`, by the
rules its README fixed before launch: which featurization to use (question 1) and how to
quantify uncertainty (question 2). When the cells include a head only the single-probe run,
`2026-09-25_single_probe_uncertainty`, trains, it also applies that run's rules; see
[The single-probe rules](#the-single-probe-rules). It runs on CPU from the files each cell wrote,
its seed predictions and `report.json`; it trains nothing and loads no model. For the cells
themselves, see [Configuration](cli_and_config.md#uncertainty-cells).

---

## Running it

The ablation's analysis stage runs:

```bash
python -m protify.uncertainty compare \
  --runs /vol/artifacts/<experiment>/fit \
  --pattern "*/*/final/report.json" \
  --output /vol/artifacts/<experiment>/analysis
```

From the workspace, put `ws run property_lowcode --` in front of it.

| Option | Default | Meaning |
|---|---|---|
| `--runs` | required | The directory `--pattern` is matched under. |
| `--output` | required | The directory every output file is written to. |
| `--pattern` | `**/report.json` | A glob, relative to `--runs`, that must match one `report.json` per finished cell. The grid puts final cells at `<group>/<head>/final/` and tuning cells at `<group>/<head>/tune/<point>/`, so `*/*/final/report.json` leaves the tuning cells out. The default matches both, and compare then refuses to run. |
| `--replicates` | 1000 | Bootstrap replicates of the test rows, for question 1. |
| `--repeats` | 200 | Scheme T repeats, for question 2. |
| `--seed` | 0 | The analysis seed. A draw depends on it, the dataset's name, and the dataset's test labels, and on nothing else. |
| `--workers` | the CPUs available, at most 8 | Worker processes. 0 analyzes every cell in the calling process. |
| `--margin` | 0.01 | Question 1's margin, δ. |
| `--cost-order` | `l27 sae last both` | The real arms, cheapest first. |

From Python, `compare.compare_runs(runs_dir, output_dir, replicates, seed, *, pattern, repeats,
workers, rules)` does the same and returns what `verdicts.json` holds. `rules` is a
`verdicts.DecisionRules`, whose fields hold every threshold, arm list, and candidate below.

---

## What it reads

A cell is named by its dataset (`data_name`), its arm (`tags.arm`, or its `cell_id` when it has
no tags), and its head. Compare refuses to run when:

- two matched cells have one name, as when the pattern also matches tuning cells;
- cells of one dataset, or seeds of one cell, scored different test rows, by digest or by label.

| Input | Used for |
|---|---|
| each `seed_<seed>.npz`: test labels, outputs, variances, MC dropout outputs, distances | question 1's scores and all of scheme T |
| `report.json` `v_fit_primary` | choosing the single arm `both` is compared with |
| `report.json` `methods.<method>.test` | scheme V, and a check of question 1's point estimates |
| `report.json` `methods.<method>.test_std` | a check of question 1's seed spread |
| `report.json` coverage metric names | the conformal levels |
| `report.json` `identity_bins.path` | the identity file, for shift and identity-bin scores |
| `report.json` `train_label_std` | the regression conformal spread floor; see [Caveats](#caveats) |

The identity file is looked for where `report.json` names it, then at
`<runs>/../identity/<file name>`. A cell whose report names none is analyzed without the scores
that need it, and its `identity file` check passes with a note. A named file found in neither
place fails that check, since the fit wrote one, and so does a file whose test rows or bin
counts differ from the report's; the cell is then analyzed without it.

---

## Question 1: featurization

Question 1 reads the projected head. A dataset and arm score the mean over seeds of the test
primary metric, as `metrics.task_metrics` computes it: MCC for binary and multiclass tasks, F1
for multilabel, and Spearman for regression. Its 95% interval is the 2.5 and 97.5 percentiles
over 1,000 bootstrap replicates of the test rows, stratified by class, label decile, or rarest
positive label, and replicate `b` draws the same rows for every arm of the dataset.

Each comparison is first minus second:

- `sae - last`, `l27 - last`, and `sae - l27`;
- `both - best single`, where the best single arm is `sae` or `last`, whichever has the higher
  `v_fit_primary` on that dataset, never the higher test score;
- every real arm (`sae`, `last`, `l27`, `both`) minus every control (`onehot`, `random`,
  `random_init`).

A comparison's pooled difference is the mean of its per-dataset differences over the datasets
that have both arms, and replicate `b` of the pool averages replicate `b` of every dataset.
Datasets draw their replicates independently, so that is a joint bootstrap. The number of
datasets with a positive difference is reported beside it.

The verdict, in precedence order: "X better" when the pooled interval lies above 0, "Y better"
when it lies below 0, "equivalent" when it lies inside (-δ, +δ), and "unresolved" otherwise.

The seed spread sits beside every score and difference. A score's is the standard deviation over
seeds, with ddof = 1, of the test primary metric of each seed's run. A difference's is the
standard deviation over seeds, with ddof = 1, of the difference between the two arms' runs of
one seed: every arm runs the same seeds, 42 to 46 in the ablation, so the runs pair by seed. A
pooled difference's first averages each seed's difference over the datasets, over the seeds
every one of them has. The spread describes the seeds; no verdict reads it.

The best arm is the real arm with the highest pooled mean over the datasets every real arm has.
The recommendation is the cheapest real arm, in the cost order, whose pooled difference from the
best arm has its lower bound above -δ; the best arm always qualifies.

The controls pass when every real arm's pooled advantage over every control has its interval
above 0. A pair with no dataset in common fails them. Every dataset where a control's point
estimate reaches a real arm's is listed. The README counts such a dataset only once it is
investigated, so while any is listed, the recommendation's headline says it holds only once
those datasets are investigated, and names them. The sensitivity block then gives the three
pairs, `both - best single`, and the recommendation again without every listed dataset, each pool
averaging the same bootstrap replicates over the datasets that remain.

---

## Question 2: uncertainty

Question 2 is decided under scheme T, with scheme V beside it.

**Scheme V** is what each `report.json` already holds: calibration maps fitted on V_fit,
conformal quantiles on V_cal, and test scored once. Published validation splits need not be
exchangeable with test, and when validation proteins sit closer to training than test proteins,
scheme V's conformal sets under-cover.

**Scheme T** carves exchangeable splits out of test. In each of 200 repeats, drawn from the seed
and the dataset alone:

1. the test rows are split in half, stratified, into T_cal and T_eval;
2. T_cal is split in half again, stratified: T_fit fits every calibration map, in V_fit's role,
   and T_conformal every conformal quantile, in V_cal's;
3. T_eval is bootstrap-resampled once, stratified, and the resample is scored.

The methods are those `methods.build_methods` returns, unchanged. A method with nothing fitted
predicts in every repeat as it does on all test rows. A calibrated method, a temperature or for
regression `sigma^2 = a v + b`, is refitted in every repeat on T_fit by `batched_calibration`,
and every repeat is scored from the resample's draw counts by `weighted_metrics`. A score-only
method, one that ranks another method's predictions by a different uncertainty, is scored for
AURC and error detection only, as `report.score_method` scores it. On a few repeats, this path
is tested equal to `build_methods` and `score_method` given the same rows. Every metric of every
dataset, arm, head, and method is reported as its mean over repeats with the 2.5 and 97.5
percentiles.

The reference is `temperature`. The candidates, with their forward passes at inference, are
`sngp_temperature` (1), `ensemble_temperature` (5), and `mc_dropout_temperature` (30). On each
dataset and repeat a candidate has:

- a relative NLL, `(candidate - reference) / |reference|`, with the Gaussian NLL for regression;
- a relative AURC, computed the same way;
- a primary difference, `candidate - plain` on the primary metric.

AURC ranks rows by ascending uncertainty, and rows tied in uncertainty first take their mean
loss, weighted by draw counts in a resample. AURC is linear in the ranked losses, so that makes it
the expectation over every order of the tied rows, where it would otherwise follow the order the
rows are stored in. Ties are common: on a plain regression head, `temperature` gives every row
one sigma, and its AURC is then the mean loss. `metrics.selective_metrics` averages ties the same
way, so a `report.json` the fit writes now does too; see [Caveats](#caveats) for older ones.

Each is averaged over datasets within a repeat, and its interval is the 2.5 and 97.5 percentiles
over repeats. A candidate improves when its pooled relative NLL interval lies below 0, its pooled
relative AURC upper bound is below +0.02, and its pooled primary difference has its lower bound
above -0.01. A candidate is pooled over the datasets that have it, the reference, and `plain`.
EC has no SNGP head, so `sngp_temperature` pools eight datasets where the others pool nine;
every pooled row gives its count.

Where candidates pool over different datasets, each arm's `coverage` names the datasets each
candidate lacks, of those the arm's reference is scored on, with the task type and the reason:
`sngp_temperature` lacks EC, which is multilabel, since no SNGP cell of the arm scored it.

On question 1's recommended arm, the cheapest improving candidate is adopted. When none improves,
the adoption is temperature scaling plus split conformal on an exchangeable calibration split.
Every real arm is assessed, and the arms whose adoption differs from the recommended arm's are
named. Where an arm's adopted candidate lacks a dataset, `coverage` also names what that dataset
would take under the same adoption order: the first later candidate that improves on the arm
and has the dataset, else the fallback. This is reported, not applied; the rule is unchanged.

Scheme V's answer comes from the same rules applied to each report's seed-mean test scores, one
value per dataset, so it has no interval and its conditions compare points.

Question 2 names its methods without their head, so it reads the ablation's heads alone,
`projected` and `sngp`. A comparison that also holds the single-probe run's heads answers it
exactly as a comparison of the ablation's cells alone would.

### Shift and identity

With an identity file, scheme T adds:

- the shift AUROC, with which a method's uncertainty separates test proteins below 0.3 identity
  to training, no hit included, from those at 0.5 or above, pooled over the datasets where every
  repeat defines it;
- conformal coverage and set size or interval width in each identity bin;
- Mondrian conformal by identity bin, with adjacent bins merged, in identity order, until each
  group holds at least 100 T_conformal rows, a short last group joining the one before it.

### Conformal bands

Conformal coverage and set size or interval width are reported under both schemes, beside the
band split conformal guarantees. Scheme T's coverage is a mean over calibration sets, so it sits
beside the marginal band `[level, level + 1 / (n + 1))` for its T_conformal size `n`. Scheme V's
comes from one calibration set, so it sits beside the 95% band of `Beta(k, n + 1 - k)`, with
`k = ceil((n + 1) level)`, for its V_cal size `n`. A Mondrian set has no band, since its groups'
sizes vary.

---

## The single-probe rules

The single-probe run asks which uncertainty method built on one trained probe, read in one
forward pass, best predicts when the probe is wrong, and whether that probe should be the
projected head or a linear one. Its README's "Decision rules, fixed before launch" are
`single_probe.single_probe_verdicts`, with every threshold a field of `SingleProbeRules`.
Compare applies them whenever a matched cell's head is `linear`, `faithful`, `beta_nll`, or
`projected_qclip`, and writes `single_probe.json` and `single_probe.md` beside the other outputs.

**Heads.** `linear` joins the standardized, clamped branches and applies one Linear layer.
`faithful` and `beta_nll` are `projected` with a variance output, for regression only:
`faithful` trains its mean by squared error and its variance, read from the detached trunk, by
Gaussian NLL with the mean detached; `beta_nll` trains both by the NLL weighted by the detached
variance to the power 0.5. `projected_qclip` clips each standardized feature to its 0.1% and
99.9% training quantiles, recorded in `standardizer.npz`, in place of plus or minus 10.

**Methods.** `build_methods` adds these to a head's methods when its seed files carry what
they need; each fits on V_fit alone, and scheme T refits each choice on T_fit in every repeat:

| Method | Built from | Chosen on V_fit (T_fit in scheme T) |
|---|---|---|
| `faithful_temperature`, `beta_nll_temperature` | the head's predicted variance, as `N(mu, a v + b)` | `a` and `b` by Gaussian NLL |
| `temperature_pnorm` | `temperature`'s probabilities, ranked by minus the largest centered logit over the centered logits' p-norm; multiclass tasks only, since a binary task's centered score is constant; score-only | p in {1, 2, 3, 4, 6, 8, 10} by AURC, the earliest on a tie |
| `laplace_temperature` | a last-layer Laplace posterior (`laplace.py`, the seed file's `<split>__laplace_variance`, one column per prior precision): probit-scaled logits, then a temperature; for regression `N(mu, a v + b)` | the prior precision among 13 from 1e-4 to 1e8 and, at each, the temperature or `(a, b)`, together, by NLL |
| `loss_head` | the loss predictor's `log(loss + 1e-6)` (`<split>__loss_prediction`): the uncertainty for classification, whose probabilities are `temperature`'s; for regression `N(mu, a exp(l) + b)` | the temperature, or `a` and `b` |

A cell turns the extras on with `laplace: true`, `loss_head_folds: 5`, and
`pnorm_uncertainty: true`. The loss predictor's training targets are cross-fitted: the training
rows are split into stratified folds by the run seed, a head with the task head's settings
trains on the others and scores each fold's rows, and a `projected` regression head learns the
logarithm of those losses.

**The rules,** per family, classification (binary, multiclass, multilabel) and regression, on
scheme T quantities pooled as question 2 pools them:

- **The head.** `linear` minus `projected` on the plain primary metric, pooled, worded as
  question 1 words a pair.
- **Candidates,** each a head and method, against `temperature` on `projected`. A candidate
  improves when its primary lower bound against plain `projected` is above -0.01, its relative
  NLL interval lies below 0, and its relative AURC upper bound is below 0. `temperature_pnorm`
  has `temperature`'s probabilities, so its primary metric and NLL are `temperature`'s on its
  head, and it improves on the first and third conditions alone.
- **Adoption.** The cheapest tier holding an improving candidate, then the lowest pooled
  relative AURC in it: tier 1 is `temperature` on `linear`, `temperature_pnorm`, and
  `laplace_temperature`; tier 2 `faithful_temperature` and `beta_nll_temperature`; tier 3
  `loss_head`. With none, `temperature` with split conformal, on `linear` when the head verdict
  is "linear better" or "equivalent" and on `projected` otherwise.
- **The ensemble's gain kept.** For every candidate, its pooled relative NLL and AURC over
  `ensemble_temperature`'s on `projected`, over the candidate's datasets: the ratio of the
  pooled means, with the percentiles over repeats of each repeat's ratio.
- **The clamp.** `projected_qclip` minus `projected` on the plain primary metric, with
  `temperature`'s relative NLL and AURC. The quantile clip replaces plus or minus 10 only when
  the primary verdict is "projected_qclip better" and neither relative interval lies above 0.
- **Shift,** each method's pooled shift AUROC, as question 2 reports it.

`loss_head` on a classification task has `temperature`'s probabilities, so its relative NLL is 0
and it cannot meet the NLL condition there; the rule is the README's, and this is its
consequence.

`single_probe.json` holds `rules`, then per arm and family `datasets`, `head` (`per_dataset`,
`pooled`, `verdict`), `reference` and `ceiling`, `candidates` (each with `tier`, `datasets`,
`lacking`, `pooled` summaries of `primary_difference`, `relative_nll`, and `relative_aurc`,
`conditions`, `improves`, `failed`, and `ensemble_share`), `adopted`, and `clamp`; and `shift`.
`single_probe.md` is its tables in words.

---

## Outputs

| File | Contents |
|---|---|
| `verdicts.json` | Everything below the tables draw on, as strict JSON: an undefined number is `null`. |
| `q1_featurization.md` | The recommendation, provisional while a control reaches a real arm, and the controls; the scores table, each score written mean ± seed sd; each comparison per dataset and pooled, with its seed sd and its verdict in words; the recommendation's rows in cost order; and, last, the pairs and the recommendation without the datasets where a control reaches a real arm. |
| `q2_uncertainty.md` | The adoption, every candidate on every arm pooled, with the datasets each candidate lacks and what the adopted candidate's gaps would take, the recommended arm's candidates per dataset, scheme V's answer, the shift AUROCs, and conformal coverage by scheme and by identity bin. |
| `q1_scores.csv` | Per cell: `data_name`, `arm`, `head`, `cell_id`, `primary_metric`, `test_primary` and its `_lower_95` and `_upper_95`, `test_primary_seed_std_ddof1` (the standard deviation over seeds, ddof = 1, of each seed's test primary), `v_fit_primary`, and `report_test_primary`. |
| `q1_pairs.csv` | Per comparison and dataset, then pooled: `section` (`pair`, `control`, or `recommendation`), `comparison`, `data_name` (`pooled` for the pool), `first`, `second`, `difference`, `lower_95`, `upper_95`, `seed_std_ddof1` (the seed spread of the difference), `datasets`, `positive_datasets`, and `verdict`. |
| `q1_sensitivity.csv` | `q1_pairs.csv`'s pair and recommendation rows recomputed without the datasets where a control reaches a real arm, each led by `excluded_datasets`, which is empty when there are none. |
| `q2_scheme_t.csv` | Per cell, method, and metric under scheme T: `score_only`, `mean`, `lower_95`, `upper_95`, `defined` (repeats with a finite value), and `repeats`. |
| `q2_pooled.csv` | Per scheme (`T`, `V`), arm, candidate or reference, and dataset, then pooled: `quantity` (`relative_nll`, `relative_aurc`, `primary_difference`), `mean`, `lower_95`, `upper_95`, `defined`, and on pooled rows `improves` and `failed`. |
| `q2_shift.csv` | Each method's shift AUROC per cell (`level` `dataset`) and pooled (`level` `pooled`), with the datasets included and those left out. |
| `q2_identity_coverage.csv` | Scheme T coverage and size or width per cell, method, set, level, and identity bin, marginal sets and Mondrian (`mondrian_lac`, `mondrian_aps`, `mondrian_conformal`). |
| `conformal_bands.csv` | Per scheme, cell, method, set, and level: `coverage`, its scheme T interval, `size_or_width`, and the band: `calibration_rows`, `rank`, `marginal`, `marginal_lower`, `marginal_upper`, `beta_lower`, `beta_upper`. |
| `checks.csv` | Every check: cell, `check`, `method`, `seed`, `passed`, and `detail` as JSON. |
| `arms.csv`, `methods.csv`, `identity_bins.csv`, `paired_bootstrap.csv`, `summary.md` | The tables `compare` wrote before the decision rules, in the same columns; `summary.md` gains a first line naming both answers. |
| `single_probe.json`, `single_probe.md` | Written only when a cell's head is one only the single-probe run trains: its head verdict, candidates, adoption, ensemble share, and clamp verdict per arm and family; see [The single-probe rules](#the-single-probe-rules). |

A CSV leaves an undefined number empty and writes a float at full precision. Every file is
UTF-8 with LF line ends.

`verdicts.json`:

| Key | Contents |
|---|---|
| `parameters` | `runs`, `pattern`, `seed`, `replicates`, `repeats`, `calibration_half`, `interval_percentiles`, `mondrian_minimum_rows`, `shift` (`novel_below`, `familiar_from`), and `rules`, every field of `DecisionRules`. |
| `cells` | Per cell: its name, `cell_id`, `directory`, `task_type`, `primary_metric`, `seeds`, `test_primary` with its interval, `test_primary_per_seed` (each seed's test primary, by seed) and `test_primary_seed_std_ddof1`, `v_fit_primary`, `test_rows`, `report_test_primary`, `levels`, `train_label_std` and `train_label_std_source`, `identity_file`, `scheme_t_calibration_rows`, and `mondrian_groupings`. |
| `question_1` | `pairs`, each with `first`, `second`, `per_dataset` rows (`difference`, `lower`, `upper`, `seed_std`, `seeds`), `pooled` (`difference`, `lower`, `upper`, `datasets`, `positive_datasets`, `undefined_replicates`, `seed_std`, `seeds`), and `verdict`; the combined arm's pair adds `chosen_on` and `datasets_without_v_fit`. Then `best_arm`, `pooled_means`, `common_datasets`, `missing_real_arms`, `recommendation`, `recommendation_rows` (`arm`, `difference`, `lower`, `upper`, `seed_std`, `qualifies`), `controls` (`passed`, `pairs` with `above_zero`, `unmeasured`, `reached`), and `sensitivity`: `excluded_datasets`, every dataset in `reached`, then `pairs` and the recommendation's keys recomputed without them. `seeds` counts the seeds a seed sd pairs. |
| `question_2` | `reference_method`, `recommended_arm`, `adopted`, `per_arm` (per real arm: `candidates`, `reference`, `adopted`, and `coverage`), and `arms_differing`; `scheme_v`, the same for scheme V; and `shift`, each (arm, head, method)'s pooled shift AUROC. A candidate holds `method`, `arm`, `datasets`, `per_dataset` and `pooled` summaries of `relative_nll`, `relative_aurc`, and `primary_difference` (`mean`, `lower`, `upper`, `defined`, `draws`), `improves`, and `failed`, the conditions it missed. `coverage` holds `datasets`, those the arm's reference is scored on; `lacking`, each candidate's missing datasets (`method`, `data_name`, `task_type`, `reason`); and `substitutes`, what each dataset the adopted candidate lacks would take (`data_name`, `task_type`, `takes`). |
| `conformal` | The rows of `conformal_bands.csv`. |
| `checks` | `total`, `failed`, counts `by_check`, and every failed check in full. |

---

## Checks

Compare checks its own arithmetic against the fit's and records every result in `checks.csv`
and `verdicts.json`. A failed check does not stop the analysis; read `checks.failed` before the
answers.

| Check | Passes when |
|---|---|
| `test primary against report.json` | The head's seed-mean test primary computed here equals the report's, within 1e-6 relative. |
| `seed spread against report.json` | The standard deviation over seeds, ddof = 1, of the head's test primary computed here equals the report's `test_std`, within 1e-6 relative. |
| `identity file` | The report names no identity file, or the file it names is found and has the report's test rows and bin counts. A named file found in neither place fails. |
| `calibration map` | Given the map `build_methods` fitted on repeat 0's T_fit, a calibrated run predicts every test row exactly as that build does, and the map fitted here reaches at most that map's objective. For a method that also chooses a prior precision or a norm order, the objective is the one it chooses by: the calibrated NLL, or for `temperature_pnorm` the AURC (`aurc_gap`). |
| `repeat 0 against score_method` | Repeat 0 scored here, with the maps `build_methods` fitted, matches `report.score_method`, within 1e-5 relative. |

---

## Workers, determinism, and time

Each cell is analyzed in its own worker process, heaviest first, with one BLAS thread per
process. `--workers` defaults to the CPUs the process may use, counting a cgroup CPU quota, and
at most 8. A draw depends only on the seed, the dataset, and its test labels, so the output
depends on the seed alone: one seed writes the same bytes with any number of worker processes,
and `verdicts.json` records no timing. `--workers 0` runs every cell in the calling process,
where BLAS threads can move the last digits. Workers start by spawning, so each imports Protify
afresh, which took about 9 seconds per worker on the machine timed below.

A synthetic grid shaped like the ablation's took 9.0 minutes with the defaults: the nine
datasets at their capped sizes (up to 10,000 test rows, and EC's 538 labels), seven arms, the
SNGP head on the four real arms of the eight datasets that take it, and five seeds, 95 cells in
all, with 1,000 replicates, 200 repeats, and 8 workers, on a laptop with an Intel i9-14900HX
that other jobs shared. At the peak, all processes together held 13.9 GiB, and one worker at
most 1.9 GiB. The workers spent 70 minutes on the 95 cells, 65% of it on EC's seven, about 6.5
minutes each; those start first and run side by side, so no run finishes sooner than one EC
cell. Alone, one EC cell took 3.3 minutes, most of it in scheme T, which scores each calibrated
run as an array of 200 repeats by 1,604 test rows by 538 labels.

---

## Caveats

- **The README governs.** The experiment README's "Decision rules, fixed before launch" are the
  rules; where this page or the code reads them differently, the README wins. Three readings are
  compare's own. Pooling "over the nine datasets" pools over the datasets both arms of a
  comparison have, with the count recorded, so a missing cell shows as a smaller count rather
  than an error. The "better single arm of `sae` and `last` on V_fit" is chosen per dataset, from
  that dataset's `v_fit_primary`, the cheaper arm on a tie. And the README asks that a dataset
  where a control reaches an ESMC arm be investigated before its numbers count: compare lists
  those datasets, calls the recommendation provisional until they are investigated, and answers
  again without them, but the investigation is for a reader.
- **Scheme V's AURC.** Scheme V reads the AURC each `report.json` holds, as the fit computed it. A
  report written before AURC averaged tied rows holds the AURC of the order its rows were stored
  in, and scheme V's relative AURC inherits that until the report is scored again. Scheme T, which
  decides, computes its own.
- **The regression spread floor.** The fit adds 0.001 of the training labels' standard deviation
  to a regression conformal interval's spread, and `report.json` does not record that standard
  deviation. Compare uses `train_label_std` when a report has it, and otherwise the standard
  deviation of the V_fit and V_cal labels, recording which in `cells[].train_label_std_source`.
  The stand-in moves only intervals whose predicted spread is near zero.
- **Multilabel identity bins.** The repeat-0 check against `score_method` skips per-bin scores for
  multilabel tasks, which have no conformal sets, since the per-bin reference AUPRC would cost
  more than the whole scheme.
