# Tap probing

Tap probing asks where inside a protein language model a property becomes linearly readable. One forward pass of the
model yields every token-wise state of every block, and a probe is trained on each state independently: the residual
stream, the attention input, the queries, keys and values before and after the rotary embedding, the attention context
and output, and the inputs, gates and activations of the MLP. The implementation is in
[`interpretability/`](../src/protify/interpretability/) and runs as `python -m protify.interpretability`.

The default model is ESMC-300 (30 blocks, width 960, 15 heads of 64, SwiGLU width 2560). `--model ESMC-600` (36
blocks, width 1152, 18 heads, SwiGLU width 3072) and `--model ESMC-6B` (80 blocks, width 2560, 40 heads, SwiGLU width
6912) run the same way, since the hooks read every width from the model, and so do Protify's two baseline encodings
(below). The concept tasks come from the `Synthyra/protein_concept_probing` dataset (22 configurations: ProtParam
values, InterPro entries, GO terms, localization, and per-residue secondary structure, topology, peptides, sites and
modifications); a protein-protein interaction task (below) is probed apart from them.

## Sites

`--sites` takes `all`, `residual`, or comma-separated names. `embedding` and `final_norm` occur once; every other site
occurs in every block.

| Site | State | Width |
|---|---|---|
| `embedding` | token embeddings, the residual stream before block 0 | d |
| `attn_ln` | attention input after the block's layer norm | d |
| `q`, `k` | queries and keys after their layer norms, before the rotary embedding | d |
| `v` | values | d |
| `q_rot`, `k_rot` | queries and keys after the rotary embedding | d |
| `attn_context` | attention-weighted values, heads concatenated, before the output projection | d |
| `attn_out` | the output projection, before it is scaled and added to the residual stream | d |
| `resid_mid` | residual stream between attention and MLP | d |
| `mlp_ln` | MLP input after its layer norm | d |
| `mlp_gate`, `mlp_up` | the two halves of the SwiGLU input projection | f |
| `mlp_act` | SiLU(gate) times up, the input of the down projection | f |
| `mlp_out` | the down projection, before it is scaled and added to the residual stream | d |
| `resid_post` | residual stream after the block | d |
| `final_norm` | the last layer norm's output | d |

The six head-split sites (`q`, `k`, `q_rot`, `k_rot`, `v`, `attn_context`) also get one probe per head unless
`--no-heads` is given. For ESMC-300, `all` is 452 states and 2,700 per-head probes.

## Workflow

```bash
# 1. pooled stores of the protein tasks, and the control stores of their test proteins
python -m protify.interpretability embed --root RUN --controls
# 2. a grid of learning rates and weight decays, scored on validation only, for each batch size to compare
python -m protify.interpretability sweep --root RUN --label p_b256 --tasks protein --batch-size 256 --lr 1e-4 1e-3 1e-2 --wd 0 0.1
python -m protify.interpretability sweep --root RUN --label p_b1024 --tasks protein --batch-size 1024 --lr 1e-4 1e-3 1e-2 --wd 0 0.1
# 3. one global setting per loss: cross entropy for the classification tasks, squared error for ProtParam
python -m protify.interpretability select --root RUN --labels p_b256 p_b1024 --loss classification
python -m protify.interpretability select --root RUN --labels p_b256 p_b1024 --loss regression
# 4. every probe of every task under its setting, with the controls, scored on test
python -m protify.interpretability train --root RUN --label linear --tasks protein --batch-size 256 --lr 1e-3 --wd 0
# 5. figures and summary tables, over seed replicates and against a randomly initialized model
python -m protify.interpretability figures --roots RUN RUN_2 RUN_3 --random-roots RANDOM RANDOM_2 RANDOM_3 --out FIGURES
# 6. trends across models, the random-init model and the baselines
python -m protify.interpretability trends --models ESMC-300=RUN,RUN_2,RUN_3 ESMC-600=RUN_600 --random "ESMC-300 random init=RANDOM" --out FIGURES/trends
```

Residue tasks follow the same steps with `--tasks residue` and `--train-tokens` in place of `--batch-size`.
`--pilot` on `embed`, `train`, `sweep` or `phospho` times the first steps with `foundry.pilot`, writes the report under
`--root`, and stops.

## Storage and throughput

**Protein tasks** read pooled stores. `embed` runs each sequence once and writes, for every site and layer, the mean and
population variance over the CLS, residue and EOS tokens, as one bf16 memory map per site of shape (proteins, layers,
2 x width). A protein task then trains every probe of a site at once from rows gathered out of that map.

**Residue tasks** never store token states. Each batch runs the model once with hooks; as each state arrives, the hook
gathers its residue rows and every probe that reads that site and layer takes its gradient there, so memory holds one
state at a time. One optimizer step follows each forward pass.

**Probe banks.** All probes of a site live in one `ProbeBank` with a group axis: linear or MLP probes with
hand-written gradients, per-group AdamW, standardized inputs and early stopping on validation loss with the best
weights kept. Linear probes that read the same input share one bf16 matrix multiplication: the standardization is
folded into the weights and every grid setting becomes more output columns, so a sweep of K settings costs about one
pass over the data.

**Large models.** A protein or pair probe holds a site's pooled rows on the device; when they would pass
`--site-budget-gb` (8 by default), the site trains in chunks of layers. Each chunk keeps the initial weights its probes
have in the whole site, and probes of different layers train independently, so chunks change memory and nothing else.
`--no-untrained` on `embed` and `train` leaves the untrained model out of the controls, which spares a large model's
untrained stores and forward passes.

`tokens` writes token-wise states of chosen sites for a list of sequences, for analyses that need them.

## Global settings

`sweep` trains every probe under every learning rate and weight decay and scores each on the validation split.
`select` ranks the settings by mean shortfall: for each probe, the gap between its validation score under a setting
and under its own best setting, averaged over probes within a task and probe family, then over families and tasks.
Among the settings within `--tolerance` (default 0.005) of the lowest shortfall, the one whose label took the least
time wins, so a larger batch is preferred when it costs little. The choice and the full ranking are written to
`RUN/sweeps/selection_<labels>.json`. Test splits are read only by `train`.

`--loss classification` or `--loss regression` ranks the settings over the tasks of one loss only, written to
`selection_<labels>_<loss>.json`. Squared error on standardized targets wants much smaller steps than cross entropy:
for ESMC-300, the learning rate that serves the classification tasks best leaves ProtParam's probes 0.29 below their
best validation score on average, so ProtParam, the one regression task, takes its own setting.

## Controls and scores

With controls (the default for `train`), every trained probe is also scored, at test time, under:

| Control | What changes |
|---|---|
| untrained model | every module of the model reset to PyTorch's default initialization |
| scrambled sequence | every residue after the first shuffled; ProtParam is recomputed on the scrambled sequence, residue labels and multilabel vectors are shuffled within the protein by an independent draw, and a single class (localization, metal ion binding) is kept |
| untrained probe | the probe's initial weights |
| L2-normalized | each token state scaled to unit norm |
| mean state | the mean state of a batch of 8 proteins (position by position for residue tasks), with the batch's mean or most common label |
| Gaussian input | standard normal inputs |

A probe standardizes its inputs with the training split's moments, so an input whose scale departs from those moments,
as L2-normalized residue states do, reaches it near a constant. Scores are the Matthews correlation coefficient (cosine
similarity for the ProtParam regression), on the full test split and on 7 random halves of it, so every score has a
spread. Residue test passes run in batches that fill the token budget; with controls, a batch holds whole groups of 8
consecutive test proteins, the groups the mean control averages. Results go to `RUN/results/<label>/<task>/`:
`scores.parquet` (one row per probe, condition and resample), `elementwise.parquet` (per label),
`training.parquet` (steps, epochs and selected setting of each probe), `validation.parquet` (each kept probe's score
on the whole validation split, in the test metric, and its validation loss), `probes.safetensors` (the selected
weights) and `run.json` (the task, the settings and the elapsed time; its presence marks a finished task).

**Choosing a best state.** Wherever a figure or table names the best state of a task, a site or a family of sites,
the state is the one with the highest validation score, and its test score is reported. The highest test score over
hundreds of probes would overstate small test sets and favor the families with more states, and validation loss is no
guide across states: probes on stronger features stop at more confident, worse calibrated outputs, so the lowest
loss often belongs to a weak early state. `validate --root RUN --label LABEL` adds `validation.parquet` to runs trained
before `train` wrote it, rebuilding each probe from `probes.safetensors`; it checks every recomputed validation loss
against the one training recorded.

## Phosphomimetic scanning

`phospho` mutates every S, T and Y of randomly drawn phosphoproteins to A, G and E. A mutant counts as active when
it keeps an annotated site's function (the wild type, or E at an annotated phosphosite), inactive when it removes it
(A or G at an annotated phosphosite), and unknown at any other S, T or Y. For every site and layer, a three-class
logistic regression on pooled mean states (float32 forward passes, a stratified 70:30 split per protein, scikit-learn's
default objective with C = 1) predicts the class; the same regression on shuffled labels and on the untrained model's
states are the controls. Each regression is fit twice, on the raw means and on means standardized over the protein's
training mutants, and solved exactly by Newton's method in the span of the training rows (a protein has far fewer
mutants than a site has columns). `--tf32` allows TF32 products in the forward passes, at twice the speed.
`tests.parquet` holds, per input treatment, site and layer, the mean MCC under each condition and a Wilcoxon
signed-rank test of true against shuffled labels, paired by protein, with a Bonferroni correction over every site and
layer. `--split-seed` redraws each protein's split and label shuffle while `--data-seed` keeps the same proteins, so
replicates (with their own `--seed` for the untrained model) can be averaged protein by protein before testing.

## A randomly initialized model

`--random-init` on `embed`, `train`, `phospho` or `tokens` resets every module of the model to PyTorch's default
initialization, seeded by `--seed`, as the untrained-model control does. Run under its own `--root`, the whole analysis
then repeats with probes trained on the random model's states, which shows how much of each score the architecture and
the tokens give without training:

```bash
python -m protify.interpretability embed --root RANDOM --random-init
python -m protify.interpretability train --root RANDOM --random-init --no-controls --tasks protein --batch-size 256 --lr 3e-2 --wd 0
```

## Baseline encodings

`--model OneHot-Protein` and `--model Random` probe Protify's two baseline encodings with the same stores, probes and
scores. A baseline's one state is its token rows (`embedding`): a one-hot row over 26 residue symbols, CLS and EOS (28
columns), whose pooled mean is the composition and the length; or a standard normal row of width 960 drawn from the
sequence's SHA-256, whose pooled statistics carry the length alone. The rows come from the embedding engine behind
Protify's control bases, so they are those bases' rows. A baseline has no weights, so it runs without `--random-init`
and without the untrained-model controls:

```bash
python -m protify.interpretability embed --root ONEHOT --model OneHot-Protein --tasks protein,ppi
python -m protify.interpretability train --root ONEHOT --model OneHot-Protein --no-controls --tasks protein --batch-size 256 --lr 3e-2 --wd 0
```

## Protein-protein interaction

`--tasks ppi` adds a protein-protein interaction task, apart from the 22 concepts: the Bernett gold-standard pairs
(`--pair-repo`, by default `Synthyra/bernett_gold_ppi`; human pairs, half interacting, no protein in two splits, and
each protein's count of negative pairs close to its count of positive ones). Proteins are cut to their first 2,046
residues, ESMC's context less CLS and EOS. `embed --tasks ppi` writes their stores (`stores/ppi_*`, controls included),
and a pair probe reads one row per pair from the two partners' pooled states, each standardized with the training
proteins' moments: [z_a + z_b | z_a * z_b], which a swap of the partners leaves unchanged. The sum lets a linear probe
score each partner on its own, and the product lets it weigh the columns both partners share. Rows are built batch by
batch, so no pair-by-layer tensor is ever held. Everything else follows the protein tasks, with each pair keeping its
label under scrambled partners, and `ranking.parquet` adds each probe's ROC AUC and average precision on the test pairs.

```bash
python -m protify.interpretability embed --root RUN --tasks ppi --controls
python -m protify.interpretability sweep --root RUN --label ppi_b1024 --tasks ppi --batch-size 1024 --lr 1e-4 1e-3 1e-2 --wd 0 0.1
python -m protify.interpretability train --root RUN --label linear --tasks ppi --batch-size 1024 --lr 1e-3 --wd 0
```

## Seed replicates

A replicate repeats the analysis under another `--seed` in its own root: the probes' initialization and batch order,
the scrambled sequences, the untrained-model control and, with `--random-init`, the random model itself all follow the
seed. The trained model's pooled store does not depend on the seed, so replicates can link it rather than write it
again; their control stores do. `figures --roots` averages every score over the replicates, and reports both the
standard deviation of a probe's full-test score over replicates and its spread over test halves.

## Variants

Two options change what the probes read, and each variant runs under its own `--root` or `--label`:

- `--normalize` on `embed` and `train` scales every token state to unit L2 norm along its width before pooling
  (protein tasks) or probing (residue tasks), the controls included; `train` refuses a store whose manifest disagrees.
  The `l2_normalized` control then repeats the original condition.
- `--pooling mean` on `train` makes protein probes read the pooled mean alone (width w, or d_h per head) instead of the
  mean and variance (2w).

`compare --variants NAME=ROOT[:LABEL][,ROOT[:LABEL]...] ...` sets variants against the first one: `variants.csv`
holds each task's best residual-stream and any-state scores and its mean over full-width probes per variant,
`variants_depth.png` overlays their residual streams, and `variants_maps_<name>.png` and `variants_summary_<name>.png`
map each variant's difference from the first by site and block.

```bash
python -m protify.interpretability embed --root NORM --controls --normalize
python -m protify.interpretability train --root NORM --normalize --tasks protein --batch-size 256 --lr 3e-2 --wd 0
python -m protify.interpretability train --root RUN --label linear_mean --pooling mean --tasks protein --batch-size 256 --lr 3e-2 --wd 0
python -m protify.interpretability compare --variants raw=RUN normalized=NORM mean=RUN:linear_mean --out FIGURES/variants
```

## Principal components

`pca` projects each protein's pooled mean state at chosen sites and layers onto its first two principal components,
grouping proteins by their exact label set (the `--top` most frequent sets are named, the rest are "other").
Scrambled copies of the test proteins are projected onto the same components. `figures` draws one panel per layer.

## Figures

`figures` reads the tables above, from one root per replicate (`--roots`) and optionally from the replicates of a
randomly initialized model (`--random-roots`), and writes 300 dpi PNGs and summary tables:

| File | Shows |
|---|---|
| `depth_curves.png` | each task's residual stream along depth, against the random-init model and the controls |
| `selected_tasks.png` | five tasks' residual streams with the best depth and its spread over replicates |
| `controls.png` | all twenty conditions (model, probe and input) along the residual stream |
| `site_maps.png` | every site by block, relative to each task's highest score |
| `site_gaps.png` | each site's best score minus the best residual-stream score, and the block each best probe reads |
| `site_summary.png` | the same gap, one dot per task for each site, sites ordered by their median |
| `site_families.png` | attention and MLP sites along the blocks against the block output |
| `rotation.png` | rotated minus unrotated queries and keys, full width and per head |
| `heads.png` | every head of the head-split sites by block |
| `best_states.png` | the best residual-stream, attention, MLP and single-head probe of each task |
| `random_init_maps.png`, `random_init_summary.png` | trained minus random-init scores by site and block, and by site |
| `replicate_spread.png` | the spread of a probe's score over replicates against its spread over test halves |
| `prot_param_targets.png`, `phospho.png`, `pca_*.png`, `sweeps_*.png` | ProtParam targets, phosphomimetic tests, principal components, sweeps |

`summary.parquet` (and `summary_random_init.parquet`), `best_states.csv`, `site_bests.csv`, `random_init.csv`,
`replicate_spread.csv`, `phospho_tests.parquet` and `phospho_replicates.csv` hold the numbers behind them.
`--interventions` names the phosphomimetic replicates, whose scores are averaged per protein before testing.

## Trends across models

`trends` sets several models side by side: each `--models NAME=ROOT[,ROOT...]` names one model's replicate roots,
`--random` the randomly initialized model's and `--baselines` the baseline encodings'. Depth becomes relative, 0 at the
token embedding and 1 at the final norm, so models of different depth share one axis; a model's blocks come from its
store manifest, or else from the deepest block its probes read. A task's onset is the first depth at which its
residual stream comes within 5% of its highest score, and tasks are grouped into six kinds of concept (composition,
secondary structure, sorting and topology, InterPro membership, function, residue sites and modifications).

| File | Shows |
|---|---|
| `emergence.png` | each task's residual stream along relative depth as a share of its highest score, one panel per model, with the onset and the best state chosen on validation |
| `state_rankings.png` | each site's best state minus the best residual-stream state, median and quartiles over tasks, per model |
| `controls_summary.png` | each task's best residual-stream score for every model, the random-init model and the baselines, with the first model's controls |
| `model_size.png` | each task's best score and the relative depth of its best stream state, joined across models |
| `ppi.png` | the pair task along relative depth (MCC and ROC AUC) for every model and control, and by site |

`stream_profiles.csv`, `category_profiles.csv`, `site_rankings.csv`, `best_states_by_model.csv`,
`pair_best_states.csv` and `pair_ranking.csv` hold the numbers behind them.

```bash
python -m protify.interpretability trends --models ESMC-300=RUN,RUN_2,RUN_3 ESMC-600=RUN_600 --random "ESMC-300 random init=RANDOM" \
    --baselines OneHot-Protein=ONE_HOT Random=RANDOM_ROWS --out FIGURES/trends
```
