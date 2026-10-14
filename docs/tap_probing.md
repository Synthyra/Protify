# Tap probing

Tap probing asks where inside a protein language model a property becomes linearly readable. One forward pass of the
model yields every token-wise state of every block, and a probe is trained on each state independently: the residual
stream, the attention input, the queries, keys and values before and after the rotary embedding, the attention context
and output, and the inputs, gates and activations of the MLP. The implementation is in
[`interpretability/`](../src/protify/interpretability/) and runs as `python -m protify.interpretability`.

The default model is ESMC-300 (30 blocks, width 960, 15 heads of 64, SwiGLU width 2560). The concept tasks come from
the `Synthyra/protein_concept_probing` dataset (22 configurations: ProtParam values, InterPro entries, GO terms,
localization, and per-residue secondary structure, topology, peptides, sites and modifications).

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
# 3. one global setting
python -m protify.interpretability select --root RUN --labels p_b256 p_b1024
# 4. every probe of every task under that setting, with the controls, scored on test
python -m protify.interpretability train --root RUN --label linear --tasks protein --batch-size 256 --lr 1e-3 --wd 0
# 5. figures and summary tables
python -m protify.interpretability figures --root RUN --label linear --out FIGURES
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

`tokens` writes token-wise states of chosen sites for a list of sequences, for analyses that need them.

## Global settings

`sweep` trains every probe under every learning rate and weight decay and scores each on the validation split.
`select` ranks the settings by mean shortfall: for each probe, the gap between its validation score under a setting
and under its own best setting, averaged over probes within a task and probe family, then over families and tasks.
Among the settings within `--tolerance` (default 0.005) of the lowest shortfall, the one whose label took the least
time wins, so a larger batch is preferred when it costs little. The choice and the full ranking are written to
`RUN/sweeps/selection_<labels>.json`. Test splits are read only by `train`.

## Controls and scores

With controls (the default for `train`), every trained probe is also scored, at test time, under:

| Control | What changes |
|---|---|
| untrained model | every module of the model reset to PyTorch's default initialization |
| scrambled sequence | every residue after the first shuffled (residue labels follow their residues) |
| untrained probe | the probe's initial weights |
| L2-normalized | each token state scaled to unit norm |
| mean state | the mean state of a batch of 8 proteins (position by position for residue tasks), with the batch's mean or most common label |
| Gaussian input | standard normal inputs |

A probe standardizes its inputs with the training split's moments, so an input whose scale departs from those moments,
as L2-normalized residue states do, reaches it near a constant. Scores are the Matthews correlation coefficient (cosine
similarity for the ProtParam regression), on the full test split and on 7 random halves of it, so every score has a
spread. Without controls (`--no-controls`), residue tasks are scored in batches that fill the token budget. Results go to `RUN/results/<label>/<task>/`:
`scores.parquet` (one row per probe, condition and resample), `elementwise.parquet` (per label),
`training.parquet` (steps, epochs and selected setting of each probe), `probes.safetensors` (the selected weights) and
`run.json` (the task, the settings and the elapsed time; its presence marks a finished task).

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
layer.

## A randomly initialized model

`--random-init` on `embed`, `train`, `phospho` or `tokens` resets every module of the model to PyTorch's default
initialization, seeded by `--seed`, as the untrained-model control does. Run under its own `--root`, the whole analysis
then repeats with probes trained on the random model's states, which shows how much of each score the architecture and
the tokens give without training:

```bash
python -m protify.interpretability embed --root RANDOM --random-init
python -m protify.interpretability train --root RANDOM --random-init --no-controls --tasks protein --batch-size 256 --lr 3e-2 --wd 0
```

## Principal components

`pca` projects each protein's pooled mean state at chosen sites and layers onto its first two principal components,
grouping proteins by their exact label set (the `--top` most frequent sets are named, the rest are "other").
Scrambled copies of the test proteins are projected onto the same components. `figures` draws one panel per layer.

## Figures

`figures` reads the tables above and writes 300 dpi PNGs and summary tables: residual-stream curves along depth with
one standard deviation over the test halves and the untrained-model and scrambled controls, site by block maps of
every task relative to its best probe, the best state of each task by family, the block at which each site peaks,
every control along depth, per-head maps, the ProtParam targets one by one, the sweep curves, the phosphomimetic
results and the principal component panels. `best_states.csv` and `summary.parquet` hold the numbers behind them.
