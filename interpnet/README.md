---
id: interpnet
kind: model_family
status: active
aliases:
- interpnet
converted: true
depends_on:
- fastplms
- foundry.layers
tests:
  unit:
  - tests
---

# interpnet

InterpNet is the probe behind synth's property oracles. It reads the residue embeddings of a
protein language model, optionally runs a small transformer over them, and scores every residue
against one learned parameter vector per label. A sequence probe averages those per-residue
logits over the residues its attention mask keeps, and returns them as `sims` for per-residue
attribution. A token probe returns them as its predictions.

## Public surface

- `interpnet`: `InterpNetConfig`, `InterpNetForSequenceClassification`,
  `InterpNetForTokenClassification`, and `InterpNetOutput`. Checkpoints load with
  `from_pretrained`, such as the published `Synthyra/*_oracle_interp` probes.
- `interpnet.transformer.Transformer`, the stack, and `interpnet.attention_utils`, which selects
  the attention backend: `flex`, `sdpa`, or `kernels`. Only `kernels` imports FastPLMs, which loads
  FlashAttention through its `kernels.lock` and raises rather than falling back when a supported
  kernel fails to load.
- `interpnet.attention_logits`, the token-parameter readouts, and `interpnet.losses`.
- `interpnet.mlp`, the feed-forward block of each transformer layer, which is
  `foundry.layers.feedforward`'s with a biased layer normalization, `nn.Linear`'s default
  initialization, and a bias in both projections under `use_bias`.

Loading under Transformers 5 depends on two things the tests guard: the heads declare that they
tie no weights (`all_tied_weights_keys`), and `_init_weights` rebuilds the rotary tables, which no
checkpoint holds.

## Who imports it

- `property_oracles`: `InterpNetOracle` loads each oracle's probe.
- `serving_api`: `synth.services.protify.interpnet_probe` registers InterpNet as Protify's
  `interpnet` probe type, and the Atlas oracle runtime loads probes through `property_oracles`.
- `sequence_gradient`: `StandaloneInterpNetOracle` and `workflows.oracles` load probes as design
  critics.

Each of these projections vendors this directory through `github.includes`: at `src/interpnet/` in
synth, and at the repository root in ProteinDesignEnvironment and Oracles. `ws publish` vendors
the foundry modules InterpNet imports beside it. Edit it here, not in a projection. The decision is
[2026-09-23_interpnet_model_family](../../docs/decisions/models/interpnet/2026-09-23_interpnet_model_family.md).

## Tests

`ws test --projects interpnet` runs `tests/` on CPU: loading round trips, the tied-weights
declaration, the rebuilt rotary tables, kernel loading through FastPLMs' lock, the readouts, and a
forward and backward pass. `--include gpu` adds a bfloat16 pass on CUDA.
