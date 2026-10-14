---
id: foundry.checkpoint_loading
kind: library_module
status: active
aliases: [foundry.checkpoint_loading]
---
# foundry.checkpoint_loading

Test support for one failure: a checkpoint that loads without an error into a model whose buffers
it does not hold, so the model computes with whatever memory those buffers were allocated in.

Transformers 5 builds a model on the meta device, copies in the tensors the checkpoint holds, and
allocates every other parameter and buffer as uninitialized memory for the model's `_init_weights`
to fill. A buffer registered with `persistent=False`, such as a rotary frequency table, is in no
checkpoint. The default `_init_weights` rebuilds only Transformers' own rotary modules, so a model
that defines its own has to rebuild the table itself. Atlas did not, and every protein feature it
computed came out NaN.

| Name | Does |
|---|---|
| `uninitialized_memory_as_nan()` | Context manager: every allocation inside it holds NaN, or the maximum for integers. An unfilled tensor then fails every comparison, rather than passing when the allocator recycles a block that happened to hold the right value |
| `assert_unsaved_buffers_match(model, checkpoint_keys, build_fresh)` | Asserts that each buffer of the loaded model that the checkpoint lacks equals, exactly, the one a directly built model holds. Returns how many there were, so a test can assert the buffers under test exist |

A test saves a small model, or reads the tensor names of a cached checkpoint, loads it with
`from_pretrained` inside `uninitialized_memory_as_nan()`, and compares:

```python
with uninitialized_memory_as_nan():
    loaded = AtlasModel.from_pretrained(directory)
unsaved = assert_unsaved_buffers_match(loaded, saved.state_dict(), lambda: AtlasModel(loaded.config))
assert unsaved, "the rotary tables are the buffers under test"
```

Consumers: the test suites of atlas, serving_api, sequence_gradient,
property_lowcode, and dsm. A consumer that publishes a projection lists this module under
`github.includes`, so `ws publish` vendors it beside the tests. FastPLMs keeps its own
`test_pretrained_buffers.py`, because its packages are published standalone.

Tests: `tests/tier1_unit/test_checkpoint_loading.py`, including a Transformers model whose rotary
table the load leaves unfilled. They need torch and Transformers, so the tooling environment skips
them; `ws test --projects foundry` runs them under the research interpreter.
