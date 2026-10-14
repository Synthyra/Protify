---
id: foundry.gpu_memory
kind: library_module
status: active
aliases: [foundry.gpu_memory]
---

# foundry.gpu_memory

The page cache that a GH200 keeps in its GPU's memory, read so a budget does not count it as used. Standard library
only. [The machine's record](../../environments/gh200_suite/README.md#read-before-you-run-anything-here) has the
measurement.

| Name | Does |
|---|---|
| `hbm_page_cache_bytes(node=HBM_NODE)` | Bytes of file cache on node 1 of a GH200 (aarch64, a node with no CPUs); 0 on any other host; raises when a GH200's `meminfo` has no `FilePages` line |
| `usable_free_bytes(free, total, allocated, page_cache)` | `min(free + page_cache, total - allocated)`, floored at 0: what an allocation can take, as the kernel drops the cache on demand |

Consumers: the batch-fill sizing of `foundry.embedding.job` and the memory ceiling of `queue_arms.py` in the Atlas v2
project.
