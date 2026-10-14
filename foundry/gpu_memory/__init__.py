"""What a GPU's memory really holds on a GH200, where the kernel's page cache sits in the device's memory.

A GH200's 96 GB of HBM is Linux NUMA node 1, a node with no CPUs, and the kernel fills it with cached file pages after
large writes (93.1 of 94.5 GiB on 2026-10-01, with no process on the GPU). `nvidia-smi` and `torch.cuda.mem_get_info`
count those pages as used, yet the kernel drops them within seconds when a CUDA allocation needs the room. Anything that
sizes a job from "free" device memory must add them back, or it reads a GH200 as full.
"""

from __future__ import annotations

import platform

from pathlib import Path


HBM_NODE = Path("/sys/devices/system/node/node1")  # A GH200's HBM as Linux sees it: a NUMA node with no CPUs.


def hbm_page_cache_bytes(node: Path = HBM_NODE) -> int:
    """Bytes of file page cache on a GH200's HBM node; 0 on any other host.

    Only an aarch64 host whose ``node`` exists and has an empty ``cpulist`` is a GH200: an x86-64 host, a host with one
    NUMA node, and a node with CPUs (a socket's ordinary memory) all count nothing. A GH200 whose ``meminfo`` has no
    ``FilePages`` line raises, since a guess would put cache the kernel cannot drop into a budget.
    """
    if platform.machine() != "aarch64" or not node.is_dir() or (node / "cpulist").read_text().strip():
        return 0
    for line in (node / "meminfo").read_text().splitlines():
        if "FilePages:" in line:  # "Node 1 FilePages:      97612345 kB"
            return int(line.split()[-2]) * 1024
    raise RuntimeError(f"{node / 'meminfo'} has no FilePages line, so the page cache of the HBM node is unknown.")


def usable_free_bytes(free: int, total: int, allocated: int, page_cache: int) -> int:
    """Memory a job can allocate: the device's free bytes plus the page cache the kernel returns, within what is left.

    ``free`` and ``total`` are `torch.cuda.mem_get_info`, ``allocated`` is this process's tensors
    (`torch.cuda.memory_allocated`) and ``page_cache`` is `hbm_page_cache_bytes`. The sum cannot pass ``total`` minus
    ``allocated``, which holds the figure to the device when the two readings, taken a moment apart, overlap.
    """
    return max(min(free + page_cache, total - allocated), 0)
