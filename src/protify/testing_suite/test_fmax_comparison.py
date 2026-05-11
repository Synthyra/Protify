"""
Compare Protify's max_metrics (micro, grid search) against TorchDrug's
count_f1_max (macro over samples, exact threshold search) across many
random multi-label configurations.

Not a pass/fail test -- prints statistics on the divergence between the two.
Run with:  py -m testing_suite.test_fmax_comparison
"""
import os
import sys
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from metrics import max_metrics


def count_f1_max(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TorchDrug Fmax: macro-averaged P/R across samples, exact threshold search."""
    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = order + torch.arange(order.shape[0], device=order.device).unsqueeze(1) * order.shape[1]
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - \
                    torch.where(is_start, torch.zeros_like(precision), precision[all_order - 1])
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - \
                 torch.where(is_start, torch.zeros_like(recall), recall[all_order - 1])
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


def run_comparison(n_trials: int = 100_000, seed: int = 42, sparsity_range: tuple = (0.1, 0.9), label: str = "default"):
    rng = np.random.RandomState(seed)
    diffs = []
    protify_vals = []
    torchdrug_vals = []
    protify_time = 0.0
    torchdrug_time = 0.0

    for i in tqdm(range(n_trials)):
        B = rng.randint(2, 32)
        N = rng.randint(2, 32)
        # Random predictions (sigmoid-like range 0-1)
        pred = torch.tensor(rng.rand(B, N), dtype=torch.float32)
        # Random binary labels with variable sparsity
        sparsity = rng.uniform(sparsity_range[0], sparsity_range[1])
        target = torch.tensor((rng.rand(B, N) < sparsity).astype(np.float32))

        # Skip degenerate cases (all 0 or all 1 labels)
        if target.sum() == 0 or target.sum() == B * N:
            continue

        # Protify: flatten and run max_metrics
        t0 = time.perf_counter()
        f1_protify, _, _, _ = max_metrics(pred.flatten(), target.flatten().int())
        protify_time += time.perf_counter() - t0

        # TorchDrug
        t0 = time.perf_counter()
        f1_torchdrug = count_f1_max(pred, target).item()
        torchdrug_time += time.perf_counter() - t0

        diff = abs(f1_protify - f1_torchdrug)
        diffs.append(diff)
        protify_vals.append(f1_protify)
        torchdrug_vals.append(f1_torchdrug)

    diffs = np.array(diffs)
    protify_vals = np.array(protify_vals)
    torchdrug_vals = np.array(torchdrug_vals)

    print(f"\n{'='*60}")
    print(f"Fmax Comparison: {label}")
    print(f"Sparsity range (positive rate): {sparsity_range}")
    print(f"{'='*60}")
    print(f"Trials:            {len(diffs):,}")
    print(f"")
    print(f"--- Absolute difference ---")
    print(f"  Mean:            {diffs.mean():.6f}")
    print(f"  Median:          {np.median(diffs):.6f}")
    print(f"  Std:             {diffs.std():.6f}")
    print(f"  Max:             {diffs.max():.6f}")
    print(f"  95th pctile:     {np.percentile(diffs, 95):.6f}")
    print(f"  99th pctile:     {np.percentile(diffs, 99):.6f}")
    print(f"")
    print(f"--- Agreement ---")
    print(f"  |diff| < 0.001:  {(diffs < 0.001).mean()*100:.1f}%")
    print(f"  |diff| < 0.01:   {(diffs < 0.01).mean()*100:.1f}%")
    print(f"  |diff| < 0.05:   {(diffs < 0.05).mean()*100:.1f}%")
    print(f"  |diff| >= 0.05:  {(diffs >= 0.05).mean()*100:.1f}%")
    print(f"")
    print(f"--- Which is higher ---")
    print(f"  Protify > TD:    {(protify_vals > torchdrug_vals + 1e-8).mean()*100:.1f}%")
    print(f"  TD > Protify:    {(torchdrug_vals > protify_vals + 1e-8).mean()*100:.1f}%")
    print(f"  Equal (1e-8):    {(diffs < 1e-8).mean()*100:.1f}%")
    print(f"")
    print(f"--- Mean F1 values ---")
    print(f"  Protify:         {protify_vals.mean():.6f}")
    print(f"  TorchDrug:       {torchdrug_vals.mean():.6f}")
    print(f"")
    print(f"--- Timing ---")
    print(f"  Protify total:   {protify_time:.2f}s ({protify_time/len(diffs)*1000:.3f}ms/trial)")
    print(f"  TorchDrug total: {torchdrug_time:.2f}s ({torchdrug_time/len(diffs)*1000:.3f}ms/trial)")

    # Show worst-case examples
    worst_idx = np.argsort(diffs)[-5:][::-1]
    print(f"\n--- Top 5 worst disagreements ---")
    for rank, idx in enumerate(worst_idx):
        print(f"  #{rank+1}: Protify={protify_vals[idx]:.6f}  TorchDrug={torchdrug_vals[idx]:.6f}  diff={diffs[idx]:.6f}")

    return protify_vals, torchdrug_vals, diffs, label


def plot_histograms(results: list, save_path: str = "fmax_comparison.png"):
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (protify_vals, torchdrug_vals, diffs, label) in enumerate(results):
        # Left: overlapping F1 distributions
        ax = axes[i, 0]
        bins = np.linspace(0, 1, 60)
        ax.hist(protify_vals, bins=bins, alpha=0.5, label="Protify (micro)", color="tab:blue", edgecolor="white", linewidth=0.3)
        ax.hist(torchdrug_vals, bins=bins, alpha=0.5, label="TorchDrug (macro)", color="tab:orange", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Fmax")
        ax.set_ylabel("Count")
        ax.set_title(label)
        ax.legend()

        # Right: difference distribution
        ax = axes[i, 1]
        ax.hist(diffs, bins=60, alpha=0.7, color="tab:red", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("|Protify - TorchDrug|")
        ax.set_ylabel("Count")
        ax.set_title(f"Absolute difference ({label})")
        ax.axvline(np.median(diffs), color="black", linestyle="--", linewidth=1, label=f"median={np.median(diffs):.4f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved plot to {save_path}")
    plt.close()


if __name__ == "__main__":
    all_results = []
    all_results.append(run_comparison(n_trials=100_000, seed=42, sparsity_range=(0.1, 0.9), label="Uniform sparsity (10-90% positive)"))
    all_results.append(run_comparison(n_trials=100_000, seed=43, sparsity_range=(0.01, 0.1), label="Sparse labels (1-10% positive)"))
    all_results.append(run_comparison(n_trials=100_000, seed=44, sparsity_range=(0.001, 0.01), label="Very sparse labels (0.1-1% positive)"))
    plot_histograms(all_results)
