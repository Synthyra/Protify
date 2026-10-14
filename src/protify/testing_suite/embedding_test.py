"""
Embedding Test Suite CLI

Tests embedding quality by sampling sequences from EC dataset (default),
embedding them with various pooling methods, and reporting statistics
on distribution, NaNs, and sparsity.
"""

import argparse
import json
import math
import random
import numpy as np
import torch

from protify.base_models.get_base_models import standard_models
from protify.console_output import print_message
from protify.data.data_mixin import DataMixin, DataArguments
from protify.embedder import Embedder, EmbeddingArguments
from protify.seed_utils import set_global_seed, get_global_seed


# Default test datasets
DEFAULT_TEST_DATASETS = [
    'EC',  # multilabel
]

# Diagnostics by dataset, then model, then pooling method.
SuiteResults = dict[str, dict[str, dict[str, dict[str, float]]]]


seed = get_global_seed()
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)

def load_and_sample_sequences(
    dataset_names: list[str],
    sample_frac: float = 0.1,
    max_length: int = 1024,
    trim: bool = False
) -> dict[str, list[str]]:
    """
    Load datasets and sample sequences from them.

    Args:
        dataset_names: List of dataset names to load
        sample_frac: Fraction of sequences to sample (default 0.1 = 10%)
        max_length: Maximum sequence length
        trim: Whether to trim sequences to max_length
    
    Returns:
        Dictionary mapping dataset names to lists of sampled sequences
    """
    dataset_seqs = {}
    
    for dataset_name in dataset_names:
        print_message(f"Loading dataset: {dataset_name}")
        
        try:
            # Load dataset using DataMixin
            data_args = DataArguments(
                data_names=[dataset_name],
                max_length=max_length,
                trim=trim
            )
            data_mixin = DataMixin(data_args)
            datasets, all_seqs = data_mixin.get_data()
            
            # Get sequences from all splits
            sequences = []
            if dataset_name in datasets:
                train_set, valid_set, test_set, _, _, ppi = datasets[dataset_name]
                
                if ppi:
                    # For PPI datasets, combine SeqA and SeqB
                    sequences.extend(list(train_set['SeqA']))
                    sequences.extend(list(train_set['SeqB']))
                    sequences.extend(list(valid_set['SeqA']))
                    sequences.extend(list(valid_set['SeqB']))
                    sequences.extend(list(test_set['SeqA']))
                    sequences.extend(list(test_set['SeqB']))
                else:
                    sequences.extend(list(train_set['seqs']))
                    sequences.extend(list(valid_set['seqs']))
                    sequences.extend(list(test_set['seqs']))
            else:
                # Use all sequences if dataset processing failed
                sequences = list(all_seqs)
            
            # Sample
            sequences = list(set(sequences))
            n_samples = max(1, math.ceil(len(sequences) * sample_frac))
            sampled = random.sample(sequences, min(n_samples, len(sequences)))
            dataset_seqs[dataset_name] = sampled
            
            print_message(f"Sampled {len(sampled)} sequences from {len(sequences)} total")
            
        except Exception as e:  # noqa: broad-except  a dataset that fails to load is reported and skipped
            print_message(f"Error loading dataset {dataset_name}: {e}")
            continue
    
    return dataset_seqs


def compute_diagnostics(embeddings: torch.Tensor, zero_eps: float = 1e-8) -> dict[str, float]:
    # embeddings: (n, d)
    emb = embeddings.detach().float().cpu().numpy()  # (n, d)
    flat = emb.ravel()  # (n * d,)

    is_nan = np.isnan(flat)  # (n * d,)
    is_inf = np.isinf(flat)  # (n * d,)
    is_finite = np.isfinite(flat)  # (n * d,)

    finite = flat[is_finite]  # (m,), m finite entries
    if finite.size == 0:
        # If everything is NaN/Inf
        return {
            "n_samples": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]),
            "finite_count": 0,
            "nan_count": int(is_nan.sum()),
            "inf_count": int(is_inf.sum()),
        }

    near_zero = np.abs(finite) < zero_eps  # (m,)

    sample_l2 = np.linalg.norm(emb, axis=1)  # (n,)

    return {
        "n_samples": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),

        "finite_count": int(finite.size),
        "finite_fraction": float(finite.size / flat.size),

        "nan_count": int(is_nan.sum()),
        "nan_fraction": float(is_nan.mean()),

        "inf_count": int(is_inf.sum()),
        "inf_fraction": float(is_inf.mean()),

        "zero_eps": float(zero_eps),
        "near_zero_count": int(near_zero.sum()),
        "near_zero_fraction": float(near_zero.mean()),

        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p50": float(np.percentile(finite, 50)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),

        "mean_l2": float(np.mean(sample_l2)),
        "std_l2": float(np.std(sample_l2)),
        "p95_l2": float(np.percentile(sample_l2, 95)),
    }


def embed_and_diagnose(
    sequences: list[str],
    model_name: str,
    pooling_types: list[str],
    batch_size: int = 16,
    num_workers: int = 0
) -> dict[str, dict[str, float]]:
    """
    Embed sequences and compute diagnostics for each pooling type.
    
    Args:
        sequences: List of sequences to embed
        model_name: Name of the model to use
        pooling_types: List of pooling types to test
        batch_size: Batch size for embedding
        num_workers: Number of workers for data loading
    
    Returns:
        Dictionary mapping pooling types to their diagnostics
    """
    print_message(f"Embedding {len(sequences)} sequences with {model_name}")
    
    # Parse pooling types so combinations can be tested
    pooling_list = {}
    for pool_type in pooling_types:
        # Check if it's a combination
        if ',' in pool_type:
            # Split and create a list for the combination
            pool_list = [p.strip() for p in pool_type.split(',')]
            pooling_list[pool_type] = pool_list
        else:
            # Single pooling type
            pooling_list[pool_type] = [pool_type]
    
    results = {}
    
    # Load model once and reuse for all pooling types
    print_message(f"Loading model: {model_name}")
    from protify.base_models.get_base_models import get_base_model
    model, tokenizer = get_base_model(model_name)
    
    for pool_type, pool_list in pooling_list.items():
        print_message(f"Testing pooling: {pool_type} (types: {pool_list})")
        
        # Set up embedder for this pooling type
        embedder_args = EmbeddingArguments(
            embedding_batch_size=batch_size,
            embedding_num_workers=num_workers,
            download_embeddings=False,
            matrix_embed=False,
            embedding_pooling_types=pool_list,
            save_embeddings=False,
            embed_dtype=torch.float32,
            sql=False,
            embedding_save_dir='embeddings'
        )
        
        embedder = Embedder(embedder_args, sequences)
        
        try:
            # read embeddings from disk if they exist
            to_embed, save_path, embeddings_dict = embedder._read_embeddings_from_disk(model_name)
            
            if len(to_embed) > 0:
                embedded = embedder._embed_sequences(
                    to_embed, save_path, model, tokenizer, embeddings_dict
                )
                if embedded is not None:
                    embeddings_dict = embedded
            
            if embeddings_dict is None or len(embeddings_dict) == 0:
                print_message(f"Warning: No embeddings returned for {model_name} with {pool_type}")
                continue
            
            embedding_tensors = []
            for seq in sequences:
                if seq in embeddings_dict:
                    embedding_tensors.append(embeddings_dict[seq])
            
            if len(embedding_tensors) == 0:
                print_message(f"Error: No embeddings found for {pool_type}")
                continue
            
            embeddings = torch.stack(embedding_tensors)
            
            diagnostics = compute_diagnostics(embeddings)
            results[pool_type] = diagnostics
            
        except Exception as e:  # noqa: broad-except  a model that fails to embed is reported and skipped
            print_message(f"Error embedding with {model_name} using {pool_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def run_test_suite(
    dataset_names: list[str] | None = None,
    model_names: list[str] | None = None,
    pooling_methods: list[str] | None = None,
    sample_frac: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 0
) -> SuiteResults:
    """
    Run the embedding test suite.
    """
    if dataset_names is None:
        dataset_names = DEFAULT_TEST_DATASETS
    
    if model_names is None:
        model_names = standard_models

    if pooling_methods is None:
        pooling_methods = ['cls', 'mean,var']
    
    print_message("Running embedding test suite")
    print_message(f"Datasets: {dataset_names}")
    print_message(f"Models: {model_names}")
    print_message(f"Pooling methods: {pooling_methods}")
    print_message(f"Sample fraction: {sample_frac}")
    
    dataset_seqs = load_and_sample_sequences(dataset_names, sample_frac=sample_frac)
    
    if len(dataset_seqs) == 0:
        print_message("Error: No sequences loaded")
        return {}
    
    all_results = {}
    
    for dataset_name, sequences in dataset_seqs.items():
        print_message(f"\nProcessing dataset: {dataset_name}")
        all_results[dataset_name] = {}
        
        for model_name in model_names:
            print_message(f"Model: {model_name}")
            model_results = embed_and_diagnose(
                sequences,
                model_name,
                pooling_methods,
                batch_size=batch_size,
                num_workers=num_workers
            )
            
            if model_results:
                all_results[dataset_name][model_name] = model_results
    
    print_table_results(all_results)
    print_json_results(all_results)
    
    return all_results


def print_table_results(results: SuiteResults) -> None:
    """Print results in table format."""
    print("\n" + "="*100)
    print("EMBEDDING TEST SUITE RESULTS")
    print("="*100)
    
    for dataset_name, dataset_results in results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 100)
        
        for model_name, model_results in dataset_results.items():
            print(f"\n  Model: {model_name}")
            
            for pool_type, diagnostics in model_results.items():
                print(f"\nPooling: {pool_type}")
                print(f"Samples: {diagnostics['n_samples']}, Dim: {diagnostics['embedding_dim']}")
                print(f"Mean: {diagnostics['mean']:.6f}, Std: {diagnostics['std']:.6f}")
                print(f"Min: {diagnostics['min']:.6f}, Max: {diagnostics['max']:.6f}")
                print(f"Percentiles: P25={diagnostics['p25']:.6f}, P50={diagnostics['p50']:.6f}, "
                      f"P75={diagnostics['p75']:.6f}, P95={diagnostics['p95']:.6f}, P99={diagnostics['p99']:.6f}")
                print(f"NaN: {diagnostics['nan_count']} ({diagnostics['nan_fraction']*100:.2f}%)")
                if 'near_zero_count' in diagnostics:
                    print(f"Near zeros: {diagnostics['near_zero_count']} ({diagnostics['near_zero_fraction']*100:.2f}%)")
                print(f"Inf: {diagnostics['inf_count']} ({diagnostics['inf_fraction']*100:.2f}%)")
                
                # Flag anomalies
                anomalies = []
                if diagnostics['nan_fraction'] > 0:
                    anomalies.append(f"NaNs detected ({diagnostics['nan_fraction']*100:.2f}%)")
                if 'near_zero_fraction' in diagnostics and diagnostics['near_zero_fraction'] > 0.2:
                    anomalies.append(f"High sparsity ({diagnostics['near_zero_fraction']*100:.2f}%)")
                if diagnostics['inf_fraction'] > 0:
                    anomalies.append(f"Infs detected ({diagnostics['inf_fraction']*100:.2f}%)")
                if abs(diagnostics['mean']) > 100:
                    anomalies.append(f"Extreme mean ({diagnostics['mean']:.2f})")
                if diagnostics['std'] > 100:
                    anomalies.append(f"Extreme std ({diagnostics['std']:.2f})")
                
                if anomalies:
                    print(f"Anomalies: {', '.join(anomalies)}")
                else:
                    print("No anomalies detected")


def print_json_results(results: SuiteResults) -> None:
    """Print results in JSON format."""
    print("\n" + "="*50)
    print("JSON RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Embedding Test Suite - Test embedding quality across datasets and models'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='List of dataset names to test (default: EC)'
    )
    
    parser.add_argument(
        '--model_names',
        nargs='+',
        default=None,
        help='List of model names to test (default: all currently_supported_models)'
    )
    
    parser.add_argument(
        '--pooling_methods',
        nargs='+',
        default=['cls', 'mean,var'],
        help='List of pooling methods to test (default: mean, var, cls, parti, mean,var)'
    )
    
    parser.add_argument(
        '--sample_frac',
        type=float,
        default=0.1,
        help='Fraction of sequences to sample from each dataset (default: 0.1)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for embedding (default: 16)'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loading (default: 0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        set_global_seed(args.seed)
    
    # Run test suite
    results = run_test_suite(
        dataset_names=args.datasets,
        model_names=args.model_names,
        pooling_methods=args.pooling_methods,
        sample_frac=args.sample_frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    return results


if __name__ == '__main__':
    main()
