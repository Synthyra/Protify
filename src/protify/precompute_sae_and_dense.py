"""Fill many sparse-autoencoder caches and dense caches at any depth from one backbone pass.

A sparse autoencoder reads one hidden state, so by the time its codebook features exist the
dense representation of that same layer has already been computed, and so has the input
every other sparse autoencoder trained at that layer needs. Returning the hidden states runs
every block, so the same pass has also computed every later state, up to the final
normalized state an ordinary run pools. Running each arm as its own model therefore puts
every protein through ESMC once per arm to produce several views of one pass. On the
campaign's remaining datasets that was roughly 126,000 redundant forward passes for two
arms alone, and a sparsity by codebook-width sweep multiplies it again.

This writes every cache in one pass, into the feature stores the separate runs would have
used, so an ordinary Protify run afterwards finds them populated and skips embedding
entirely. Nothing here changes the Embedder: both write through `feature_cache.CacheWriter`
under the descriptors `feature_cache.embedding_descriptor` builds, so the stores are the ones
readers already open. Rows are float32, which a store serves to a run of any narrower dtype.

`--sae_k` and `--sae_codebook_dim` each accept several values and fill their cross product
at one layer. The first configuration is attached to the backbone and its features come
back from the same forward that produces the hidden states; the rest are
`EsmcSaeResidueEncoder`s applied to the residues of the SAE's layer. Backbone passes
therefore equal batches, whatever the size of the grid. What does grow with the grid is
resident codebook weight, roughly `2 * d * c` parameters per configuration, reported
before anything loads.

`--dense_hidden_state_indices` names the hidden states the dense mean,var caches pool, one
cache per index, each keyed as an ordinary run with that `--embedding_hidden_state_index`
keys its cache. The default is the SAE's own layer. Index -1 is the final normalized
state, so it fills the cache a plain run of the backbone reads, the ESMC-600 store of mean
and variance pooled vectors, at no extra forward.

Resumption is per cache. A protein already present in some caches and missing from others
is still embedded once, and written only where it is missing. Rows commit a window at a time,
so a pass that stops loses at most the window it had open in each cache.

`--max_train_rows`, `--max_valid_rows`, `--max_test_rows`, and `--split_row_cap_seed` cap
each dataset's splits exactly as `main.py` does, so the caches hold the rows a capped run
reads and no protein it drops.

`--shard i/n` embeds only the i-th of n deterministic slices of those proteins, so a pass too
long for one job runs as n jobs, each into its own directory. The slices are disjoint and
together hold every protein. To read the n shards' caches as the whole pass's, copy each
shard store's `segments/*` directories into one store of the same name and open it once for
writing (`FeatureStore.open`), which rebuilds its index; segment names come from their rows,
so shards never collide.

Usage:
    precompute_sae_and_dense.py --model_names ESMC-6B-SAE --sae_layer 60 --sae_k 64 \\
        --sae_codebook_dim 8192 16384 32768 65536 131072 \\
        --data_names GO-rigor [--data_revisions <commit>] [--embedding_batch_size 4]
    precompute_sae_and_dense.py --model_names ESMC-600-SAE --sae_layer 27 --sae_k 64 \\
        --sae_codebook_dim 16384 --dense_hidden_state_indices 27 -1 --data_names DeepLoc-2
"""

from __future__ import annotations

import argparse
import re
import sys
import torch

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from base_models.esmc import ESMTokenizerWrapper
    from base_models.esmc_sae import (
        ESMC_SAE_COVERAGE,
        EsmcSaeForEmbedding,
        EsmcSaeResidueEncoder,
        EsmcSaeSelection,
        build_esmc_sae_model,
        parse_sae_model_name,
        resolve_sae_model_name,
    )
    from base_models.get_base_models import BaseModelArguments, resolve_data_max_length
    from data.data_mixin import DataArguments, DataMixin
    from data.declared_dataset import load_declared_dataset
    from feature_cache import CacheWriter, embedding_descriptor, embedding_layout, locate_feature, missing_sequences
    from pooler import Pooler
    from console_output import print_message
except ImportError:
    from protify.base_models.esmc import ESMTokenizerWrapper
    from protify.base_models.esmc_sae import (
        ESMC_SAE_COVERAGE,
        EsmcSaeForEmbedding,
        EsmcSaeResidueEncoder,
        EsmcSaeSelection,
        build_esmc_sae_model,
        parse_sae_model_name,
        resolve_sae_model_name,
    )
    from protify.base_models.get_base_models import BaseModelArguments, resolve_data_max_length
    from protify.data.data_mixin import DataArguments, DataMixin
    from protify.data.declared_dataset import load_declared_dataset
    from protify.feature_cache import (
        CacheWriter, embedding_descriptor, embedding_layout, locate_feature, missing_sequences,
    )
    from protify.pooler import Pooler
    from protify.console_output import print_message


# The dense caches this pass also fills. By default one reads the SAE's own layer, so the only
# difference between the two arms is the representation, never the depth; any other depth,
# the final normalized state included, comes from the same pass. Each hidden state fills one cache
# per dense pooling set, mean,var by default, and `mean` alone for the suite's `mean_final` arm.
DENSE_POOLING_TYPES = ('mean', 'var')
DENSE_BACKBONE_NAME = {'ESMC-300': 'ESMC-300', 'ESMC-600': 'ESMC-600', 'ESMC-6B': 'ESMC-6B'}
# The rows this pass computes are float32 (pooled from float32 hidden states), so its stores are float32.
# A run of a narrower dtype is served by them, because a store serves every dtype it holds exactly.
CACHE_DTYPE = torch.float32


def sae_descriptor(model_name: str, pooling: Sequence[str], max_length: int) -> dict[str, Any]:
    """The descriptor of one SAE configuration's cache, as a plain run of that model builds it."""
    return embedding_descriptor(
        model_name, matrix_embed=False, pooling_types=list(pooling), hidden_state_index=-1,
        max_length=max_length, dtype=CACHE_DTYPE,
    )


def dense_descriptor(
    backbone_name: str, hidden_state_index: int, max_length: int, pooling: Sequence[str] = DENSE_POOLING_TYPES,
) -> dict[str, Any]:
    """The descriptor of one dense cache, as a plain run with that hidden state and pooling builds it.

    Index -1 is the final normalized state, which is the cache a plain run of the backbone reads.
    """
    return embedding_descriptor(
        backbone_name, matrix_embed=False, pooling_types=list(pooling),
        hidden_state_index=hidden_state_index, max_length=max_length, dtype=CACHE_DTYPE,
    )


class _CacheWriter:
    """One representation's feature store, opened for append and resumable.

    Rows commit a window at a time (`feature_cache.CacheWriter`), so a pass that stops keeps every
    committed window, and the next run embeds only what `missing` still reports.
    """

    def __init__(self, root: str, descriptor: Mapping[str, Any]) -> None:
        self.root = root
        self.descriptor = dict(descriptor)
        self._writer = CacheWriter(root, self.descriptor, layout=embedding_layout(self.descriptor))

    @property
    def directory(self) -> Path:
        """The store that serves this cache, or where it will be made."""
        return locate_feature(self.root, self.descriptor)

    def missing(self, sequences: Sequence[str]) -> set[str]:
        """The distinct `sequences` the store does not hold yet; all of them when there is no store."""
        return set(missing_sequences(self.directory, sequences))

    def write(self, sequences: list[str], embeddings: torch.Tensor) -> None:
        # embeddings: (b, d), already on CPU in the stored dtype
        if sequences:
            self._writer.add(sequences, list(embeddings))

    def close(self) -> None:
        """Commit the rows of the window still open."""
        self._writer.close()


@dataclass(frozen=True)
class PooledCache:
    """One cache and the pooling set it stores, in stored order."""

    writer: _CacheWriter
    pooling: tuple[str, ...]


@dataclass(frozen=True)
class SaeCache:
    """One configuration's encoder and every cache its features fill.

    `encoder` is None for the configuration attached to the backbone, whose pooled features
    already come back from the forward that produced the residues. Pairing the two here is
    what keeps a cache from being written by another configuration's features.

    A configuration writes one cache per requested pooling set. Every set is a subset of
    the union the encoder reduces to once, so the second and later sets cost a column slice
    rather than another encode.
    """

    encoder: EsmcSaeResidueEncoder | None
    codebook_dim: int
    pooled_caches: tuple[PooledCache, ...]


@dataclass(frozen=True)
class DenseCache:
    """One dense cache, the hidden state it pools, and the pooling set it stores.

    Pairing them here is what keeps a cache from holding another depth's or another set's
    pooling under its own name.
    """

    writer: _CacheWriter
    hidden_state_index: int
    pooling: tuple[str, ...] = DENSE_POOLING_TYPES


def parse_pooling_sets(tokens: Sequence[str]) -> list[tuple[str, ...]]:
    """One pooling set per token, with the names inside a token joined by commas.

    `--embedding_pooling_types max max,mean` writes a max-pooled cache and a
    max-then-mean-pooled cache, under exactly the two filenames two separate runs would
    have used. Both are subsets of the union the encoder reduces to once, so the second
    cache costs a column slice rather than a second pass over the backbone.
    """
    sets = [tuple(name.strip() for name in token.split(',') if name.strip()) for token in tokens]
    if any(not pooling for pooling in sets):
        raise ValueError(f'Empty pooling set in {tokens}.')

    return list(dict.fromkeys(sets))


def pooling_union(sets: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    """Every reduction any cache asks for, once, in first-seen order."""
    return tuple(dict.fromkeys(name for pooling in sets for name in pooling))


def select_pooling_columns(
    pooled: torch.Tensor,
    union: Sequence[str],
    wanted: Sequence[str],
    codebook_dim: int,
) -> torch.Tensor:
    """The columns of a union-pooled row that spell out one requested pooling set.

    `Pooler` and `pool_sae_outputs` both concatenate in the order they were given, so each
    reduction owns one contiguous block of `codebook_dim` columns at its index in the
    union.
    """
    # pooled: (b, u * c) -> (b, p * c) for p wanted reductions
    if tuple(union) == tuple(wanted):
        return pooled  # (b, u * c), already in the wanted order

    starts = {name: index * codebook_dim for index, name in enumerate(union)}
    return torch.cat(  # (b, p * c)
        [pooled[:, starts[name]: starts[name] + codebook_dim] for name in wanted], dim=-1
    )


def _batched(sequences: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(sequences), size):
        yield sequences[start:start + size]


def parse_shard(text: str) -> tuple[int, int]:
    """`i/n`, the i-th of n slices counted from 1, as `--shard` takes it."""
    match = re.fullmatch(r'(\d+)/(\d+)', text)
    if match is None or not 1 <= int(match.group(1)) <= int(match.group(2)):
        raise argparse.ArgumentTypeError(f'--shard takes i/n with 1 <= i <= n, not {text!r}')

    return int(match.group(1)), int(match.group(2))


def shard_sequences(sequences: Sequence[str], shard: tuple[int, int] | None) -> list[str]:
    """The proteins one shard embeds: every n-th of them in a fixed order, from the i-th on.

    The loader breaks length ties in set order, which differs between processes, so the order
    here is longest first and then the sequence itself. It depends only on which proteins the
    pass reads, so the n shards of one pass take disjoint slices that together hold every
    protein. Dealing every n-th protein, rather than cutting contiguous blocks, gives each shard
    the same mix of lengths, and so about the same time.
    """
    if shard is None:
        return list(sequences)

    index, count = shard
    ordered = sorted(set(sequences), key=lambda sequence: (-len(sequence), sequence))
    return ordered[index - 1::count]


def sae_configuration_names(
    alias: str,
    sae_layer: int | None,
    k_values: Sequence[int | None],
    codebook_dims: Sequence[int | None],
) -> list[str]:
    """Fully qualified model names for every configuration this pass fills, at one layer.

    The cross product runs sparsity outermost, so a codebook sweep reads in the order it was
    given. A `None` keeps whatever the alias already fixes, or the published default, which
    is what makes a single-configuration call resolve exactly as it did before.
    `resolve_sae_model_name` rejects a cell Biohub does not train, and an already qualified
    alias the sweep contradicts, so an unpublished cell fails here rather than opening an
    empty cache and embedding into it.
    """
    names = [
        resolve_sae_model_name(alias, sae_layer=sae_layer, sae_k=k, sae_codebook_dim=codebook_dim)
        for k in k_values
        for codebook_dim in codebook_dims
    ]
    # A repeated --sae_k or --sae_codebook_dim value would otherwise open one cache twice.
    return list(dict.fromkeys(names))


def dense_hidden_state_indices(requested: Sequence[int] | None, selection: EsmcSaeSelection) -> list[int]:
    """The hidden states the dense caches pool, checked against the backbone, each once.

    None keeps the single dense cache at the SAE's own layer. The backbone returns
    `num_hidden_layers + 1` states, the input to each block and then the final normalized
    state, so any Python index into that tuple is valid, -1 among them. `main` calls this
    before any cache opens or anything loads, so a wrong index costs nothing.
    """
    if requested is None:
        return [selection.layer]

    state_count = ESMC_SAE_COVERAGE[selection.backbone].num_hidden_layers + 1
    outside = [index for index in requested if not -state_count <= index < state_count]
    if outside:
        raise ValueError(
            f'{selection.backbone} returns {state_count} hidden states, so '
            f'--dense_hidden_state_indices takes {-state_count} to {state_count - 1}; '
            f'{outside} lie outside that range.'
        )

    # A repeated index would otherwise open one cache twice.
    return list(dict.fromkeys(requested))


def codebook_weight_bytes(selections: Sequence[EsmcSaeSelection], dtype: torch.dtype) -> int:
    """Codebook weight held resident for a whole sweep.

    Each sparse autoencoder carries `W_enc` of (d, c) and `W_dec` of (c, d). Activations are
    not counted: configurations are encoded one at a time, so only one dense (n, c) feature
    tensor is ever alive. This is the number that decides how wide a grid a card can hold.
    """
    element_size = torch.empty((), dtype=dtype).element_size()
    return sum(
        2 * ESMC_SAE_COVERAGE[selection.backbone].hidden_size * selection.codebook_dim * element_size
        for selection in selections
    )


def fill_caches(
    model: EsmcSaeForEmbedding,
    tokenizer: ESMTokenizerWrapper,
    sae_caches: Sequence[SaeCache],
    dense_caches: Sequence[DenseCache],
    sequences: Sequence[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    sae_layer: int,
    union: Sequence[str] = ('max',),
) -> int:
    """Embed every protein missing from any cache exactly once, writing only where missing.

    `sae_layer` is the hidden state every codebook reads, and each dense cache pools the
    hidden state it names; all of them come from the one backbone pass per batch. `union`
    is every reduction the encoders compute, in the order their columns appear. Each SAE
    cache stores a subset of it.

    Returns the number of proteins embedded, which is how many the caches between them were
    missing rather than how many rows were written.
    """
    absent_sae = [
        [pooled.writer.missing(sequences) for pooled in cache.pooled_caches] for cache in sae_caches
    ]  # one set per cache, grouped by configuration
    absent_dense = [cache.writer.missing(sequences) for cache in dense_caches]  # one set per dense cache
    todo = list(dict.fromkeys(
        sequence
        for sequence in sequences
        if any(sequence in absent for absent in absent_dense)
        or any(sequence in absent for group in absent_sae for absent in group)
    ))
    distinct = len(set(sequences))
    print_message(
        f'{[distinct - len(absent) for absent in absent_dense]} already in the dense caches, '
        f'{[distinct - len(absent) for group in absent_sae for absent in group]} in the SAE caches, '
        f'{len(todo)} proteins to embed once'
    )

    poolers = {pooling: Pooler(list(pooling)) for pooling in {dense.pooling for dense in dense_caches}}
    # Longest first, so a batch size the widest protein cannot fit fails on the first batch
    # rather than hours in.
    todo = sorted(todo, key=len, reverse=True)
    embedded = 0
    with torch.inference_mode():
        for chunk in _batched(todo, batch_size):
            batch = tokenizer(chunk, padding='longest', truncation=True, max_length=max_length, return_tensors='pt')
            batch = {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
            attention_mask = batch.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones_like(batch['input_ids'])  # (b, l)

            # The one backbone pass this module exists for. Every configuration below reads
            # the SAE layer's residues and every dense cache reads its own hidden state, so
            # forwards equal batches rather than batches times caches.
            pooled_attached, hidden_states = model.sae_and_hidden_states(batch['input_ids'], attention_mask)
            # pooled_attached: (b, p * c) for the attached codebook; hidden_states: n_layers + 1 of (b, l, d)

            # Pooled before any encoder runs, as the single dense cache always was.
            for dense, absent in zip(dense_caches, absent_dense, strict=True):
                dense_rows = [(sequence, row) for row, sequence in enumerate(chunk) if sequence in absent]
                if not dense_rows:
                    continue

                # Pooled over every attended token, CLS and EOS included: l + 2 rows per protein.
                pooled_dense = poolers[dense.pooling](
                    emb=hidden_states[dense.hidden_state_index].float(),  # (b, l + 2, d)
                    attention_mask=attention_mask,
                    attentions=None,
                    input_ids=batch['input_ids'],
                    eos_token_id=getattr(tokenizer, 'pooling_token_id', None),
                )  # (b, p * d), p pooled statistics side by side
                dense.writer.write(
                    [sequence for sequence, _ in dense_rows],
                    pooled_dense[[row for _, row in dense_rows]].float().cpu(),  # (m, p * d), m rows missing here
                )

            residues = hidden_states[sae_layer]  # (b, l, d), the stream every codebook reads
            for cache, group in zip(sae_caches, absent_sae, strict=True):
                missing = []
                for pooled, absent in zip(cache.pooled_caches, group, strict=True):
                    rows = [
                        (sequence, row)
                        for row, sequence in enumerate(chunk) if sequence in absent
                    ]
                    if rows:
                        missing.append((pooled, rows))

                if not missing:
                    continue

                # Encoded one configuration at a time and shared by that configuration's
                # pooling sets, so only the widest single codebook's dense (n, c) activation
                # is alive rather than one per grid cell.
                pooled_union = (
                    pooled_attached
                    if cache.encoder is None
                    else cache.encoder(residues, attention_mask)
                )  # (b, u * c)
                for pooled, rows in missing:
                    pooled_sae = select_pooling_columns(
                        pooled_union, union, pooled.pooling, cache.codebook_dim
                    )  # (b, p * c)
                    pooled.writer.write(
                        [sequence for sequence, _ in rows],
                        pooled_sae[[row for _, row in rows]].float().cpu(),
                    )

            embedded += len(chunk)
            if embedded % (batch_size * 50) == 0:
                print_message(f'{embedded}/{len(todo)} proteins embedded')

    return embedded


def load_input_sequences(arguments: argparse.Namespace, resolved_names: list[str]) -> list[str]:
    """Read one deduplicated feature inventory, keeping declared releases unchanged."""
    max_length = resolve_data_max_length(resolved_names, arguments.max_length, trim=False)
    if arguments.dataset_manifest:
        if arguments.data_names or arguments.data_revisions or arguments.dna_to_aa or any(
            getattr(arguments, name) is not None for name in ('max_train_rows', 'max_valid_rows', 'max_test_rows')
        ):
            raise ValueError('declared embeddings forbid legacy sources, row caps, and sequence conversion')
        digests = arguments.dataset_sha256 or []
        if len(digests) != len(arguments.dataset_manifest):
            raise ValueError('one immutable dataset SHA-256 is required for each declared manifest')
        sequences: set[str] = set()
        for path, digest in zip(arguments.dataset_manifest, digests, strict=True):
            release = load_declared_dataset(path, digest, max_sequence_length=max_length)
            for split in release.splits.values():
                sequences.update(split['seqs'])
        return sorted(sequences, key=lambda sequence: (len(sequence), sequence))
    if arguments.dataset_sha256:
        raise ValueError('dataset_sha256 requires declared dataset manifests')
    data_mixin = DataMixin()
    data_mixin.data_args = DataArguments(
        data_names=arguments.data_names, data_revisions=arguments.data_revisions,
        max_length=arguments.max_length, max_train_rows=arguments.max_train_rows,
        max_valid_rows=arguments.max_valid_rows, max_test_rows=arguments.max_test_rows,
        split_row_cap_seed=arguments.split_row_cap_seed, dna_to_aa=arguments.dna_to_aa,
    )
    data_mixin.model_args = BaseModelArguments(model_names=resolved_names)
    data_mixin._sql = True
    data_mixin._full = False
    data_mixin._max_length = arguments.max_length
    data_mixin._trim = False
    data_mixin._dna_to_aa = arguments.dna_to_aa
    data_mixin._data_max_length = max_length
    _datasets, all_seqs = data_mixin.get_data()
    return all_seqs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_names', nargs='+', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--data_names', nargs='+')
    source.add_argument('--dataset_manifest', nargs='+', help='Completed, immutable four-way dataset JSON files; all their sequences share one feature pass.')
    parser.add_argument('--dataset_sha256', nargs='+', help='Manifest SHA-256 pins aligned one-for-one with --dataset_manifest.')
    parser.add_argument('--data_revisions', nargs='+', default=None, help='Optional immutable Hugging Face revisions aligned one-for-one with --data_names, as main.py takes them.')
    parser.add_argument('--max_train_rows', type=int, default=None, help='Most training rows each dataset keeps, drawn exactly as main.py draws them, so the caches hold the rows a capped run reads.')
    parser.add_argument('--max_valid_rows', type=int, default=None, help='Most validation rows each dataset keeps, as main.py takes it.')
    parser.add_argument('--max_test_rows', type=int, default=None, help='Most test rows each dataset keeps, as main.py takes it.')
    parser.add_argument('--split_row_cap_seed', type=int, default=0, help='Seed of the row caps, as main.py takes it.')
    parser.add_argument('--dna_to_aa', action='store_true', help='Translate DNA codon sequences to proteins before embedding, as main.py does, so the caches hold the proteins a --dna_to_aa run reads.')
    parser.add_argument('--sae_layer', type=int, default=None)
    parser.add_argument('--sae_k', type=int, nargs='+', default=None, help='One or more sparsities; every value pairs with every codebook width.')
    parser.add_argument('--sae_codebook_dim', type=int, nargs='+', default=None, help='One or more codebook widths; every value pairs with every sparsity.')
    parser.add_argument('--embedding_pooling_types', nargs='+', default=['max'], help="One pooling set per token, names inside a token joined by commas. 'max max,mean' writes both caches from one pass.")
    parser.add_argument('--dense_hidden_state_indices', type=int, nargs='+', default=None, help='Hidden states the dense caches pool, one cache per pooling set each, named as an ordinary run with that --embedding_hidden_state_index names it. Defaults to the SAE layer; -1 is the final normalized state a plain run pools.')
    parser.add_argument('--dense_pooling', nargs='+', default=['mean,var'], help="One dense pooling set per token, names inside a token joined by commas. 'mean,var mean' fills the default branch's cache and the mean_final arm's.")
    parser.add_argument('--embedding_batch_size', type=int, default=4)
    parser.add_argument('--embedding_save_dir', type=str, default='embeddings')
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--model_dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
    parser.add_argument('--shard', type=parse_shard, default=None, help='i/n: embed only the i-th of n disjoint slices of the proteins, which together hold all of them. Give each shard its own --embedding_save_dir.')
    arguments = parser.parse_args()

    # An omitted sweep axis stays omitted, so the alias or the published default decides it.
    k_values = arguments.sae_k if arguments.sae_k is not None else [None]
    codebook_dims = arguments.sae_codebook_dim if arguments.sae_codebook_dim is not None else [None]
    resolved_names = sae_configuration_names(
        arguments.model_names[0], arguments.sae_layer, k_values, codebook_dims
    )
    selections = [parse_sae_model_name(name) for name in resolved_names]
    dense_name = DENSE_BACKBONE_NAME[selections[0].backbone]
    dense_indices = dense_hidden_state_indices(arguments.dense_hidden_state_indices, selections[0])
    print_message(
        f'{len(resolved_names)} SAE configurations at layer {selections[0].layer}, '
        f'dense {dense_name} at hidden states {dense_indices}'
    )
    for name in resolved_names:
        print_message(f'  {name}')

    all_seqs = load_input_sequences(arguments, resolved_names)
    print_message(f'{len(all_seqs)} unique proteins across {arguments.dataset_manifest or arguments.data_names}')
    if arguments.shard is not None:
        all_seqs = shard_sequences(all_seqs, arguments.shard)
        print_message(f'shard {arguments.shard[0]} of {arguments.shard[1]}: {len(all_seqs)} of them')

    pooling_sets = parse_pooling_sets(arguments.embedding_pooling_types)
    union = pooling_union(pooling_sets)
    sae_descriptors = [
        [sae_descriptor(name, pooling, arguments.max_length) for pooling in pooling_sets]
        for name in resolved_names
    ]
    # Index -1 is the final normalized state, so the final-state cache is the one a plain run of
    # the backbone keys and reads.
    dense_pooling_sets = parse_pooling_sets(arguments.dense_pooling)
    dense_descriptors = [
        (dense_descriptor(dense_name, index, arguments.max_length, pooling), index, pooling)
        for index in dense_indices
        for pooling in dense_pooling_sets
    ]

    model_dtype = torch.bfloat16 if arguments.model_dtype == 'bf16' else torch.float32
    print_message(
        f'codebook weights resident for this sweep: about '
        f'{codebook_weight_bytes(selections, model_dtype) / 1024 ** 3:.1f} GiB'
    )

    model, tokenizer = build_esmc_sae_model(
        resolved_names[0],
        dtype=model_dtype,
        pooling_types=union,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    # Everything past the first configuration reads residues the backbone already produced.
    side_encoders = [
        EsmcSaeResidueEncoder(selection, pooling_types=union, device=device, dtype=model_dtype).eval()
        for selection in selections[1:]
    ]

    # Each cache takes its own checkpoint's storage decision: sparsity and codebook width
    # move the active fraction, so one grid can straddle the sparse-storage threshold, and
    # `embedding_layout` reads that decision from the model name the store is keyed by.
    encoders = [None, *side_encoders]
    sae_caches = [
        SaeCache(
            encoder=encoder,
            codebook_dim=selection.codebook_dim,
            pooled_caches=tuple(
                PooledCache(_CacheWriter(arguments.embedding_save_dir, descriptor), pooling)
                for descriptor, pooling in zip(descriptors, pooling_sets, strict=True)
            ),
        )
        for selection, encoder, descriptors in zip(selections, encoders, sae_descriptors, strict=True)
    ]
    dense_caches = [
        DenseCache(_CacheWriter(arguments.embedding_save_dir, descriptor), index, pooling)
        for descriptor, index, pooling in dense_descriptors
    ]

    for cache in sae_caches:
        for pooled in cache.pooled_caches:
            print_message(f'SAE cache   {pooled.writer.directory}')
    for dense in dense_caches:
        print_message(f'dense cache {dense.writer.directory}')

    writers = [pooled.writer for cache in sae_caches for pooled in cache.pooled_caches]
    writers += [dense.writer for dense in dense_caches]
    try:
        embedded = fill_caches(
            model,
            tokenizer,
            sae_caches,
            dense_caches,
            all_seqs,
            batch_size=arguments.embedding_batch_size,
            max_length=arguments.max_length,
            device=device,
            sae_layer=selections[0].layer,
            union=union,
        )
    finally:
        # A pass that stops still commits the complete rows of each open window.
        for writer in writers:
            writer.close()

    written = sum(len(cache.pooled_caches) for cache in sae_caches)
    print_message(
        f'done: one pass filled {written} SAE caches and {len(dense_caches)} dense caches '
        f'for {embedded} proteins'
    )


if __name__ == '__main__':
    sys.exit(main())
