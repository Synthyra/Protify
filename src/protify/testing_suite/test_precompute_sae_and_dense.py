"""Filling the SAE and dense caches from a single backbone pass.

A sparse autoencoder reads one hidden state, so the dense view of that layer already exists
by the time the codebook features do, and the pass that returns it has run every block, so
the final normalized state exists too. Embedding the arms separately runs the backbone once
per arm over every protein for several views of one pass.

What has to hold for the combined pass to be usable at all: it writes the exact stores the
separate arms read, the final state under the key a plain run gives it, in the layout their
reader expects; it rejects a hidden state the backbone does not have before anything loads;
it reads the same rows a capped run reads; and it resumes per cache so a protein present in
one and missing from another is embedded once and written once.
"""

import argparse
import sys
import pytest
import torch


try:
    from src.protify import feature_cache
    from src.protify import precompute_sae_and_dense as precompute
    from src.protify.base_models.esmc_sae import parse_sae_model_name, resolve_sae_model_name
    from src.protify.embedder import EmbeddingArguments
    from src.protify.precompute_sae_and_dense import (
        DENSE_POOLING_TYPES,
        _CacheWriter,
        _batched,
        dense_descriptor,
        sae_descriptor,
    )
except ImportError:
    from protify import feature_cache
    from protify import precompute_sae_and_dense as precompute
    from protify.base_models.esmc_sae import parse_sae_model_name, resolve_sae_model_name
    from protify.embedder import EmbeddingArguments
    from protify.precompute_sae_and_dense import (
        DENSE_POOLING_TYPES,
        _CacheWriter,
        _batched,
        dense_descriptor,
        sae_descriptor,
    )


def test_the_two_caches_are_keyed_exactly_as_the_separate_arms_key_them():
    """The whole point is that a later run finds these and skips embedding.

    A descriptor that differs by so much as the pooling order would leave both stores on disk
    and still re-embed everything, which is worse than not having done this at all.
    """
    resolved = resolve_sae_model_name('ESMC-6B-SAE', sae_layer=60, sae_k=512, sae_codebook_dim=131072)
    selection = parse_sae_model_name(resolved)

    sae_arm = EmbeddingArguments(embedding_pooling_types=['max'], max_length=2048)
    dense_arm = EmbeddingArguments(
        embedding_pooling_types=list(DENSE_POOLING_TYPES),
        embedding_hidden_state_index=selection.layer,
        max_length=2048,
    )

    assert sae_descriptor(resolved, ['max'], 2048) == sae_arm.cache_descriptor(resolved)
    assert dense_descriptor('ESMC-6B', selection.layer, 2048) == dense_arm.cache_descriptor('ESMC-6B')
    sae_key = feature_cache.feature_key(sae_descriptor(resolved, ['max'], 2048))
    dense_key = feature_cache.feature_key(dense_descriptor('ESMC-6B', selection.layer, 2048))
    assert sae_key.rsplit('__', 1)[0] == 'protify__ESMC-6B-SAE-l60-k512-c131072__vector__max__len2048__float32'
    assert dense_key.rsplit('__', 1)[0] == 'protify__ESMC-6B__vector__mean-var__hs60__len2048__float32'


def test_the_dense_cache_records_the_sae_layer_not_the_final_state():
    """Both arms must read the same depth, or the comparison measures depth, not representation."""
    selection = parse_sae_model_name(
        resolve_sae_model_name('ESMC-300-SAE', sae_layer=23, sae_k=64, sae_codebook_dim=8192)
    )
    descriptor = dense_descriptor('ESMC-300', selection.layer, 2048)

    assert descriptor['hidden_state_index'] == selection.layer
    assert f'hs{selection.layer}' in feature_cache.feature_key(descriptor)


@pytest.mark.parametrize('hidden_state_index', [27, -1])
def test_each_dense_cache_is_keyed_as_a_plain_run_keys_it(hidden_state_index):
    """The final-state cache is free only if a plain run finds it.

    An ordinary run names and opens its cache through `EmbeddingArguments.cache_descriptor`, so
    each dense cache has to match that for the same pooling and hidden state. Index -1 is the
    final state, which makes it the default ESMC-600 cache.
    """
    plain_run = EmbeddingArguments(
        embedding_pooling_types=['mean', 'var'],
        embedding_hidden_state_index=hidden_state_index,
        max_length=2048,
    )

    descriptor = dense_descriptor('ESMC-600', hidden_state_index, 2048)

    assert descriptor == plain_run.cache_descriptor('ESMC-600')
    if hidden_state_index == -1:
        assert feature_cache.feature_key(descriptor).rsplit('__', 1)[0] == (
            'protify__ESMC-600__vector__mean-var__len2048__float32'
        )


def test_a_plain_run_of_a_narrower_dtype_finds_the_float32_store_the_pass_filled(tmp_path):
    """The pass computes float32 rows, so a float16 or bfloat16 run must find them rather than re-embed."""
    writer = _CacheWriter(str(tmp_path), dense_descriptor('ESMC-600', -1, 2048))
    writer.write(['AAAA'], torch.tensor([[1.0, 2.0]]))
    writer.close()

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        plain_run = EmbeddingArguments(
            embedding_save_dir=str(tmp_path), embedding_pooling_types=['mean', 'var'],
            embed_dtype=dtype, max_length=2048,
        )
        assert plain_run.feature_directory('ESMC-600') == str(writer.directory)
        assert feature_cache.missing_sequences(plain_run.feature_directory('ESMC-600'), ['AAAA', 'CCCC']) == ('CCCC',)


def test_written_rows_read_back_through_the_ordinary_reader(tmp_path):
    """The cache is only useful if the reader every run uses returns what this wrote."""
    writer = _CacheWriter(str(tmp_path), dense_descriptor('ESMC-600', -1, 2048))
    embeddings = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)

    writer.write(['AAAA', 'CCCC'], embeddings)
    writer.close()

    rows = feature_cache.read_embeddings(writer.directory, ['AAAA', 'CCCC'])
    assert torch.equal(rows['AAAA'], embeddings[0])
    assert torch.equal(rows['CCCC'], embeddings[1])


def test_sparse_storage_round_trips_through_the_same_reader(tmp_path):
    """SAE features are stored as coordinates, which the reader densifies on the way out."""
    descriptor = sae_descriptor('ESMC-300-SAE-l23-k16-c8192', ['max'], 2048)
    assert feature_cache.embedding_layout(descriptor) == 'csr'
    writer = _CacheWriter(str(tmp_path), descriptor)
    embeddings = torch.zeros(2, 16)  # (2, 16)
    embeddings[0, 3] = 2.5
    embeddings[1, 11] = -1.5

    writer.write(['AAAA', 'CCCC'], embeddings)
    writer.close()

    rows = feature_cache.read_embeddings(writer.directory, ['AAAA', 'CCCC'])
    assert torch.equal(rows['AAAA'], embeddings[0])
    assert torch.equal(rows['CCCC'], embeddings[1])


def test_an_existing_cache_is_resumed_rather_than_restarted(tmp_path):
    """Resumption is per cache, so a restart re-embeds only what is genuinely missing."""
    descriptor = dense_descriptor('ESMC-600', -1, 2048)
    first = _CacheWriter(str(tmp_path), descriptor)
    first.write(['AAAA'], torch.tensor([[1.0, 2.0]]))
    first.close()

    second = _CacheWriter(str(tmp_path), descriptor)
    assert second.missing(['AAAA', 'CCCC']) == {'CCCC'}
    second.write(['CCCC'], torch.tensor([[3.0, 4.0]]))
    second.close()
    assert second.missing(['AAAA', 'CCCC']) == set()


def test_a_window_that_was_never_closed_is_lost_and_the_next_run_embeds_it_again(tmp_path):
    """The store commits a window at a time, so a pass that dies loses only its open window."""
    descriptor = dense_descriptor('ESMC-600', -1, 2048)
    dead = _CacheWriter(str(tmp_path), descriptor)
    dead.write(['AAAA'], torch.tensor([[1.0, 2.0]]))  # buffered, never committed

    assert _CacheWriter(str(tmp_path), descriptor).missing(['AAAA']) == {'AAAA'}


def test_a_missing_cache_starts_empty_and_opening_it_makes_no_store(tmp_path):
    writer = _CacheWriter(str(tmp_path), dense_descriptor('ESMC-600', -1, 2048))

    assert writer.missing(['AAAA', 'CCCC']) == {'AAAA', 'CCCC'}
    writer.close()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    'count, size, expected',
    [(5, 2, [2, 2, 1]), (4, 2, [2, 2]), (1, 4, [1]), (0, 3, [])],
)
def test_batching_covers_every_sequence_exactly_once(count, size, expected):
    sequences = [f'SEQ{index}' for index in range(count)]

    chunks = list(_batched(sequences, size))

    assert [len(chunk) for chunk in chunks] == expected
    assert [seq for chunk in chunks for seq in chunk] == sequences


def test_a_shard_is_a_fixed_slice_whatever_order_the_loader_returns():
    """The loader breaks length ties in set order, which differs between processes.

    Shards run as separate processes, so a slice taken from the loader's own order could give
    two shards one protein and none another. Ordered by length and then sequence, the slices
    are the same in every process, disjoint, and together hold every protein.
    """
    sequences = ['MKAW', 'MKCW', 'MKDW', 'MKAAW', 'MW', 'MKAAAW']

    shards = [precompute.shard_sequences(sequences, (index, 2)) for index in (1, 2)]

    assert shards == [precompute.shard_sequences(list(reversed(sequences)), (index, 2)) for index in (1, 2)]
    # Longest first, ties by sequence: MKAAAW, MKAAW, MKAW, MKCW, MKDW, MW, dealt alternately.
    assert shards == [['MKAAAW', 'MKAW', 'MKDW'], ['MKAAW', 'MKCW', 'MW']]
    assert precompute.shard_sequences(sequences, None) == sequences


def test_shard_takes_i_of_n_counted_from_one():
    assert precompute.parse_shard('2/3') == (2, 3)
    assert precompute.parse_shard('1/1') == (1, 1)
    for text in ('0/2', '3/2', '1', 'a/b', '1/0', '-1/2', '1/2/3'):
        with pytest.raises(argparse.ArgumentTypeError, match='--shard takes i/n'):
            precompute.parse_shard(text)


class _StopBeforeLoading(Exception):
    """Raised in place of loading the datasets, once the loader's arguments are known."""


@pytest.mark.parametrize('revisions', [['0123abcd'], None])
def test_data_revisions_reach_the_loader_one_for_one(monkeypatch, revisions):
    """A pinned revision decides the sequences, which are the cache keys.

    Caches filled from whatever the Hub serves today would miss every protein a pinned run
    reads, so the revisions must reach the loader exactly as main.py hands them over.
    """
    captured = {}

    def stop_before_loading(self):
        captured['data_args'] = self.data_args
        raise _StopBeforeLoading

    monkeypatch.setattr(precompute.DataMixin, 'get_data', stop_before_loading)
    argv = [
        'precompute_sae_and_dense.py', '--model_names', 'ESMC-600-SAE', '--sae_layer', '27',
        '--sae_k', '64', '--sae_codebook_dim', '16384', '--data_names', 'DeepLoc-2',
    ]
    if revisions is not None:
        argv += ['--data_revisions', *revisions]
    monkeypatch.setattr(sys, 'argv', argv)

    with pytest.raises(_StopBeforeLoading):
        precompute.main()

    data_args = captured['data_args']
    assert data_args.data_revisions == revisions
    assert data_args.revision_by_path == ({'GleghornLab/DL2_reg': '0123abcd'} if revisions else {})


@pytest.mark.parametrize(
    'cap_argv, expected',
    [
        ([], ({'train': None, 'valid': None, 'test': None}, 0)),
        (
            ['--max_train_rows', '5000', '--max_valid_rows', '600', '--max_test_rows', '700', '--split_row_cap_seed', '3'],
            ({'train': 5000, 'valid': 600, 'test': 700}, 3),
        ),
    ],
)
def test_row_caps_reach_the_loader(monkeypatch, cap_argv, expected):
    """The caps decide which rows, and so which sequences, a capped run reads.

    Caches filled from uncapped splits would cost embedding the proteins the run drops, and
    caches filled under other caps would miss proteins it keeps, so the caps must reach the
    loader exactly as main.py hands them over.
    """
    captured = {}

    def stop_before_loading(self):
        captured['data_args'] = self.data_args
        raise _StopBeforeLoading

    monkeypatch.setattr(precompute.DataMixin, 'get_data', stop_before_loading)
    monkeypatch.setattr(sys, 'argv', [
        'precompute_sae_and_dense.py', '--model_names', 'ESMC-600-SAE', '--sae_layer', '27',
        '--sae_k', '64', '--sae_codebook_dim', '16384', '--data_names', 'DeepLoc-2', *cap_argv,
    ])

    with pytest.raises(_StopBeforeLoading):
        precompute.main()

    data_args = captured['data_args']
    assert (data_args.split_row_caps, data_args.split_row_cap_seed) == expected


@pytest.mark.parametrize('dna_argv, expected', [([], False), (['--dna_to_aa'], True)])
def test_dna_translation_reaches_the_loader(monkeypatch, dna_argv, expected):
    """A --dna_to_aa run of main.py embeds the translated proteins, which are its cache keys.

    Caches filled from the DNA letters themselves would hold none of them, so the translation
    must reach the loader exactly as main.py sets it.
    """
    captured = {}

    def stop_before_loading(self):
        captured['mixin'] = self
        raise _StopBeforeLoading

    monkeypatch.setattr(precompute.DataMixin, 'get_data', stop_before_loading)
    monkeypatch.setattr(sys, 'argv', [
        'precompute_sae_and_dense.py', '--model_names', 'ESMC-600-SAE', '--sae_layer', '27',
        '--sae_k', '64', '--sae_codebook_dim', '16384', '--data_names', 'DeepLoc-2', *dna_argv,
    ])

    with pytest.raises(_StopBeforeLoading):
        precompute.main()

    mixin = captured['mixin']
    assert (mixin.data_args.dna_to_aa, mixin._dna_to_aa) == (expected, expected)
    assert mixin._active_translation_mode() == ('dna_to_aa' if expected else None)


def test_an_out_of_range_dense_index_is_rejected_before_anything_loads(monkeypatch):
    """ESMC-600 returns 37 hidden states; asking for a 38th must cost neither data nor weights."""
    def stop_before_loading(self):
        raise _StopBeforeLoading

    def stop_before_weights(*args, **kwargs):
        raise _StopBeforeLoading

    monkeypatch.setattr(precompute.DataMixin, 'get_data', stop_before_loading)
    monkeypatch.setattr(precompute, 'build_esmc_sae_model', stop_before_weights)
    monkeypatch.setattr(sys, 'argv', [
        'precompute_sae_and_dense.py', '--model_names', 'ESMC-600-SAE', '--sae_layer', '27',
        '--sae_k', '64', '--sae_codebook_dim', '16384', '--data_names', 'DeepLoc-2',
        '--dense_hidden_state_indices', '27', '37',
    ])

    with pytest.raises(ValueError, match=r'takes -37 to 36; \[37\]'):
        precompute.main()
