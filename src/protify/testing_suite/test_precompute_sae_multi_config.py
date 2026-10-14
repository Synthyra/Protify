"""Filling a whole sparsity by codebook-width grid, and dense caches at any depth, in one pass.

Every sparse autoencoder trained at one layer reads the same residual stream, so a grid of
configurations needs one ESMC forward per batch and one cheap encode per cell, not one
forward per cell. The forward that returns that stream runs every block, so dense caches at
the SAE's layer and at the final normalized state come out of it too. That single-pass
property is the reason this precompute exists, so the tests below count backbone forwards
exactly rather than asserting a bound.

The rest pins what makes the caches usable at all: every cell and every dense cache keys
the store its own single-configuration run would have keyed, each dense cache holds its own
depth's pooling, resumption stays per cache when the caches disagree about what they already
hold, and each SAE cache takes its own checkpoint's storage decision, because sparsity and
codebook width move the active fraction across the coordinate-list threshold.

The backbone, its codebooks, and the tokenizer are stand-ins that implement the contract
`fill_caches` uses. The cache writers, the feature stores, and the dense pooler are real.
"""

import os
import shutil
import sys
import pytest
import torch

from pathlib import Path

from fastplms.features import CSR, DENSE, FeatureStore


try:
    from src.protify import feature_cache
    from src.protify import precompute_sae_and_dense as precompute
    from src.protify.base_models import esmc, esmc_sae
    from src.protify.base_models.esmc_sae import ESMC_SAE_COVERAGE, parse_sae_model_name, resolve_sae_model_name
    from src.protify.base_models.get_base_models import resolve_data_max_length
    from src.protify.embedder import EmbeddingArguments
    from src.protify.precompute_sae_and_dense import (
        DENSE_BACKBONE_NAME,
        DENSE_POOLING_TYPES,
        DenseCache,
        PooledCache,
        SaeCache,
        _CacheWriter,
        codebook_weight_bytes,
        dense_descriptor,
        dense_hidden_state_indices,
        fill_caches,
        parse_pooling_sets,
        pooling_union,
        sae_configuration_names,
        sae_descriptor,
        select_pooling_columns,
    )
except ImportError:
    from protify import feature_cache
    from protify import precompute_sae_and_dense as precompute
    from protify.base_models import esmc, esmc_sae
    from protify.base_models.esmc_sae import ESMC_SAE_COVERAGE, parse_sae_model_name, resolve_sae_model_name
    from protify.base_models.get_base_models import resolve_data_max_length
    from protify.embedder import EmbeddingArguments
    from protify.precompute_sae_and_dense import (
        DENSE_BACKBONE_NAME,
        DENSE_POOLING_TYPES,
        DenseCache,
        PooledCache,
        SaeCache,
        _CacheWriter,
        codebook_weight_bytes,
        dense_descriptor,
        dense_hidden_state_indices,
        fill_caches,
        parse_pooling_sets,
        pooling_union,
        sae_configuration_names,
        sae_descriptor,
        select_pooling_columns,
    )


MAX_LENGTH = 2048
# Five checkpoints at ESMC-300's depth layer, so each configuration keys a store of its own. Each
# k / codebook_dim is above the coordinate-list threshold, so their stores are dense.
SAE_NAMES = (
    'ESMC-300-SAE-l23-k64-c8192',
    'ESMC-300-SAE-l23-k128-c8192',
    'ESMC-300-SAE-l23-k256-c8192',
    'ESMC-300-SAE-l23-k512-c8192',
    'ESMC-300-SAE-l23-k512-c16384',
)
HIDDEN_SIZE = 8
CODEBOOK_DIM = 6
# The stand-in returns NUM_HIDDEN_LAYERS + 1 hidden states, as FastPLMs does: the input to
# each block, then the final normalized state.
NUM_HIDDEN_LAYERS = 4
SAE_LAYER = 2
# Hidden state i is the token ids plus LAYER_OFFSET * i, so a pooled mean names its depth.
LAYER_OFFSET = 100.0


class _CountingBackbone:
    """Stand-in for the ESMC backbone that records how often it is actually run.

    It implements the one method `fill_caches` calls, plus the `to` and `eval` that `main`
    calls on a real model. `pooled` stands for the attached configuration's codebook features
    and `hidden_states` for every state of the pass: the SAE layer's, which every other
    configuration reads, and whichever ones the dense caches pool. All are derived from the
    token ids, so a stored vector identifies the protein it came from, and each state adds
    its own multiple of LAYER_OFFSET, so a stored mean also identifies the depth.
    """

    def __init__(
        self,
        hidden_size: int = HIDDEN_SIZE,
        codebook_dim: int = CODEBOOK_DIM,
        num_hidden_layers: int = NUM_HIDDEN_LAYERS,
    ) -> None:
        self.hidden_size = hidden_size
        self.codebook_dim = codebook_dim
        self.num_hidden_layers = num_hidden_layers
        self.sparse_storage = False
        self.forward_count = 0

    def to(self, device: torch.device) -> '_CountingBackbone':
        return self

    def eval(self) -> '_CountingBackbone':
        return self

    def sae_and_hidden_states(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # input_ids: (b, l); attention_mask: (b, l)
        self.forward_count += 1
        batch_size, length = input_ids.shape
        token_states = input_ids.unsqueeze(-1).to(torch.float32)  # (b, l, 1)
        token_states = token_states.expand(batch_size, length, self.hidden_size)  # (b, l, d)
        hidden_states = tuple(
            token_states + LAYER_OFFSET * index for index in range(self.num_hidden_layers + 1)
        )  # n_layers + 1 of (b, l, d)
        signature = (input_ids * attention_mask).sum(dim=1, keepdim=True).to(torch.float32)  # (b, 1)
        pooled = signature.expand(batch_size, self.codebook_dim).contiguous()  # (b, c)
        return pooled, hidden_states


class _CountingEncoder:
    """Stand-in for one further configuration, which reads residues rather than tokens."""

    def __init__(self, codebook_dim: int = CODEBOOK_DIM, scale: float = 1.0) -> None:
        self.codebook_dim = codebook_dim
        self.scale = scale
        self.sparse_storage = False
        self.call_count = 0

    def __call__(self, residues: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # residues: (b, l, d); attention_mask: (b, l)
        self.call_count += 1
        signature = (residues[:, :, 0] * attention_mask).sum(dim=1, keepdim=True)  # (b, 1)
        signature = signature * self.scale  # (b, 1)
        return signature.expand(residues.shape[0], self.codebook_dim).contiguous()  # (b, c)


class _ResidueTokenizer:
    """The narrow slice of the ESMC tokenizer `fill_caches` uses.

    One id per residue between a start and an end token, right padded to the longest
    sequence in the batch, so the mask and the two special tokens behave as they do in a
    real run.
    """

    pooling_token_id = None
    start_id = 1
    end_id = 2

    def __call__(self, sequences, padding='longest', truncation=True, max_length=64, return_tensors='pt'):
        encoded = [
            [self.start_id, *(ord(residue) for residue in sequence[: max_length - 2]), self.end_id]
            for sequence in sequences
        ]
        width = max(len(tokens) for tokens in encoded)
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)  # (b, l)
        attention_mask = torch.zeros(len(encoded), width, dtype=torch.long)  # (b, l)
        for row, tokens in enumerate(encoded):
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row, : len(tokens)] = 1
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


def _sae_caches(tmp_path, encoders, pooling_sets=(('max',),)) -> list:
    """One cache per configuration and pooling set, `None` marking the attached encoder."""
    return [
        SaeCache(
            encoder=encoder,
            codebook_dim=CODEBOOK_DIM,
            pooled_caches=tuple(
                PooledCache(
                    _CacheWriter(str(tmp_path), sae_descriptor(SAE_NAMES[index], pooling, MAX_LENGTH)),
                    pooling,
                )
                for pooling in pooling_sets
            ),
        )
        for index, encoder in enumerate(encoders)
    ]


def _dense_caches(tmp_path, indices=(SAE_LAYER,)) -> list:
    """One dense cache per hidden-state index, its store keyed by the index it pools."""
    return [
        DenseCache(_CacheWriter(str(tmp_path), dense_descriptor('ESMC-300', index, MAX_LENGTH)), index)
        for index in indices
    ]


def _prefill(writer, sequences, rows) -> None:
    """Commit rows a cache already holds, as an earlier pass would have left them."""
    writer.write(list(sequences), rows)
    writer.close()


def _stored(writer, sequences) -> dict:
    """The rows of these proteins in a cache, read back through the reader every run uses."""
    rows = feature_cache.read_embeddings(writer.directory, sequences)  # a protein the cache lacks raises KeyError
    return {sequence: row.reshape(-1) for sequence, row in rows.items()}


def _held(writer, sequences) -> set:
    """Which of these proteins a cache holds."""
    return set(sequences) - writer.missing(sequences)


def _token_ids(sequence: str) -> torch.Tensor:
    """The ids `_ResidueTokenizer` gives one protein, special tokens included."""
    tokens = [_ResidueTokenizer.start_id, *(ord(residue) for residue in sequence), _ResidueTokenizer.end_id]
    return torch.tensor(tokens, dtype=torch.float32)  # (l,)


def _close(caches, dense_caches) -> None:
    for cache in caches:
        for pooled in cache.pooled_caches:
            pooled.writer.close()
    for dense in dense_caches:
        dense.writer.close()


def test_one_backbone_pass_per_batch_whatever_the_grid_size(tmp_path):
    """The acceptance criterion: backbone forwards equal batches, never batches times cells.

    A refactor that reintroduced a pass per configuration would still write correct caches
    and would still pass every other test in this file, which is why this counts the exact
    number rather than bounding it.
    """
    sequences = [f'MK{"A" * index}W' for index in range(12)]
    backbone = _CountingBackbone()
    side_encoders = [_CountingEncoder() for _ in range(4)]
    caches = _sae_caches(tmp_path, [None, *side_encoders])
    dense_caches = _dense_caches(tmp_path)

    embedded = fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=4,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )

    _close(caches, dense_caches)

    assert embedded == 12
    # 12 proteins at batch size 4 is 3 batches, and 5 configurations does not change that.
    assert backbone.forward_count == 3
    assert [encoder.call_count for encoder in side_encoders] == [3, 3, 3, 3]
    for cache in caches:
        assert len(_held(cache.pooled_caches[0].writer, sequences)) == 12


def test_the_dense_cache_is_filled_from_that_same_pass(tmp_path):
    """The dense baseline reads the residues the codebooks read, so it costs no extra pass."""
    sequences = [f'MK{"A" * index}W' for index in range(6)]
    backbone = _CountingBackbone()
    side_encoders = [_CountingEncoder() for _ in range(3)]
    caches = _sae_caches(tmp_path, [None, *side_encoders])
    dense_caches = _dense_caches(tmp_path)

    fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=3,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )

    _close(caches, dense_caches)

    assert backbone.forward_count == 2
    stored_dense = _stored(dense_caches[0].writer, sequences)
    assert set(stored_dense) == set(sequences)
    # mean and var concatenated over the hidden state, which is what DENSE_POOLING_TYPES asks for.
    assert len(DENSE_POOLING_TYPES) == 2
    for vector in stored_dense.values():
        assert vector.numel() == len(DENSE_POOLING_TYPES) * HIDDEN_SIZE


def test_nothing_is_embedded_when_every_cache_already_holds_every_protein(tmp_path):
    """A rerun over finished caches must cost zero backbone passes, not one per configuration."""
    sequences = ['MAAW', 'MCCW', 'MDDW']
    backbone = _CountingBackbone()
    caches = _sae_caches(tmp_path, [None, _CountingEncoder()])
    dense_caches = _dense_caches(tmp_path, (SAE_LAYER, -1))
    for cache in caches:
        _prefill(cache.pooled_caches[0].writer, sequences, torch.zeros(len(sequences), CODEBOOK_DIM))
    for dense in dense_caches:
        _prefill(dense.writer, sequences, torch.zeros(len(sequences), 2 * HIDDEN_SIZE))

    embedded = fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=2,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )

    assert embedded == 0
    assert backbone.forward_count == 0
    _close(caches, dense_caches)


def test_resumption_writes_only_the_caches_a_protein_is_missing_from(tmp_path):
    """Cells disagree about what they hold, so a protein is embedded once and written where missing."""
    sequences = ['MAAW', 'MCCW', 'MDDW', 'MEEW', 'MFFW', 'MGGW']
    backbone = _CountingBackbone()
    side_encoder = _CountingEncoder()
    caches = _sae_caches(tmp_path, [None, side_encoder])
    dense_caches = _dense_caches(tmp_path)
    _prefill(dense_caches[0].writer, sequences, torch.zeros(len(sequences), 2 * HIDDEN_SIZE))

    # A value no encode here can produce, so a rewrite of a row that was already present
    # would be visible rather than hidden behind an identical recomputation.
    sentinel = torch.full((2, CODEBOOK_DIM), -99.0)
    _prefill(caches[0].pooled_caches[0].writer, ['MAAW', 'MCCW'], sentinel)
    _prefill(caches[1].pooled_caches[0].writer, ['MAAW', 'MDDW'], sentinel)

    embedded = fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=3,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )

    _close(caches, dense_caches)

    # MAAW is the only protein in both caches, so it is the only one not embedded again.
    assert embedded == 5
    assert backbone.forward_count == 2

    attached = _stored(caches[0].pooled_caches[0].writer, sequences)
    side = _stored(caches[1].pooled_caches[0].writer, sequences)
    assert set(attached) == set(sequences)
    assert set(side) == set(sequences)

    assert torch.allclose(attached['MAAW'], sentinel[0])
    # MCCW was embedded for the second cache, and must not have been rewritten into the first.
    assert torch.allclose(attached['MCCW'], sentinel[0])
    assert not torch.allclose(attached['MDDW'], sentinel[0])
    assert torch.allclose(side['MDDW'], sentinel[0])
    assert not torch.allclose(side['MCCW'], sentinel[0])


def test_the_final_state_cache_is_filled_from_the_same_pass(tmp_path):
    """The SAE forward already runs every block, so the final state costs no second pass.

    A dense cache at the SAE's layer and one at the final normalized state still take one
    backbone forward per batch between them, and every cache ends up holding every protein.
    """
    sequences = [f'MK{"A" * index}W' for index in range(10)]
    backbone = _CountingBackbone()
    side_encoder = _CountingEncoder()
    caches = _sae_caches(tmp_path, [None, side_encoder])
    dense_caches = _dense_caches(tmp_path, (SAE_LAYER, -1))

    embedded = fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=4,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )
    _close(caches, dense_caches)

    assert embedded == 10
    # 10 proteins at batch size 4 is 3 batches, and a second dense depth does not make it 6.
    assert backbone.forward_count == 3
    assert side_encoder.call_count == 3
    for dense in dense_caches:
        assert set(_stored(dense.writer, sequences)) == set(sequences)


def test_each_dense_cache_holds_its_own_depths_pooling(tmp_path):
    """A cache named for one depth must hold that depth's mean and variance, not another's.

    Each stand-in state is the token ids shifted by its own multiple of LAYER_OFFSET, so the
    stored mean names the depth it was pooled from, and the variance, which a shift cannot
    move, checks that it was pooled over the protein's own tokens.
    """
    sequences = ['MAAW', 'MCCCW', 'MDW']
    backbone = _CountingBackbone()
    caches = _sae_caches(tmp_path, [None, _CountingEncoder()])
    dense_caches = _dense_caches(tmp_path, (0, SAE_LAYER, -1))

    fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=2,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )
    _close(caches, dense_caches)

    assert backbone.forward_count == 2
    by_index = {dense.hidden_state_index: dense for dense in dense_caches}
    # Index -1 is the final normalized state, the last of the n_layers + 1.
    for index, depth in ((0, 0), (SAE_LAYER, SAE_LAYER), (-1, NUM_HIDDEN_LAYERS)):
        stored = _stored(by_index[index].writer, sequences)
        for sequence in sequences:
            token_ids = _token_ids(sequence)  # (l,)
            expected = torch.cat([
                torch.full((HIDDEN_SIZE,), token_ids.mean().item() + LAYER_OFFSET * depth),
                torch.full((HIDDEN_SIZE,), token_ids.var(unbiased=False).item()),
            ])  # (2 * d,) mean then var
            torch.testing.assert_close(stored[sequence], expected, rtol=0, atol=1e-3)


def test_resumption_is_per_dense_cache(tmp_path):
    """A protein the layer cache holds but the final-state cache lacks is written only where missing."""
    sequences = ['MAAW', 'MCCW', 'MDDW']
    backbone = _CountingBackbone()
    caches = _sae_caches(tmp_path, [None])
    _prefill(caches[0].pooled_caches[0].writer, sequences, torch.zeros(len(sequences), CODEBOOK_DIM))
    layer_cache, final_cache = _dense_caches(tmp_path, (SAE_LAYER, -1))
    # A value no pooling here can produce, so a rewrite of a present row would show.
    sentinel = torch.full((len(sequences), 2 * HIDDEN_SIZE), -99.0)
    _prefill(layer_cache.writer, sequences, sentinel)
    _prefill(final_cache.writer, ['MAAW'], sentinel[:1])

    embedded = fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        [layer_cache, final_cache],
        sequences,
        batch_size=8,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
    )
    _close(caches, [layer_cache, final_cache])

    assert embedded == 2
    assert backbone.forward_count == 1
    layer = _stored(layer_cache.writer, sequences)
    final = _stored(final_cache.writer, sequences)
    for sequence in sequences:
        assert torch.allclose(layer[sequence], sentinel[0])
    assert set(final) == set(sequences)
    assert torch.allclose(final['MAAW'], sentinel[0])
    assert not torch.allclose(final['MCCW'], sentinel[0])


def test_dense_indices_are_checked_against_the_backbone_and_kept_once():
    """Any Python index into the n_layers + 1 hidden states is valid, and each opens one cache."""
    selection = parse_sae_model_name('ESMC-600-SAE-l27-k64-c16384')
    state_count = ESMC_SAE_COVERAGE['ESMC-600'].num_hidden_layers + 1

    assert state_count == 37
    # Omitted, the one dense cache stays at the SAE's own layer.
    assert dense_hidden_state_indices(None, selection) == [27]
    assert dense_hidden_state_indices([27, -1, 27, -1], selection) == [27, -1]
    assert dense_hidden_state_indices([0, state_count - 1, -state_count], selection) == [0, 36, -37]
    for outside in (state_count, -state_count - 1):
        with pytest.raises(ValueError, match='takes -37 to 36'):
            dense_hidden_state_indices([27, outside], selection)


def test_the_final_state_cache_keys_are_the_keys_a_plain_run_reads():
    """Truncation decides the stored sequence, which is the cache key.

    This pass truncates for the SAE names, and a plain ESMC-600 run truncates for its own
    name. The two limits must agree, or the final-state cache would be filled under keys the
    plain run never looks up.
    """
    names = sae_configuration_names('ESMC-600-SAE', 27, (64,), (16384,))
    dense_name = DENSE_BACKBONE_NAME[parse_sae_model_name(names[0]).backbone]

    assert dense_name == 'ESMC-600'
    for max_length in (512, 2048):
        assert resolve_data_max_length([dense_name], max_length, trim=False) == (
            resolve_data_max_length(names, max_length, trim=False)
        )


def _plain_run_stores(root, hidden_state_indices=(27, -1)):
    """The stores the ordinary runs of the ESMC-600 arms read, found the way a plain run finds them."""
    sae_name = 'ESMC-600-SAE-l27-k64-c16384'
    sae = EmbeddingArguments(embedding_save_dir=str(root), embedding_pooling_types=['max'], max_length=MAX_LENGTH)
    stores = {'sae': sae.feature_directory(sae_name)}
    for index in hidden_state_indices:
        plain = EmbeddingArguments(
            embedding_save_dir=str(root), embedding_pooling_types=['mean', 'var'],
            embedding_hidden_state_index=index, max_length=MAX_LENGTH,
        )
        stores[index] = plain.feature_directory('ESMC-600')
    return stores


def test_main_fills_the_layer_and_final_state_caches_under_their_ordinary_names(tmp_path, monkeypatch):
    """`--dense_hidden_state_indices 27 -1` writes the stores the two ordinary runs read, in one pass.

    Index -1 must land in the store a plain ESMC-600 run pools its final state into, holding the
    final state's pooling rather than layer 27's.
    """
    sequences = [f'MK{"A" * index}W' for index in range(5)]
    num_hidden_layers = ESMC_SAE_COVERAGE['ESMC-600'].num_hidden_layers
    backbone = _CountingBackbone(num_hidden_layers=num_hidden_layers)
    monkeypatch.setattr(precompute.DataMixin, 'get_data', lambda self: ({}, list(sequences)))
    monkeypatch.setattr(precompute, 'build_esmc_sae_model', lambda name, **kwargs: (backbone, _ResidueTokenizer()))
    # CPU whatever this machine has.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'argv', [
        'precompute_sae_and_dense.py', '--model_names', 'ESMC-600-SAE', '--sae_layer', '27',
        '--sae_k', '64', '--sae_codebook_dim', '16384', '--data_names', 'DeepLoc-2',
        '--embedding_save_dir', str(tmp_path), '--embedding_batch_size', '2',
        '--dense_hidden_state_indices', '27', '-1',
    ])

    precompute.main()

    stores = _plain_run_stores(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        os.path.basename(directory) for directory in stores.values()
    )
    assert [os.path.basename(directory).rsplit('__', 1)[0] for directory in stores.values()] == [
        'protify__ESMC-600-SAE-l27-k64-c16384__vector__max__len2048__float32',
        'protify__ESMC-600__vector__mean-var__hs27__len2048__float32',
        'protify__ESMC-600__vector__mean-var__len2048__float32',
    ]
    # 5 proteins at batch size 2 is 3 batches for all three caches together.
    assert backbone.forward_count == 3
    layer = feature_cache.read_embeddings(stores[27], sequences)
    final = feature_cache.read_embeddings(stores[-1], sequences)
    assert set(layer) == set(final) == set(sequences)
    layer = {sequence: row.reshape(-1) for sequence, row in layer.items()}
    final = {sequence: row.reshape(-1) for sequence, row in final.items()}
    for sequence in sequences:
        # Only the means move with depth: final minus layer 27 is 36 - 27 offsets.
        torch.testing.assert_close(
            final[sequence][:HIDDEN_SIZE] - layer[sequence][:HIDDEN_SIZE],
            torch.full((HIDDEN_SIZE,), LAYER_OFFSET * (num_hidden_layers - 27)),
            rtol=0,
            atol=1e-3,
        )


def test_shards_fill_disjoint_caches_that_together_hold_every_protein(tmp_path, monkeypatch):
    """`--shard i/n` splits one pass into n runs, each into its own directory.

    Merging the n directories' caches must give the caches the whole pass writes: every
    protein once, in every cache the pass fills, so no protein is embedded twice or missed.
    A store merges by copying the shards' segment directories into one store and opening it
    once for writing, which rebuilds its index.
    """
    # Three proteins tie at length 4, which the loader orders differently between processes.
    sequences = [f'MK{"A" * index}W' for index in range(7)] + ['MKCW', 'MKDW']
    num_hidden_layers = ESMC_SAE_COVERAGE['ESMC-600'].num_hidden_layers
    monkeypatch.setattr(precompute.DataMixin, 'get_data', lambda self: ({}, list(sequences)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    held = []
    shard_stores = []
    for index in (1, 2, 3):
        backbone = _CountingBackbone(num_hidden_layers=num_hidden_layers)
        # Bound as a default, so each shard's run counts its own backbone's batches.
        monkeypatch.setattr(
            precompute, 'build_esmc_sae_model', lambda name, built=backbone, **kwargs: (built, _ResidueTokenizer())
        )
        directory = tmp_path / f'shard-{index}'
        monkeypatch.setattr(sys, 'argv', [
            'precompute_sae_and_dense.py', '--model_names', 'ESMC-600-SAE', '--sae_layer', '27',
            '--sae_k', '64', '--sae_codebook_dim', '16384', '--data_names', 'DeepLoc-2',
            '--embedding_save_dir', str(directory), '--embedding_batch_size', '2',
            '--dense_hidden_state_indices', '-1', '--shard', f'{index}/3',
        ])

        precompute.main()

        stores = _plain_run_stores(directory, hidden_state_indices=(-1,))
        assert sorted(path.name for path in directory.iterdir()) == sorted(
            os.path.basename(store) for store in stores.values()
        )
        stored = [
            set(sequences) - set(feature_cache.missing_sequences(store, sequences)) for store in stores.values()
        ]
        assert stored[0] == stored[1], 'each shard fills every cache for the proteins it holds'
        held.append(stored[0])
        shard_stores.append(stores)
        # 3 proteins at batch size 2 is 2 batches.
        assert backbone.forward_count == 2

    assert [len(proteins) for proteins in held] == [3, 3, 3]
    assert set().union(*held) == set(sequences)
    assert sum(len(proteins) for proteins in held) == len(sequences), 'no protein is in two shards'

    # Merge shards 2 and 3 into shard 1: copy their segments, then open each store once for writing.
    for name, target in shard_stores[0].items():
        target = Path(target)
        for other in shard_stores[1:]:
            for segment in (Path(other[name]) / 'segments').iterdir():
                shutil.copytree(segment, target / 'segments' / segment.name)
        FeatureStore.open(target.parent, FeatureStore.read_only(target).spec)
        assert not feature_cache.missing_sequences(target, sequences)
        assert feature_cache.read_embeddings(target, sequences).keys() == set(sequences)


def test_every_grid_cell_names_the_cache_its_single_configuration_run_would_name(tmp_path):
    """The whole point: a later ordinary run must find these stores and skip embedding."""
    k_values = (16, 64, 256, 512)
    codebook_dims = (8192, 131072)
    plain_run = EmbeddingArguments(
        embedding_save_dir=str(tmp_path), embedding_pooling_types=['max'], max_length=MAX_LENGTH
    )

    names = sae_configuration_names('ESMC-6B-SAE', 60, k_values, codebook_dims)
    swept = [feature_cache.feature_key(sae_descriptor(name, ('max',), MAX_LENGTH)) for name in names]
    single = [
        feature_cache.feature_key(plain_run.cache_descriptor(
            resolve_sae_model_name('ESMC-6B-SAE', sae_layer=60, sae_k=k, sae_codebook_dim=codebook_dim)
        ))
        for k in k_values
        for codebook_dim in codebook_dims
    ]

    assert swept == single
    assert len(set(swept)) == len(names), 'every cell keys a store of its own'
    assert any(
        key.startswith('protify__ESMC-6B-SAE-l60-k512-c131072__vector__max__len2048__float32__') for key in swept
    )


def test_the_grid_is_the_cross_product_with_sparsity_outermost():
    names = sae_configuration_names('ESMC-6B-SAE', 60, (64, 256), (8192, 16384))

    assert names == [
        'ESMC-6B-SAE-l60-k64-c8192',
        'ESMC-6B-SAE-l60-k64-c16384',
        'ESMC-6B-SAE-l60-k256-c8192',
        'ESMC-6B-SAE-l60-k256-c16384',
    ]


def test_a_repeated_sweep_value_does_not_open_the_same_cache_twice():
    names = sae_configuration_names('ESMC-6B-SAE', 60, (64, 64), (8192, 8192))

    assert names == ['ESMC-6B-SAE-l60-k64-c8192']


def test_an_omitted_axis_resolves_exactly_as_a_single_configuration_run_did():
    """Leaving --sae_k or --sae_codebook_dim off has to keep the published default."""
    assert sae_configuration_names('ESMC-6B-SAE', None, [None], [None]) == ['ESMC-6B-SAE-l60-k64-c8192']
    assert sae_configuration_names('ESMC-6B-SAE', 60, (512,), [None]) == ['ESMC-6B-SAE-l60-k512-c8192']


def test_an_unpublished_cell_is_rejected_before_any_cache_opens():
    """Away from the depth layer Biohub trains one sparsity and a short list of widths."""
    with pytest.raises(ValueError, match='only publishes k=64 at codebook widths'):
        sae_configuration_names('ESMC-6B-SAE', 30, (64, 512), (16384,))


def test_a_sparsity_biohub_never_trained_is_rejected():
    with pytest.raises(ValueError, match='--sae_k 7 is not published'):
        sae_configuration_names('ESMC-6B-SAE', 60, (7,), (8192,))


def test_a_sweep_contradicting_an_already_qualified_alias_is_rejected():
    with pytest.raises(ValueError, match='already fixes its SAE checkpoint'):
        sae_configuration_names('ESMC-6B-SAE-l60-k512-c131072', 60, (16, 512), (131072,))


def test_the_dataset_truncation_limit_does_not_move_with_the_grid_size():
    """Truncation decides the stored sequence string, which is the cache key.

    Every cell shares one backbone, so widening the grid must not shift the limit and split
    the caches across two sets of keys.
    """
    names = sae_configuration_names('ESMC-6B-SAE', 60, (16, 512), (8192, 131072))

    assert resolve_data_max_length(names, 2048, trim=False) == 2048
    assert resolve_data_max_length(names[:1], 2048, trim=False) == 2048


def test_each_cache_uses_its_own_checkpoint_storage_decision(tmp_path):
    """One grid straddles the coordinate-list threshold, so the layout cannot be shared."""
    dense_cell = parse_sae_model_name('ESMC-6B-SAE-l60-k512-c8192')
    sparse_cell = parse_sae_model_name('ESMC-6B-SAE-l60-k64-c131072')
    assert not dense_cell.sparse_storage
    assert sparse_cell.sparse_storage

    mostly_zero = torch.zeros(1, 64)  # (1, c)
    mostly_zero[0, 7] = 3.5
    layouts = {}
    for name, selection in (('dense_cell', dense_cell), ('sparse_cell', sparse_cell)):
        model_name = f'ESMC-6B-SAE-l60-k{selection.k}-c{selection.codebook_dim}'
        writer = _CacheWriter(str(tmp_path), sae_descriptor(model_name, ('max',), MAX_LENGTH))
        writer.write(['MAAW'], mostly_zero)
        writer.close()
        layouts[name] = FeatureStore.read_only(writer.directory).spec.layout
        # Whichever layout holds the row, it reads back exactly.
        assert torch.equal(_stored(writer, ['MAAW'])['MAAW'], mostly_zero[0])

    assert layouts == {'dense_cell': DENSE, 'sparse_cell': CSR}


def test_the_residue_encoder_maxes_the_real_fastplms_encoder_over_every_token(monkeypatch):
    """The added configurations run FastPLMs' own encoder, and Protify only routes and pools.

    The shard here is a randomly initialized `ESMplusplusSAELayer` standing in for a published
    one at a width that keeps the reference readable; what is under test is that residues reach
    that encoder correctly and that the result is max-pooled over every attended token, the
    leading and trailing special tokens included (canonical pooling covers l + 2 rows).
    """
    # FastPLMs is importable only once esmc_sae has installed the submodule path.
    from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import (
        ESMplusplusSAELayer,
        ESMplusplusSAEParams,
    )

    torch.manual_seed(0)
    hidden_size, codebook_dim = 4, 8
    shard = ESMplusplusSAELayer(ESMplusplusSAEParams(d_model=hidden_size, codebook_dim=codebook_dim, k=2, layer=60))
    with torch.no_grad():
        shard.W_enc.normal_()  # (d, c)
        shard.W_dec.normal_()  # (c, d)
        shard.b_dec.zero_()  # (d,)
    monkeypatch.setattr(
        esmc_sae, 'load_esmc_sae_layers', lambda repository, layers, **kwargs: {layers[0]: shard}
    )

    selection = parse_sae_model_name('ESMC-6B-SAE-l60-k64-c131072')
    encoder = esmc_sae.EsmcSaeResidueEncoder(selection, pooling_types=('max',))
    residues = torch.randn(2, 6, hidden_size)  # (b, l, d)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]])  # (b, l)

    pooled = encoder(residues, attention_mask)  # (b, c)

    expected = torch.zeros(2, codebook_dim)  # (b, c)
    for row, token_count in enumerate(attention_mask.sum(dim=1).tolist()):
        tokens = residues[row, :token_count]  # (l + 2, d), CLS and EOS kept
        magnitudes = shard(tokens).feature_magnitudes  # (l + 2, c)
        expected[row] = magnitudes.max(dim=0).values  # (c,)

    assert pooled.shape == (2, codebook_dim)
    assert torch.allclose(pooled, expected)
    # A padded position must not reach the encoder at all, so lengthening the padding cannot
    # move the answer.
    padded = torch.cat([residues, torch.randn(2, 3, hidden_size)], dim=1)  # (b, l + 3, d)
    padded_mask = torch.cat([attention_mask, torch.zeros(2, 3, dtype=torch.long)], dim=1)  # (b, l + 3)
    assert torch.allclose(encoder(padded, padded_mask), expected)


def test_the_sae_pass_returns_the_state_a_plain_run_pools():
    """Through the real ESM++ stack: the SAE pass's last state is the plain run's `last_hidden_state`.

    A randomly initialized three-block backbone with a codebook attached at layer 2 stands in
    for a published checkpoint. The final-state cache is a free copy of the plain run's cache
    only if index -1 of the SAE pass is exactly what the plain ESMC adapter pools, and side
    encoders match the attached codebook only if index `layer` is the stream it read.
    """
    # FastPLMs is importable only once esmc_sae has installed the submodule path.
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import ESMplusplusConfig, ESMplusplusModel
    from fastplms.models.esm_plusplus.modeling_esm_plusplus_sae import (
        ESMplusplusSAELayer,
        ESMplusplusSAEParams,
    )

    torch.manual_seed(0)
    hidden_size, num_hidden_layers, layer = 16, 3, 2
    config = ESMplusplusConfig(
        hidden_size=hidden_size, num_attention_heads=2, num_hidden_layers=num_hidden_layers, attn_backend='sdpa'
    )
    backbone = ESMplusplusModel(config).eval()
    shard = ESMplusplusSAELayer(ESMplusplusSAEParams(d_model=hidden_size, codebook_dim=8, k=2, layer=layer))
    with torch.no_grad():
        shard.W_enc.normal_()  # (d, c)
        shard.W_dec.normal_()  # (c, d)
        shard.b_dec.zero_()  # (d,)
    backbone.add_sae_models([shard])

    # The adapters without their loaders, which would download a checkpoint.
    sae_model = esmc_sae.EsmcSaeForEmbedding.__new__(esmc_sae.EsmcSaeForEmbedding)
    torch.nn.Module.__init__(sae_model)
    sae_model.esm, sae_model.sae_key, sae_model.pooling_types = backbone, f'layer{layer}', ('max',)
    plain_model = esmc.ESMplusplusForEmbedding.__new__(esmc.ESMplusplusForEmbedding)
    torch.nn.Module.__init__(plain_model)
    plain_model.esm = backbone
    side_encoder = esmc_sae.EsmcSaeResidueEncoder.__new__(esmc_sae.EsmcSaeResidueEncoder)
    torch.nn.Module.__init__(side_encoder)
    side_encoder.sae, side_encoder.pooling_types = shard, ('max',)

    # Start 0, end 2, pad 1, as the ESM tokenizer writes them, and no mask token.
    input_ids = torch.tensor([[0, 5, 6, 7, 8, 2], [0, 9, 10, 2, 1, 1]])  # (b, l)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]])  # (b, l)
    with torch.inference_mode():
        pooled, hidden_states = sae_model.sae_and_hidden_states(input_ids, attention_mask)
        # pooled: (b, c); hidden_states: n_layers + 1 of (b, l, d)
        plain_state = plain_model(input_ids, attention_mask)  # (b, l, d), hidden_state_index -1
        re_encoded = side_encoder(hidden_states[layer], attention_mask)  # (b, c)

    assert len(hidden_states) == num_hidden_layers + 1
    torch.testing.assert_close(hidden_states[-1], plain_state, rtol=0, atol=0)
    torch.testing.assert_close(re_encoded, pooled)


def test_resident_codebook_weight_counts_every_cell():
    """The grid's cost is codebook weight, which is what decides how wide a card can go."""
    names = sae_configuration_names('ESMC-6B-SAE', 60, (64,), (8192, 131072))
    selections = [parse_sae_model_name(name) for name in names]

    hidden_size = ESMC_SAE_COVERAGE['ESMC-6B'].hidden_size
    # Encoder and decoder, two bytes per bfloat16 parameter.
    expected = 2 * hidden_size * (8192 + 131072) * 2
    assert codebook_weight_bytes(selections, torch.bfloat16) == expected
    assert codebook_weight_bytes(selections[:1], torch.bfloat16) < expected


def test_several_pooling_sets_are_filled_from_one_encode(tmp_path):
    """A second pooling set costs a column slice, not another pass over the backbone.

    The reduction a cache stores is a subset of the union the encoder computes, so a run
    that wants both a max cache and a max-then-mean cache pays for one encode and writes
    the max columns twice.
    """
    sequences = [f'MK{"A" * index}W' for index in range(8)]
    backbone = _CountingBackbone(codebook_dim=CODEBOOK_DIM * 2)  # union of two reductions
    side_encoder = _CountingEncoder(codebook_dim=CODEBOOK_DIM * 2)
    caches = _sae_caches(
        tmp_path, [None, side_encoder], pooling_sets=(('max',), ('max', 'mean'))
    )
    dense_caches = _dense_caches(tmp_path)

    fill_caches(
        backbone,
        _ResidueTokenizer(),
        caches,
        dense_caches,
        sequences,
        batch_size=4,
        max_length=32,
        device=torch.device('cpu'),
        sae_layer=SAE_LAYER,
        union=('max', 'mean'),
    )
    _close(caches, dense_caches)

    # Two batches, and two pooling sets do not make it four.
    assert backbone.forward_count == 2
    assert side_encoder.call_count == 2

    narrow = _stored(caches[0].pooled_caches[0].writer, sequences)  # max only
    wide = _stored(caches[0].pooled_caches[1].writer, sequences)  # max then mean
    assert set(narrow) == set(sequences)
    for sequence in sequences:
        assert narrow[sequence].shape == (CODEBOOK_DIM,)
        assert wide[sequence].shape == (2 * CODEBOOK_DIM,)
        # The max block is the leading columns of the union, so the narrow cache holds it
        # unchanged rather than holding some other reduction under the same name.
        torch.testing.assert_close(narrow[sequence], wide[sequence][:CODEBOOK_DIM])


def test_a_pooling_set_names_the_store_its_own_run_would_have_named(tmp_path):
    """Two caches from one pass must be the two stores two separate runs would have read."""
    sets = parse_pooling_sets(['max', 'max,mean'])
    assert sets == [('max',), ('max', 'mean')]
    assert pooling_union(sets) == ('max', 'mean')

    name = 'ESMC-6B-SAE-l60-k16-c131072'
    keys = {}
    for pooling in (('max',), ('max', 'mean'), ('mean', 'max')):
        plain_run = EmbeddingArguments(
            embedding_save_dir=str(tmp_path), embedding_pooling_types=list(pooling), max_length=MAX_LENGTH
        )
        key = feature_cache.feature_key(sae_descriptor(name, pooling, MAX_LENGTH))
        assert key == feature_cache.feature_key(plain_run.cache_descriptor(name))
        keys[pooling] = key

    # Each pooling set is its own store, and the order the reductions are joined in is part of the name,
    # because the row's columns follow it.
    assert len(set(keys.values())) == 3
    assert keys[('max',)].startswith('protify__ESMC-6B-SAE-l60-k16-c131072__vector__max__len2048__float32__')
    assert keys[('max', 'mean')].startswith(
        'protify__ESMC-6B-SAE-l60-k16-c131072__vector__max-mean__len2048__float32__'
    )


def test_selecting_a_reduction_reads_its_own_block():
    """A wrong offset here would store one reduction's numbers under another's name."""
    union = ('max', 'mean', 'sum')
    codebook_dim = 3
    pooled = torch.tensor([[1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 100.0, 200.0, 300.0]])  # (1, u * c)

    torch.testing.assert_close(
        select_pooling_columns(pooled, union, ('mean',), codebook_dim),
        torch.tensor([[10.0, 20.0, 30.0]]),
    )
    torch.testing.assert_close(
        select_pooling_columns(pooled, union, ('sum', 'max'), codebook_dim),
        torch.tensor([[100.0, 200.0, 300.0, 1.0, 2.0, 3.0]]),
    )
    # The whole union is returned as it stands, without a copy through cat.
    assert select_pooling_columns(pooled, union, union, codebook_dim) is pooled


def test_an_empty_pooling_set_is_rejected():
    with pytest.raises(ValueError, match="Empty pooling set"):
        parse_pooling_sets(['max', ''])
