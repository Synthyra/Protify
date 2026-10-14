"""The Embedder reads and writes the feature store, adopts old caches into it, and converts downloads.

What a run does with `--embedding_save_dir`: it finds the store of its own descriptor, embeds only the
sequences that store lacks, and commits what it embeds a window at a time. An old SQLite or `.pth` cache
of the same representation is converted into the store first, without re-embedding, and a file
downloaded from the Hub is converted the same way. The old files in these tests are written by the
test-local writer of the old bytes (`legacy_cache_writer`); the model and tokenizer are small stand-ins
that follow the calls `Embedder` makes, so no weights load.
"""

import functools
import gzip
import pytest
import torch

from pathlib import Path

from fastplms.features import FeatureStore, RAGGED


try:
    from src.protify import embedder as embedder_module
    from src.protify import feature_cache
    from src.protify.embedder import Embedder, EmbeddingArguments
    from src.protify.testing_suite.embedding_store_builder import build_store_for
    from src.protify.testing_suite.legacy_cache_writer import write_sqlite_cache, write_torch_cache
except ImportError:
    from protify import embedder as embedder_module
    from protify import feature_cache
    from protify.embedder import Embedder, EmbeddingArguments
    from protify.testing_suite.embedding_store_builder import build_store_for
    from protify.testing_suite.legacy_cache_writer import write_sqlite_cache, write_torch_cache


pytestmark = [
    pytest.mark.filterwarnings("ignore:Downloading embeddings is recommended"),
    pytest.mark.filterwarnings("ignore:'pin_memory' argument is set as true"),
]

MODEL = "ESM2-8"
WIDTH = 4
SEQUENCES = ["MKTAYIAKQR", "MKTAYIAKQRQISFVKSHFSRQ", "GGGGSGGGGS", "ACDEFGHIKL", "WWWWWWWWWWWWWW"]


def vectors(sequences=SEQUENCES, value=None, dtype=torch.float32):
    """One row of `WIDTH` values per sequence: distinct per sequence unless `value` fixes them."""
    return {
        sequence: torch.full((WIDTH,), float(value) if value is not None else float(index + 1)).to(dtype)
        for index, sequence in enumerate(sequences)
    }


def arguments(root, **overrides):
    settings = dict(embedding_save_dir=str(root), embedding_pooling_types=["mean"], max_length=2048)
    settings.update(overrides)
    return EmbeddingArguments(**settings)


def raw(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def committed(directory):
    """The sequences of `SEQUENCES` a store holds."""
    return set(SEQUENCES) - set(feature_cache.missing_sequences(directory, SEQUENCES))


class Tokenizer:
    """Residues to ids between a start and an end token, padded the way a Hugging Face tokenizer pads."""

    pooling_token_id = None

    def __call__(self, sequences, return_tensors="pt", padding="longest", truncation=True, max_length=64,
                 pad_to_multiple_of=None):
        encoded = [[1, *(ord(residue) for residue in sequence[: max_length - 2]), 2] for sequence in sequences]
        width = max_length if padding == "max_length" else max(len(tokens) for tokens in encoded)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        input_ids = torch.zeros(len(encoded), width, dtype=torch.long)  # (b, l)
        attention_mask = torch.zeros(len(encoded), width, dtype=torch.long)  # (b, l)
        for row, tokens in enumerate(encoded):
            input_ids[row, : len(tokens)] = torch.tensor(tokens)
            attention_mask[row, : len(tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class Model(torch.nn.Module):
    """Residue states `token + channel`, so a mean over a sequence is its mean token plus the channel."""

    def __init__(self, fail_on_call=None):
        super().__init__()
        self.calls = 0
        self.fail_on_call = fail_on_call

    def forward(self, input_ids, attention_mask, **kwargs):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("the model stopped")
        states = input_ids.unsqueeze(-1).to(torch.float32)  # (b, l, 1)
        return states + torch.arange(WIDTH, dtype=torch.float32, device=input_ids.device)  # (b, l, d)


def expected_mean(sequence):
    tokens = torch.tensor([1.0, *(float(ord(residue)) for residue in sequence), 2.0])  # (l,)
    return tokens.mean() + torch.arange(WIDTH, dtype=torch.float32)  # (d,)


@pytest.fixture(autouse=True)
def on_the_cpu(monkeypatch):
    """Every Embedder here runs on the CPU, whatever this machine has."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def embedder_for(monkeypatch):
    """`Embedder`s over a stand-in model, and the model they share, whose calls the test can count."""
    model = Model()
    loaded = []

    def get_base_model(name, dtype=None, model_path=None, pooling_types=None):
        loaded.append(name)
        return model, Tokenizer()

    monkeypatch.setattr(embedder_module, "get_base_model", get_base_model)

    def make(root, sequences=SEQUENCES, **overrides):
        settings = dict(padding="longest", max_length=64, embedding_batch_size=2)
        settings.update(overrides)
        return Embedder(arguments(root, **settings), sequences)

    return make, model, loaded


def test_reading_without_a_store_embeds_everything_and_creates_nothing(tmp_path):
    embedder = Embedder(arguments(tmp_path), SEQUENCES + SEQUENCES[:2])

    to_embed, save_path, rows = embedder._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES  # distinct, in the order given
    assert rows == {}
    assert save_path == embedder.args.feature_directory(MODEL)
    assert list(tmp_path.iterdir()) == []


def test_a_partial_store_leaves_only_its_missing_sequences_to_embed(tmp_path):
    args = arguments(tmp_path)
    held = vectors(SEQUENCES[:2])
    build_store_for(args, MODEL, held)

    to_embed, save_path, rows = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES[2:]
    assert set(rows) == set(SEQUENCES[:2])
    for sequence, row in held.items():
        assert raw(rows[sequence]) == raw(row)
    # A row read is the caller's own: writing to it does not change the store.
    rows[SEQUENCES[0]].zero_()
    assert raw(feature_cache.read_embeddings(save_path, SEQUENCES[:1])[SEQUENCES[0]]) == raw(held[SEQUENCES[0]])


def test_under_sql_the_rows_stay_on_disk(tmp_path):
    args = arguments(tmp_path, sql=True)
    build_store_for(args, MODEL, vectors(SEQUENCES[:2]))

    to_embed, save_path, rows = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES[2:]
    assert rows == {}
    assert feature_cache.has_feature(save_path)


def test_a_store_of_a_wider_dtype_serves_a_narrower_run(tmp_path):
    build_store_for(arguments(tmp_path), MODEL, vectors())  # float32
    narrow = arguments(tmp_path, embed_dtype=torch.float16)

    to_embed, save_path, rows = Embedder(narrow, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == []
    assert save_path == arguments(tmp_path).feature_directory(MODEL)
    assert {row.dtype for row in rows.values()} == {torch.float32}


def test_an_old_sqlite_cache_is_converted_into_the_store_and_left_in_place(tmp_path):
    args = arguments(tmp_path, sql=True)
    old_rows = vectors(SEQUENCES[:3], dtype=torch.float32)
    old = write_sqlite_cache(tmp_path / args.cache_filename(MODEL, extension="db"), old_rows, wal=False)
    before = old.read_bytes()

    to_embed, save_path, _ = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES[3:]
    assert old.read_bytes() == before
    stored = feature_cache.read_embeddings(save_path, SEQUENCES[:3])
    for sequence, row in old_rows.items():
        assert raw(stored[sequence]) == raw(row)
    segments = sorted(entry.name for entry in (Path(save_path) / "segments").iterdir())
    assert segments and all(name.startswith("convert-") for name in segments)


def test_an_old_pickled_cache_is_converted_and_its_float16_rows_widen_exactly(tmp_path):
    args = arguments(tmp_path)
    old_rows = vectors(SEQUENCES[:2], dtype=torch.float16)
    write_torch_cache(tmp_path / args.cache_filename(MODEL, extension="pth"), old_rows)

    to_embed, save_path, rows = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES[2:]
    for sequence, row in old_rows.items():
        assert rows[sequence].dtype == torch.float32
        assert torch.equal(rows[sequence], row.to(torch.float32))


@pytest.mark.parametrize(("sql", "winner"), [(True, "sqlite"), (False, "torch")])
def test_the_file_of_the_runs_own_mode_wins_where_both_formats_hold_a_sequence(tmp_path, sql, winner):
    args = arguments(tmp_path, sql=sql)
    sqlite_rows = {SEQUENCES[0]: torch.full((WIDTH,), 1.0), SEQUENCES[1]: torch.full((WIDTH,), 1.0)}
    torch_rows = {SEQUENCES[1]: torch.full((WIDTH,), 2.0), SEQUENCES[2]: torch.full((WIDTH,), 2.0)}
    write_sqlite_cache(tmp_path / args.cache_filename(MODEL, extension="db"), sqlite_rows, wal=False)
    write_torch_cache(tmp_path / args.cache_filename(MODEL, extension="pth"), torch_rows)

    to_embed, save_path, _ = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    stored = feature_cache.read_embeddings(save_path, SEQUENCES[:3])
    assert to_embed == SEQUENCES[3:]
    # Sequence 1 is in both files; the other file only adds what the first left out.
    assert stored[SEQUENCES[1]].tolist() == [1.0 if winner == "sqlite" else 2.0] * WIDTH
    assert stored[SEQUENCES[0]].tolist() == [1.0] * WIDTH
    assert stored[SEQUENCES[2]].tolist() == [2.0] * WIDTH


def test_an_old_cache_is_not_converted_when_its_pooling_order_cannot_be_known(tmp_path, capsys):
    # The file name lists pooling types sorted, so `var, mean` cannot be told from `mean, var` by it.
    args = arguments(tmp_path, embedding_pooling_types=["var", "mean"])
    old = write_torch_cache(
        tmp_path / args.cache_filename(MODEL, extension="pth"),
        {sequence: torch.zeros(2 * WIDTH) for sequence in SEQUENCES},
    )

    to_embed, save_path, rows = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES
    assert rows == {}
    assert not feature_cache.has_feature(save_path)
    assert old.is_file()
    assert "Not converting" in capsys.readouterr().out


def test_a_store_that_exists_is_not_changed_by_an_old_file_beside_it(tmp_path):
    args = arguments(tmp_path)
    build_store_for(args, MODEL, vectors(SEQUENCES[:1]))
    write_torch_cache(tmp_path / args.cache_filename(MODEL, extension="pth"), vectors(SEQUENCES[1:3]))

    to_embed, _, _ = Embedder(args, SEQUENCES)._read_embeddings_from_disk(MODEL)

    assert to_embed == SEQUENCES[1:]


class HubDownload:
    """A stand-in for `hf_hub_download` that serves one gzipped pickled dictionary, as the Hub dataset does."""

    def __init__(self, root, rows=None):
        self.requested = []
        self.path = root / "hub" / "download.pth.gz"
        self.path.parent.mkdir(parents=True)
        if rows is not None:
            pickled = root / "hub" / "download.pth"
            torch.save(rows, pickled)
            with gzip.open(self.path, "wb") as compressed:
                compressed.write(pickled.read_bytes())
            pickled.unlink()

    def __call__(self, repo_id, filename, repo_type):
        self.requested.append(filename)
        if not self.path.is_file():
            raise FileNotFoundError(filename)
        return str(self.path)


def test_a_download_is_converted_into_the_store_without_embedding(tmp_path, monkeypatch):
    published = vectors(SEQUENCES, dtype=torch.float16)
    hub = HubDownload(tmp_path, published)
    monkeypatch.setattr(embedder_module, "hf_hub_download", hub)
    root = tmp_path / "embeddings"
    args = arguments(root, download_embeddings=True)

    directory = Embedder(args, SEQUENCES)._download_embeddings(MODEL)

    assert hub.requested == [f"embeddings/{args.cache_filename(MODEL, extension='pth')}.gz"]
    assert Path(directory).resolve() == Path(args.feature_directory(MODEL)).resolve()
    stored = feature_cache.read_embeddings(directory, SEQUENCES)
    for sequence, row in published.items():
        assert torch.equal(stored[sequence], row.to(torch.float32))
    assert [entry.name for entry in root.iterdir()] == [Path(directory).name]


def test_a_download_only_adds_the_sequences_a_store_lacks(tmp_path, monkeypatch):
    hub = HubDownload(tmp_path, {SEQUENCES[0]: torch.full((WIDTH,), 1.0), SEQUENCES[1]: torch.full((WIDTH,), 1.0)})
    monkeypatch.setattr(embedder_module, "hf_hub_download", hub)
    args = arguments(tmp_path / "embeddings", download_embeddings=True)
    build_store_for(args, MODEL, {SEQUENCES[0]: torch.full((WIDTH,), 9.0)})

    directory = Embedder(args, SEQUENCES)._download_embeddings(MODEL)

    stored = feature_cache.read_embeddings(directory, SEQUENCES[:2])
    assert stored[SEQUENCES[0]].tolist() == [9.0] * WIDTH, "a row already embedded here is never replaced"
    assert stored[SEQUENCES[1]].tolist() == [1.0] * WIDTH


def test_a_download_is_not_converted_when_its_pooling_order_cannot_be_known(tmp_path, monkeypatch, capsys):
    hub = HubDownload(tmp_path, {sequence: torch.zeros(2 * WIDTH) for sequence in SEQUENCES})
    monkeypatch.setattr(embedder_module, "hf_hub_download", hub)
    args = arguments(tmp_path / "embeddings", embedding_pooling_types=["var", "mean"], download_embeddings=True)

    assert Embedder(args, SEQUENCES)._download_embeddings(MODEL) is None

    assert not feature_cache.has_feature(args.feature_directory(MODEL))
    assert "Not converting" in capsys.readouterr().out


def test_a_download_the_hub_lacks_changes_nothing(tmp_path, monkeypatch):
    hub = HubDownload(tmp_path)
    monkeypatch.setattr(embedder_module, "hf_hub_download", hub)
    args = arguments(tmp_path / "embeddings", download_embeddings=True)

    assert Embedder(args, SEQUENCES)._download_embeddings(MODEL) is None

    assert len(hub.requested) == 2  # the current name, then the published one at the default length
    assert not (tmp_path / "embeddings").exists()


@pytest.mark.parametrize("keep", ["save", "sql"])
def test_embedding_commits_its_rows_and_a_rerun_embeds_nothing(tmp_path, embedder_for, keep):
    make, model, loaded = embedder_for
    options = {"save_embeddings": True} if keep == "save" else {"sql": True}
    embedder = make(tmp_path, **options)

    rows = embedder(MODEL)

    directory = embedder.args.feature_directory(MODEL)
    assert committed(directory) == set(SEQUENCES)
    stored = feature_cache.read_embeddings(directory, SEQUENCES)
    for sequence in SEQUENCES:
        torch.testing.assert_close(stored[sequence], expected_mean(sequence), rtol=1e-5, atol=1e-4)
    if keep == "save":
        assert {sequence: raw(row) for sequence, row in rows.items()} == {
            sequence: raw(row) for sequence, row in stored.items()
        }
    else:
        assert rows == {}, "under sql the rows are streamed to the store and not held"

    calls = model.calls
    again = make(tmp_path, **options)(MODEL)

    assert model.calls == calls and loaded == [MODEL]
    if keep == "save":
        assert {sequence: raw(row) for sequence, row in again.items()} == {
            sequence: raw(row) for sequence, row in stored.items()
        }


def test_without_save_or_sql_rows_are_returned_and_nothing_is_written(tmp_path, embedder_for):
    make, _, _ = embedder_for

    rows = make(tmp_path / "embeddings")(MODEL)

    assert set(rows) == set(SEQUENCES)
    torch.testing.assert_close(rows[SEQUENCES[0]], expected_mean(SEQUENCES[0]), rtol=1e-5, atol=1e-4)
    assert not (tmp_path / "embeddings").exists() or list((tmp_path / "embeddings").iterdir()) == []


def test_a_run_that_stops_keeps_its_committed_windows_and_a_restart_embeds_the_rest(tmp_path, monkeypatch):
    # Windows of two rows, and a model that stops on its third batch of two.
    monkeypatch.setattr(embedder_module, "CacheWriter", functools.partial(feature_cache.CacheWriter, window_rows=2))
    model = Model(fail_on_call=3)
    monkeypatch.setattr(embedder_module, "get_base_model", lambda *a, **k: (model, Tokenizer()))
    args = dict(sql=True, padding="longest", max_length=64, embedding_batch_size=2)

    with pytest.raises(RuntimeError, match="stopped"):
        Embedder(arguments(tmp_path, **args), SEQUENCES)(MODEL)

    directory = arguments(tmp_path, **args).feature_directory(MODEL)
    assert len(committed(directory)) == 4  # the two full windows survive
    survivors = committed(directory)

    resumed = Model()
    monkeypatch.setattr(embedder_module, "get_base_model", lambda *a, **k: (resumed, Tokenizer()))
    Embedder(arguments(tmp_path, **args), SEQUENCES)(MODEL)

    assert resumed.calls == 1, "only the one sequence the store lacked was embedded"
    assert committed(directory) == set(SEQUENCES)
    assert survivors < committed(directory)


def test_matrix_rows_keep_every_token_including_the_special_ones(tmp_path, embedder_for):
    make, _, _ = embedder_for
    embedder = make(tmp_path, matrix_embed=True, sql=True)

    embedder(MODEL)

    directory = embedder.args.feature_directory(MODEL)
    assert FeatureStore.read_only(directory).spec.layout == RAGGED
    stored = feature_cache.read_embeddings(directory, SEQUENCES)
    for sequence in SEQUENCES:
        assert stored[sequence].shape == (len(sequence) + 2, WIDTH)
    first = stored[SEQUENCES[0]]  # (l, d)
    assert first[0, 0].item() == 1.0 and first[-1, 0].item() == 2.0  # start and end tokens


def test_pooling_more_than_one_reduction_widens_the_stored_rows(tmp_path, embedder_for):
    make, _, _ = embedder_for
    embedder = make(tmp_path, embedding_pooling_types=["mean", "var"], sql=True)

    embedder(MODEL)

    directory = embedder.args.feature_directory(MODEL)
    assert FeatureStore.read_only(directory).spec.width == 2 * WIDTH
    assert directory != make(tmp_path).args.feature_directory(MODEL), "each pooling set is its own store"
