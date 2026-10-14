"""How `DataMixin` finds and reads the feature store of a model's cached embeddings.

`--sql` keeps its name and now means streaming rows from the store instead of holding them in memory,
so `get_embedding_dim_sql` takes the store directory, and `_embedding_dict` is the in-memory read the
non-streaming path uses. A missing store says how to make one, and names an old cache file to convert
when one sits where the store should be.
"""

import pytest
import torch

from protify.data.data_mixin import DataMixin
from protify.embedder import EmbeddingArguments, get_embedding_filename
from protify.testing_suite.embedding_store_builder import build_store_for


MODEL = "test-model"
WIDTH = 3
ROWS = {
    "AAAA": torch.tensor([1.0, 2.0, 3.0]),
    "CCCC": torch.tensor([4.0, 5.0, 6.0]),
}


@pytest.fixture
def mixin(tmp_path):
    embedding_args = EmbeddingArguments(embedding_save_dir=str(tmp_path), embedding_pooling_types=["mean"])
    build_store_for(embedding_args, MODEL, ROWS)
    data_mixin = DataMixin()
    data_mixin._sql = True
    data_mixin._full = False
    data_mixin.embedding_args = embedding_args
    return data_mixin


def test_the_store_directory_is_the_one_the_embedder_arguments_name(mixin):
    assert mixin._embedding_store(MODEL) == mixin.embedding_args.feature_directory(MODEL, matrix_embed=False)


def test_the_old_file_name_is_still_the_one_the_converter_and_the_hub_use(mixin):
    assert mixin._embedding_cache_filename(MODEL, "db") == get_embedding_filename(
        MODEL, False, ["mean"], extension="db", max_length=2048
    )


def test_embedding_dict_reads_each_distinct_row_once_into_memory(mixin):
    rows = mixin._embedding_dict(MODEL, ["CCCC", "AAAA", "CCCC"])

    assert list(rows) == ["CCCC", "AAAA"]
    for sequence, row in ROWS.items():
        assert torch.equal(rows[sequence], row)
    rows["AAAA"].zero_()  # the caller's own copy
    assert torch.equal(mixin._embedding_dict(MODEL, ["AAAA"])["AAAA"], ROWS["AAAA"])


def test_embedding_dict_refuses_a_sequence_the_store_lacks(mixin):
    with pytest.raises(KeyError, match="no row"):
        mixin._embedding_dict(MODEL, ["AAAA", "GGGG"])


def test_a_missing_store_says_to_embed_first(mixin):
    with pytest.raises(FileNotFoundError, match="No embedding store.*Embed the sequences first"):
        mixin._embedding_dict("absent-model", ["AAAA"])


def test_a_missing_store_names_the_old_cache_that_should_be_converted(mixin, tmp_path):
    old = tmp_path / mixin._embedding_cache_filename("absent-model", "db")
    old.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match="An old cache is there.*python -m protify.legacy_cache"):
        mixin._embedding_dict("absent-model", ["AAAA"])
    assert mixin._legacy_embedding_files("absent-model") == [str(old)]


def test_the_streamed_width_is_the_stores_and_the_probe_sequence_must_be_in_it(mixin):
    store = mixin._embedding_store(MODEL)

    assert mixin.get_embedding_dim_sql(store, "AAAA", tokenizer=None) == WIDTH
    with pytest.raises(AssertionError, match="Test sequence not found"):
        mixin.get_embedding_dim_sql(store, "GGGG", tokenizer=None)


def test_the_in_memory_width_is_the_last_axis(mixin):
    rows = {"AAAA": torch.zeros(5, WIDTH)}  # (l, d)

    assert mixin.get_embedding_dim_pth(rows, "AAAA", tokenizer=None) == WIDTH
