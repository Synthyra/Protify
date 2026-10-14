"""Protify's E1 batch preparer is FastPLMs' preparer on Protify's tokenizer.

Until 2026-10-06 Protify kept its own copy of the preparer. The batches below were recorded from that copy, so
FastPLMs' preparer must give the same token ids, labels, sequence and position ids, and contexts. Three
sequences make one batch: a lone sequence, a three-sequence instance whose first two sequences are context,
and a two-sequence instance that holds X residues.

Shapes: `b` sequences in a batch, `l` tokens in the longest row, `t` tokens of one sequence.
"""

import json
import pytest
import torch

from pathlib import Path

from fastplms.models.e1.modeling_e1 import E1BatchPreparer as FastE1BatchPreparer
from fastplms.models.e1.modeling_e1 import get_tokenizer as get_fastplms_tokenizer
from protify.base_models import e1_utils


BATCH = ["MKT", "ACD,EFGH,IK", "XACXD,GGXG"]
WIDTH = 21  # l, the tokens of the three-sequence instance
PAD_TOKEN_ID = 0
NO_POSITION = -1  # sequence and position ids of padding
MASK_TOKEN_ID = 5  # "?"
X_TOKEN_ID = 31
BOUNDARY_TOKEN_IDS = [1, 2, 6, 7, 0]  # <bos>, <eos>, 1, 2, <pad>
CONTEXT = "<bos>1ACD2<eos><bos>1EFGH2<eos>"
# Rows of the default batch before padding, one list per sequence of BATCH.
DEFAULT_ROWS = {
    "input_ids": [
        [1, 6, 20, 18, 27, 7, 2],
        [1, 6, 8, 10, 11, 7, 2, 1, 6, 12, 13, 14, 15, 7, 2, 1, 6, 16, 18, 7, 2],
        [1, 6, 31, 8, 10, 31, 11, 7, 2, 1, 6, 14, 14, 31, 14, 7, 2],
    ],
    "labels": [
        [1, 6, 20, 18, 27, 7, 2],
        [PAD_TOKEN_ID] * 15 + [1, 6, 16, 18, 7, 2],
        [PAD_TOKEN_ID] * 9 + [1, 6, 14, 14, 31, 14, 7, 2],
    ],
    "sequence_ids": [[0] * 7, [0] * 7 + [1] * 8 + [2] * 6, [0] * 9 + [1] * 8],
    "within_seq_position_ids": [
        list(range(7)),
        list(range(7)) + list(range(8)) + list(range(6)),
        list(range(9)) + list(range(8)),
    ],
    "global_position_ids": [list(range(7)), list(range(21)), list(range(17))],
}
# The same rows with X residues removed: the second instance's positions keep the gaps the X residues left.
REMOVE_X_ROWS = {
    "input_ids": [
        DEFAULT_ROWS["input_ids"][0],
        DEFAULT_ROWS["input_ids"][1],
        [1, 6, 8, 10, 11, 7, 2, 1, 6, 14, 14, 14, 7, 2],
    ],
    "labels": [
        DEFAULT_ROWS["labels"][0],
        DEFAULT_ROWS["labels"][1],
        [PAD_TOKEN_ID] * 7 + [1, 6, 14, 14, 14, 7, 2],
    ],
    "sequence_ids": [DEFAULT_ROWS["sequence_ids"][0], DEFAULT_ROWS["sequence_ids"][1], [0] * 7 + [1] * 7],
    "within_seq_position_ids": [
        DEFAULT_ROWS["within_seq_position_ids"][0],
        DEFAULT_ROWS["within_seq_position_ids"][1],
        [0, 1, 3, 4, 6, 7, 8, 0, 1, 2, 3, 5, 6, 7],
    ],
    "global_position_ids": [
        DEFAULT_ROWS["global_position_ids"][0],
        DEFAULT_ROWS["global_position_ids"][1],
        [0, 1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16],
    ],
}
FILL_VALUES = {
    "input_ids": PAD_TOKEN_ID,
    "labels": PAD_TOKEN_ID,
    "sequence_ids": NO_POSITION,
    "within_seq_position_ids": NO_POSITION,
    "global_position_ids": NO_POSITION,
}


def padded(rows: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    return {key: [row + [FILL_VALUES[key]] * (WIDTH - len(row)) for row in values] for key, values in rows.items()}


def tensor_fields(batch: dict) -> dict[str, list[list[int]]]:
    return {key: batch[key].tolist() for key in FILL_VALUES}  # each (b, l) as nested lists


def test_the_preparer_is_fastplms_with_nothing_of_its_own_but_the_constructor():
    assert e1_utils.E1BatchPreparer.__mro__[1] is FastE1BatchPreparer
    assert [name for name in vars(e1_utils.E1BatchPreparer) if not name.startswith("__")] == []
    assert e1_utils.DataPrepConfig.__module__.startswith("fastplms.")
    assert e1_utils.get_context.__module__.startswith("fastplms.")
    assert e1_utils.PAD_TOKEN_ID == PAD_TOKEN_ID


def test_a_batch_has_the_ids_positions_labels_and_contexts_the_old_copy_gave():
    batch = e1_utils.E1BatchPreparer().get_batch_kwargs(BATCH)

    assert tensor_fields(batch) == padded(DEFAULT_ROWS)
    assert all(batch[key].shape == (3, WIDTH) and batch[key].dtype == torch.long for key in FILL_VALUES)
    assert batch["context"] == ["", CONTEXT, "<bos>1XACXD2<eos>"]
    assert batch["context_len"] == [0, 15, 9]


def test_removing_x_residues_keeps_the_positions_they_left_behind():
    preparer = e1_utils.E1BatchPreparer(data_prep_config=e1_utils.DataPrepConfig(remove_X_tokens=True))

    batch = preparer.get_batch_kwargs(BATCH)

    assert tensor_fields(batch) == padded(REMOVE_X_ROWS)
    assert batch["context"] == ["", CONTEXT, "<bos>1ACD2<eos>"]
    assert batch["context_len"] == [0, 15, 7]
    assert not (batch["input_ids"] == X_TOKEN_ID).any()  # (b, l)


def test_preserved_context_labels_repeat_the_input_ids():
    batch = e1_utils.E1BatchPreparer(preserve_context_labels=True).get_batch_kwargs(BATCH)

    assert torch.equal(batch["labels"], batch["input_ids"])
    assert batch["context_len"] == [0, 15, 9]


def test_a_single_sequence_is_bracketed_and_keeps_its_mask_and_x_symbols():
    single = e1_utils.E1BatchPreparer().prepare_singleseq("ACXD?")
    without_x = e1_utils.E1BatchPreparer(
        data_prep_config=e1_utils.DataPrepConfig(remove_X_tokens=True)
    ).prepare_singleseq("ACXD?")

    assert single["input_ids"].tolist() == [1, 6, 8, 10, X_TOKEN_ID, 11, MASK_TOKEN_ID, 7, 2]  # (t,)
    assert torch.equal(single["labels"], single["input_ids"])
    assert single["position_ids"].tolist() == list(range(9))  # (t,)
    assert without_x["input_ids"].tolist() == [1, 6, 8, 10, 11, MASK_TOKEN_ID, 7, 2]  # (t,)
    assert without_x["position_ids"].tolist() == [0, 1, 2, 3, 5, 6, 7, 8]  # (t,)


def test_the_special_token_ids_and_masks_are_the_ones_the_old_copy_found():
    preparer = e1_utils.E1BatchPreparer()
    tokens = torch.arange(8).reshape(1, 8)  # (1, 8)

    assert preparer.pad_token_id == PAD_TOKEN_ID
    assert preparer.mask_token_id == MASK_TOKEN_ID
    assert preparer.X_token_id == X_TOKEN_ID
    assert preparer.boundary_token_ids.tolist() == BOUNDARY_TOKEN_IDS
    assert preparer.get_boundary_token_mask(tokens).tolist() == [[True, True, True, False, False, False, True, True]]
    assert preparer.get_mask_positions_mask(torch.tensor([[MASK_TOKEN_ID, 3, MASK_TOKEN_ID]])).tolist() == [
        [True, False, True]
    ]


@pytest.mark.parametrize("sequence", ["acd", "AC1", ""])
def test_a_sequence_outside_the_alphabet_is_rejected(sequence):
    with pytest.raises(ValueError, match="Invalid sequence"):
        e1_utils.E1BatchPreparer().prepare_singleseq(sequence)


def test_a_sequence_that_is_not_text_is_a_type_error_where_the_old_copy_asserted():
    with pytest.raises(TypeError, match="Sequence must be a string"):
        e1_utils.E1BatchPreparer().prepare_singleseq(7)


def test_the_data_prep_limits_bound_the_instance_and_each_sequence():
    limited = e1_utils.E1BatchPreparer(
        data_prep_config=e1_utils.DataPrepConfig(max_num_sequences=2, max_num_positions_within_seq=3)
    )

    assert limited.prepare_multiseq("ACD,EFG")["context_len"] == 7
    with pytest.raises(ValueError, match="exceeds max number of sequences 2"):
        limited.prepare_multiseq("ACD,EFG,HIK")
    with pytest.raises(ValueError, match="exceeds max length 3"):
        limited.prepare_singleseq("ACDE")


def test_the_boundary_token_ids_sit_on_the_requested_device_and_by_default_on_the_current_one():
    on_cpu = e1_utils.E1BatchPreparer(device=torch.device("cpu"))
    by_default = e1_utils.E1BatchPreparer()

    assert on_cpu.boundary_token_ids.device == torch.device("cpu")
    assert by_default.boundary_token_ids.device == e1_utils.get_device()
    assert by_default.boundary_token_ids.dtype == torch.long


def test_the_bundled_tokenizer_holds_the_vocabulary_and_padding_of_fastplms():
    bundled = e1_utils.get_tokenizer()
    fastplms = get_fastplms_tokenizer()

    assert bundled.get_vocab() == fastplms.get_vocab()
    assert len(bundled.get_vocab()) == 34
    assert bundled.padding == fastplms.padding


def test_a_registered_path_supplies_the_tokenizer_when_the_bundled_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(e1_utils, "_E1_TOKENIZER_FILENAME", "missing_tokenizer.json")
    monkeypatch.setattr(e1_utils, "_extra_tokenizer_paths", [])
    registered = tmp_path / "e1_tokenizer.json"
    get_fastplms_tokenizer().save(str(registered))

    e1_utils.register_tokenizer_path(str(registered))
    e1_utils.register_tokenizer_path(str(registered))

    assert e1_utils._extra_tokenizer_paths == [str(registered)]
    assert e1_utils.get_tokenizer().get_vocab() == get_fastplms_tokenizer().get_vocab()


def test_a_registered_tokenizer_that_pads_with_another_id_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(e1_utils, "_E1_TOKENIZER_FILENAME", "missing_tokenizer.json")
    monkeypatch.setattr(e1_utils, "_extra_tokenizer_paths", [])
    original = tmp_path / "original.json"
    get_fastplms_tokenizer().save(str(original))
    document = json.loads(Path(original).read_text(encoding="utf-8"))
    document["padding"]["pad_id"] = 1
    shifted = tmp_path / "shifted.json"
    shifted.write_text(json.dumps(document), encoding="utf-8")
    e1_utils.register_tokenizer_path(str(shifted))

    with pytest.raises(AssertionError, match="Padding token id must be 0, but got 1"):
        e1_utils.get_tokenizer()
