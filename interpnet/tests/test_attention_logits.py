"""The token head's Euclidean readout is each residue's negative distance to each label parameter.

The sequence head computes the same geometry, so a residue closer to a label scores higher for it.
"""

import pytest
import torch

from interpnet.attention_logits import AttentionLogitsSequence, AttentionLogitsToken


@pytest.mark.parametrize("num_labels", [1, 4])
def test_token_euclidean_matches_sequence_geometry(num_labels: int) -> None:
    batch_size, seq_len, hidden_size = 2, 5, 3
    x = torch.randn(batch_size, seq_len, hidden_size)  # (b, l, d)
    p = torch.randn(batch_size, hidden_size, num_labels)  # (b, d, c)

    token_head = AttentionLogitsToken(hidden_size, num_labels, sim_type="euclidean")
    sequence_head = AttentionLogitsSequence(hidden_size, num_labels, sim_type="euclidean")

    token_logits = token_head.euclidean_distance(x, p)  # (b, l, c)
    sequence_logits = sequence_head.euclidean_distance(x, p)  # (b, l, c)

    assert token_logits.shape == (batch_size, seq_len, num_labels)
    torch.testing.assert_close(token_logits, sequence_logits)


def test_token_euclidean_forward_preserves_label_axis_and_similarity_order() -> None:
    head = AttentionLogitsToken(hidden_size=2, num_labels=2, sim_type="euclidean")
    x = torch.tensor([[[0.0, 0.0], [3.0, 4.0]]])  # (b=1, l=2, d=2)
    prototypes = torch.tensor([[[0.0, 0.0], [0.0, 4.0]]])  # (b=1, d=2, c=2)

    with torch.no_grad():
        head.Wx.weight.copy_(torch.eye(2))
        head.Wp.copy_(prototypes)

    logits = head(x)  # (b=1, l=2, c=2)
    expected = torch.tensor([[[0.0, -4.0], [-5.0, -3.0]]])  # (b=1, l=2, c=2)

    assert logits.shape == (1, 2, 2)
    torch.testing.assert_close(logits, expected)
    assert logits[0, 0, 0] > logits[0, 0, 1]
