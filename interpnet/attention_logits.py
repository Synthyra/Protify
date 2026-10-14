"""Token-parameter attention readouts that turn residue states into InterpNet logits.

Each label has a learned parameter vector, and each residue's similarity to it is that residue's
logit for the label. A sequence probe returns these per-residue logits as `sims`, which callers
such as the property oracles turn into per-residue attributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any


Linear = nn.Linear


class AttentionLogitsSequence(nn.Module):
    """Token-parameter attention for sequence tasks: (b, l, d) -> (b, l, c) -> (b, c).

    The sequence logits average the per-residue similarities over the residues the mask keeps.
    """

    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = "dot") -> None:
        super().__init__()
        self.num_labels = num_labels
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))  # (1, d, c)
        self.Wx = Linear(hidden_size, hidden_size, bias=False)
        self.sim_type = sim_type

    def mean_pooling(
        self,
        token_logits: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Average per-residue logits over the residues the mask keeps, or over all of them."""
        # token_logits: (b, l, c); attention_mask: (b, l, 1) or None
        if attention_mask is None:
            return token_logits.mean(dim=1)  # (b, c)

        return (token_logits * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, c)

    def dot_product(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return each residue's dot product with each label parameter."""
        # x: (b, l, d); p: (b, d, c)
        return torch.matmul(x, p)  # (b, l, c)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return each residue's negative Euclidean distance to each label parameter."""
        # x: (b, l, d); p: (b, d, c)
        x_expanded = x.unsqueeze(-1)  # (b, l, d, 1)
        p_expanded = p.unsqueeze(1)  # (b, 1, d, c)
        distance = torch.norm(x_expanded - p_expanded, p=2, dim=2)  # (b, l, c)
        return -distance  # (b, l, c)

    def cosine_similarity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return each kept residue's cosine similarity to each label parameter."""
        # x: (b, l, d); p: (b, d, c); attention_mask: (b, l, 1)
        x = x * attention_mask  # (b, l, d)
        x = F.normalize(x, p=2, dim=-1)  # (b, l, d)
        p = F.normalize(p, p=2, dim=1)  # (b, d, c)
        similarities = torch.matmul(x, p)  # (b, l, c)
        assert similarities.max().item() <= 1.0 and similarities.min().item() >= -1.0, (
            "Cosine similarity values should be between -1 and 1."
        )
        return similarities  # (b, l, c)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sequence logits, the per-residue logits, and the projected residue states."""
        # x: (b, l, d); attention_mask: (b, l) or None
        del kwargs
        batch_size, seq_len, _ = x.size()  # b, l, d
        parameters = self.Wp.expand(batch_size, -1, -1)  # (b, d, c)
        x = self.Wx(x)  # (b, l, d)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)  # (b, l)

        pooled_attention_mask = attention_mask.unsqueeze(-1)  # (b, l, 1)

        if self.sim_type == "dot":
            token_logits = self.dot_product(x, parameters)  # (b, l, c)
        elif self.sim_type == "euclidean":
            token_logits = self.euclidean_distance(x, parameters)  # (b, l, c)
        elif self.sim_type == "cosine":
            token_logits = self.cosine_similarity(x, parameters, pooled_attention_mask)  # (b, l, c)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")

        logits = self.mean_pooling(token_logits, pooled_attention_mask)  # (b, c)
        return logits, token_logits, x  # (b, c), (b, l, c), (b, l, d)


class AttentionLogitsToken(nn.Module):
    """Token-parameter attention for token tasks: (b, l, d) -> (b, l, c)."""

    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = "dot") -> None:
        super().__init__()
        self.num_labels = num_labels
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))  # (1, d, c)
        self.Wx = Linear(hidden_size, hidden_size, bias=False)
        self.sim_type = sim_type

    def dot_product(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return each residue's dot product with each label parameter."""
        # x: (b, l, d); p: (b, d, c)
        return torch.matmul(x, p)  # (b, l, c)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Return each residue's negative Euclidean distance to each label parameter."""
        # x: (b, l, d); p: (b, d, c)
        x_expanded = x.unsqueeze(-1)  # (b, l, d, 1)
        p_expanded = p.unsqueeze(1)  # (b, 1, d, c)
        distances = torch.linalg.vector_norm(x_expanded - p_expanded, ord=2, dim=2)  # (b, l, c)
        return -distances  # (b, l, c)

    def cosine_similarity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return each residue's cosine similarity to each label parameter, zeroing padding."""
        # x: (b, l, d); p: (b, d, c); attention_mask: (b, l) or None
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)  # (b, l, d)

        x = F.normalize(x, p=2, dim=-1)  # (b, l, d)
        p = F.normalize(p, p=2, dim=1)  # (b, d, c)
        similarities = torch.matmul(x, p)  # (b, l, c)
        assert similarities.max().item() <= 1.0 and similarities.min().item() >= -1.0, (
            "Cosine similarity values should be between -1 and 1."
        )
        return similarities  # (b, l, c)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Return the per-residue logits of the projected residue states."""
        # x: (b, l, d); attention_mask: (b, l) or None
        del kwargs
        batch_size, _, _ = x.size()  # b, l, d
        parameters = self.Wp.expand(batch_size, -1, -1)  # (b, d, c)
        x = self.Wx(x)  # (b, l, d)
        if self.sim_type == "dot":
            logits = self.dot_product(x, parameters)  # (b, l, c)
        elif self.sim_type == "euclidean":
            logits = self.euclidean_distance(x, parameters)  # (b, l, c)
        elif self.sim_type == "cosine":
            logits = self.cosine_similarity(x, parameters, attention_mask)  # (b, l, c)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")

        return logits  # (b, l, c)
