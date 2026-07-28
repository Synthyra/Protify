import torch
import torch.nn.functional as F
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU feed-forward projection with optional packed gate weights."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
        *,
        _pack_weights: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self._pack_weights = _pack_weights
        self.hidden_features = hidden_features  # d_ff
        self.in_features = in_features  # d_in
        self.out_features = out_features  # d_out

        if _pack_weights:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
            self.w1 = None
            self.w2 = None
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d_in)
        if self._pack_weights and self.w12 is not None:
            x12 = self.w12(x)  # (..., 2 * d_ff)
            x1, x2 = x12.chunk(2, dim=-1)  # each (..., d_ff)
        else:
            assert self.w1 is not None and self.w2 is not None, "Weights w1 and w2 must be initialized."
            x1 = self.w1(x)  # (..., d_ff)
            x2 = self.w2(x)  # (..., d_ff)
        hidden = F.silu(x1) * x2  # (..., d_ff)
        output = self.w3(hidden)  # (..., d_out)
        return output  # (..., d_out)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # (d,)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        mean_square = x.pow(2).mean(-1, keepdim=True)  # (..., 1)
        inverse_rms = torch.rsqrt(mean_square + self.eps)  # (..., 1)
        normalized = x * inverse_rms  # (..., d)
        return normalized  # (..., d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        # Normalize in fp32 to avoid mixed-precision instability in AMPLIFY.
        float_x = x.float()  # (..., d)
        normalized = self._norm(float_x).type_as(x)  # (..., d)
        output = normalized * self.weight  # (..., d)
        return output  # (..., d)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex rotary frequencies for ``end`` positions."""
    # dim is d_h and d_r = floor(d_h / 2); end is l.
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # (d_r,)
    positions = torch.arange(end, device=freqs.device)  # (l,)
    freqs = torch.outer(positions, freqs).float()  # (l, d_r)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (l, d_r), complex64
    return freqs_cis  # (l, d_r)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # freqs_cis: (l, d_r); x has rank k with x.shape[1] = l and x.shape[-1] = d_r.
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # (l, d_r) for k=2; otherwise (1, l, ..., 1, d_r), with k - 2 singleton axes.
    broadcast_freqs = freqs_cis.view(*shape)
    return broadcast_freqs


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    # xq, xk: (b, l, h, d_h); freqs_cis: (l, d_r), where d_r = d_h / 2.
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # (b, l, h, d_r)
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # (b, l, h, d_r)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)  # (1, l, 1, d_r)
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)  # (b, l, h, d_h)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)  # (b, l, h, d_h)
    xq_out = xq_out.type_as(xq)  # (b, l, h, d_h)
    xk_out = xk_out.type_as(xk)  # (b, l, h, d_h)
    return xq_out, xk_out  # each (b, l, h, d_h)
