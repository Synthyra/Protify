import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from einops import rearrange, repeat

from .attention_utils import (
    AttentionBackend,
    BlockMask,
    _repeat_kv,
    flex_attention_func,
    kernels_attention_func,
    resolve_attention_backend,
    sdpa_attention_func,
)


Linear = nn.Linear
LayerNorm = nn.LayerNorm


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    # x: (..., d_r), where d_r is even
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)  # (..., d_r / 2), (..., d_r / 2)
        return torch.cat((-x2, x1), dim=-1)  # (..., d_r)

    x1, x2 = x[..., ::2], x[..., 1::2]  # (..., d_r / 2), (..., d_r / 2)
    paired = torch.stack((-x2, x1), dim=-1)  # (..., d_r / 2, 2)
    return rearrange(
        paired,
        "... d two -> ... (d two)",
        two=2,
    )  # (..., d_r)


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    # x: (b, l, h, d_h); cos, sin: (l_c, d_r / 2), where l_c >= l is cache length
    rotary_dim = cos.shape[-1] * 2  # d_r
    assert rotary_dim <= x.shape[-1], "Rotary dimension cannot exceed head dimension."
    seq_len = x.size(1)  # l
    cos = repeat(cos[:seq_len], "s d -> 1 s 1 (2 d)")  # (1, l, 1, d_r)
    sin = repeat(sin[:seq_len], "s d -> 1 s 1 (2 d)")  # (1, l, 1, d_r)
    rotary_states = x[..., :rotary_dim]  # (b, l, h, d_r)
    rotated_states = (
        rotary_states * cos + rotate_half(rotary_states, interleaved) * sin
    )  # (b, l, h, d_r)
    unrotated_states = x[..., rotary_dim:]  # (b, l, h, d_h - d_r)
    return torch.cat(
        (rotated_states, unrotated_states),
        dim=-1,
    )  # (b, l, h, d_h)


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scaling_factor: float = 1.0,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim  # d_r
        self.base = float(base)
        self.interleaved = interleaved
        self.scaling_factor = scaling_factor
        self.max_seq_len = max_seq_len  # l_max
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )  # (d_r / 2,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        positions = (  # (l_max,)
            torch.arange(max_seq_len, device=device, dtype=torch.float32)
            / self.scaling_factor
        )
        freqs = torch.outer(positions, inv_freq)  # (l_max, d_r / 2)
        self.register_buffer("_cos_k", torch.cos(freqs), persistent=False)  # (l_max, d_r / 2)
        self.register_buffer("_sin_k", torch.sin(freqs), persistent=False)  # (l_max, d_r / 2)
        self._seq_len_cached = max_seq_len

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # query_states, key_states: (b, l, h, d_h)
        seq_len = query_states.shape[1]  # l
        assert seq_len <= self.max_seq_len, f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
        cos = self._cos_k[:seq_len]  # (l, d_r / 2)
        sin = self._sin_k[:seq_len]  # (l, d_r / 2)
        query_states = apply_rotary_emb_torch(query_states, cos, sin, self.interleaved)  # (b, l, h, d_h)
        key_states = apply_rotary_emb_torch(key_states, cos, sin, self.interleaved)  # (b, l, h, d_h)
        return query_states, key_states  # (b, l, h, d_h), (b, l, h, d_h)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        rotary: bool = True,
        attention_backend: str = "flex",
        use_bias: bool = False,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size  # d
        self.n_heads = n_heads  # h
        self.d_head = hidden_size // n_heads  # d_h
        assert self.d_head * self.n_heads == self.hidden_size, "hidden_size must be divisible by n_heads."
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.q_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.k_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.v_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.out_proj = Linear(hidden_size, hidden_size, bias=use_bias)
        self.rotary = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len) if rotary else None
        self.attention_backend = resolve_attention_backend(attention_backend)

    def set_attention_backend(self, attention_backend: str) -> None:
        self.attention_backend = resolve_attention_backend(attention_backend)

    def prepare_qkv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: (b, l, d)
        batch_size, seq_len, _ = hidden_states.shape  # b, l, d
        query_states = self.q_proj(hidden_states).view(  # (b, l, h, d_h)
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head,
        )
        key_states = self.k_proj(hidden_states).view(  # (b, l, h, d_h)
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head,
        )
        value_states = self.v_proj(hidden_states).view(  # (b, l, h, d_h)
            batch_size,
            seq_len,
            self.n_heads,
            self.d_head,
        )
        if self.rotary is not None:
            query_states, key_states = self.rotary(query_states, key_states)  # each (b, l, h, d_h)
        return query_states, key_states, value_states  # each (b, l, h, d_h)

    @torch.no_grad()
    def _compute_s_max(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        # query_states, key_states: (b, l, h, d_h)
        query_bhld = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = key_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = _repeat_kv(key_bhld, 1)  # (b, h, l, d_h)
        query_norm = torch.linalg.vector_norm(query_bhld, dim=-1)  # (b, h, l)
        key_norm = torch.linalg.vector_norm(key_bhld, dim=-1)  # (b, h, l)
        s_max_bound = (
            query_norm.max(dim=-1).values  # (b, h)
            * key_norm.max(dim=-1).values  # (b, h)
        ).max(dim=0).values * self.scale  # (h,)
        return [s_max_bound[head_idx] for head_idx in range(self.n_heads)]  # h tensors shaped ()

    def _manual_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
        output_s_max: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        # query_states, key_states, value_states: (b, l, h, d_h)
        # attention_mask_4d: (b, 1, 1, l) or (b, 1, l, l)
        query_bhld = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = key_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        value_bhld = value_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        attention_logits = (  # (b, h, l, l)
            torch.matmul(query_bhld, key_bhld.transpose(-2, -1)) * self.scale
        )
        if attention_mask_4d is not None:
            attention_logits = attention_logits.masked_fill(  # (b, h, l, l)
                attention_mask_4d.logical_not(),
                float("-inf"),
            )
        attention_weights = F.softmax(attention_logits, dim=-1)  # (b, h, l, l)
        attn_output = torch.matmul(attention_weights, value_bhld)  # (b, h, l, d_h)
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None  # h scalars or None
        return attn_output, attention_weights, s_max  # (b, h, l, d_h), (b, h, l, l), optional h scalars

    def _dispatch_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor],
        attention_mask_4d: Optional[torch.Tensor],
        flex_block_mask: Optional[BlockMask],
        output_attentions: bool,
        output_s_max: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor] | None]:
        # query_states, key_states, value_states: (b, l, h, d_h)
        # attention_mask_2d: (b, l); attention_mask_4d: (b, 1, 1, l) or (b, 1, l, l)
        if output_attentions:
            return self._manual_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_4d=attention_mask_4d,
                output_s_max=output_s_max,
            )

        query_bhld = query_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        key_bhld = key_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)
        value_bhld = value_states.transpose(1, 2).contiguous()  # (b, h, l, d_h)

        if self.attention_backend == AttentionBackend.KERNELS:
            attn_output = kernels_attention_func(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_2d=attention_mask_2d,
            ).transpose(1, 2).contiguous()  # (b, h, l, d_h)
        elif self.attention_backend == AttentionBackend.FLEX:
            attn_output = flex_attention_func(
                query_states=query_bhld,
                key_states=key_bhld,
                value_states=value_bhld,
                flex_block_mask=flex_block_mask,
            )  # (b, h, l, d_h)
        elif self.attention_backend == AttentionBackend.SDPA:
            attn_output = sdpa_attention_func(
                query_states=query_bhld,
                key_states=key_bhld,
                value_states=value_bhld,
                attention_mask_4d=attention_mask_4d,
            )  # (b, h, l, d_h)
        else:
            raise AssertionError(f"Unsupported attention backend: {self.attention_backend}.")

        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None  # h scalars or None
        return attn_output, None, s_max  # (b, h, l, d_h), None, optional h scalars

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor] | None]:
        # hidden_states: (b, l, d)
        query_states, key_states, value_states = self.prepare_qkv(hidden_states)  # each (b, l, h, d_h)
        attn_output, attention_weights, s_max = self._dispatch_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )  # (b, h, l, d_h), optional (b, h, l, l), optional h scalars
        attn_output = attn_output.transpose(1, 2).reshape(  # (b, l, d)
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.hidden_size,
        )
        output = self.out_proj(attn_output)  # (b, l, d)
        return output, attention_weights, s_max  # (b, l, d), optional (b, h, l, l), optional h scalars


class AttentionLogitsSequence(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, num_labels) -> (b, num_labels)
    """

    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = "dot") -> None:
        super().__init__()
        self.num_labels = num_labels  # c
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))  # (1, d, c)
        self.Wx = Linear(hidden_size, hidden_size, bias=False)
        self.sim_type = sim_type

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # emb: (b, l, c); attention_mask: (b, l, 1) in the current caller
        if attention_mask is None:
            return emb.mean(dim=1)  # (b, c)
        return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)  # (b, c)

    def dot_product(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d); p: (b, d, c)
        return torch.matmul(x, p)  # (b, l, c)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d); p: (b, d, c)
        x_exp = x.unsqueeze(-1)  # (b, l, d, 1)
        p_exp = p.unsqueeze(1)  # (b, 1, d, c)
        dist = torch.abs(torch.norm(x_exp - p_exp, p=2, dim=2))  # (b, l, c)
        return -dist  # (b, l, c)

    def cosine_similarity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (b, l, d); p: (b, d, c); attention_mask: (b, l, 1)
        x = x * attention_mask  # (b, l, d)
        x = F.normalize(x, p=2, dim=-1)  # (b, l, d)
        p = F.normalize(p, p=2, dim=1)  # (b, d, c)
        cos_sims = torch.matmul(x, p)  # (b, l, c)
        assert cos_sims.max().item() <= 1.0 and cos_sims.min().item() >= -1.0, (
            "Cosine similarity values should be between -1 and 1."
        )
        return cos_sims  # (b, l, c)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs
        # x: (b, l, d); attention_mask: (b, l)
        batch_size, seq_len, _ = x.size()  # b, l, d
        p = self.Wp.expand(batch_size, -1, -1)  # (b, d, c)
        x = self.Wx(x)  # (b, l, d)

        if attention_mask is None:
            attention_mask = torch.ones(  # (b, l)
                batch_size,
                seq_len,
                device=x.device,
                dtype=x.dtype,
            )

        pooled_attention_mask = attention_mask.unsqueeze(-1)  # (b, l, 1)

        if self.sim_type == "dot":
            y = self.dot_product(x, p)  # (b, l, c)
        elif self.sim_type == "euclidean":
            y = self.euclidean_distance(x, p)  # (b, l, c)
        elif self.sim_type == "cosine":
            y = self.cosine_similarity(x, p, pooled_attention_mask)  # (b, l, c)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")

        logits = self.mean_pooling(y, pooled_attention_mask)  # (b, c)
        return logits, y, x  # (b, c), (b, l, c), (b, l, d)


class AttentionLogitsToken(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, num_labels)
    """

    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = "dot") -> None:
        super().__init__()
        self.num_labels = num_labels  # c
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))  # (1, d, c)
        self.Wx = Linear(hidden_size, hidden_size, bias=False)
        self.sim_type = sim_type

    def dot_product(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d); p: (b, d, c)
        return torch.matmul(x, p)  # (b, l, c)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        # x: (b, l, d); p: (b, d, c). These shapes have no general
        # broadcast contract, so this legacy branch may raise RuntimeError.
        return torch.norm(x - p, p=2, dim=-1)

    def cosine_similarity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x: (b, l, d); p: (b, d, c); attention_mask: (b, l)
        if attention_mask is not None:
            expanded_mask = attention_mask.unsqueeze(-1)  # (b, l, 1)
            x = x * expanded_mask  # (b, l, d)

        x = F.normalize(x, p=2, dim=-1)  # (b, l, d)
        p = F.normalize(p, p=2, dim=1)  # (b, d, c)
        cos_sims = torch.matmul(x, p)  # (b, l, c)
        assert cos_sims.max().item() <= 1.0 and cos_sims.min().item() >= -1.0, (
            "Cosine similarity values should be between -1 and 1."
        )
        return cos_sims  # (b, l, c)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        del kwargs
        # x: (b, l, d); attention_mask: (b, l)
        batch_size, _, _ = x.size()  # b, l, d
        p = self.Wp.expand(batch_size, -1, -1)  # (b, d, c)
        x = self.Wx(x)  # (b, l, d)
        if self.sim_type == "dot":
            logits = self.dot_product(x, p)  # (b, l, c)
        elif self.sim_type == "euclidean":
            # No general output shape exists for the legacy unsupported broadcast.
            logits = self.euclidean_distance(x, p)
        elif self.sim_type == "cosine":
            logits = self.cosine_similarity(x, p, attention_mask)  # (b, l, c)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")
        return logits  # (b, l, c) for dot/cosine; legacy-dependent for euclidean
