import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Optional, Tuple, Union

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
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    x1, x2 = x[..., ::2], x[..., 1::2]
    return rearrange(
        torch.stack((-x2, x1), dim=-1),
        "... d two -> ... (d two)",
        two=2,
    )


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1] * 2
    assert rotary_dim <= x.shape[-1], "Rotary dimension cannot exceed head dimension."
    seq_len = x.size(1)
    cos = repeat(cos[:seq_len], "s d -> 1 s 1 (2 d)")
    sin = repeat(sin[:seq_len], "s d -> 1 s 1 (2 d)")
    return torch.cat(
        (
            x[..., :rotary_dim] * cos + rotate_half(x[..., :rotary_dim], interleaved) * sin,
            x[..., rotary_dim:],
        ),
        dim=-1,
    )


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scaling_factor: float = 1.0,
        max_seq_len: int = 2048,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scaling_factor = scaling_factor
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32) / self.scaling_factor
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("_cos_k", torch.cos(freqs), persistent=False)
        self.register_buffer("_sin_k", torch.sin(freqs), persistent=False)
        self._seq_len_cached = max_seq_len

    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = query_states.shape[1]
        assert seq_len <= self.max_seq_len, f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}"
        cos = self._cos_k[:seq_len]
        sin = self._sin_k[:seq_len]
        return (
            apply_rotary_emb_torch(query_states, cos, sin, self.interleaved),
            apply_rotary_emb_torch(key_states, cos, sin, self.interleaved),
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        rotary: bool = True,
        attention_backend: str = "flex",
        use_bias: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads
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
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head)
        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.n_heads, self.d_head)
        if self.rotary is not None:
            query_states, key_states = self.rotary(query_states, key_states)
        return query_states, key_states, value_states

    @torch.no_grad()
    def _compute_s_max(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        query_bhld = query_states.transpose(1, 2).contiguous()
        key_bhld = key_states.transpose(1, 2).contiguous()
        key_bhld = _repeat_kv(key_bhld, 1)
        query_norm = torch.linalg.vector_norm(query_bhld, dim=-1)
        key_norm = torch.linalg.vector_norm(key_bhld, dim=-1)
        s_max_bound = (
            query_norm.max(dim=-1).values
            * key_norm.max(dim=-1).values
        ).max(dim=0).values * self.scale
        return [s_max_bound[head_idx] for head_idx in range(self.n_heads)]

    def _manual_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask_4d: Optional[torch.Tensor],
        output_s_max: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        query_bhld = query_states.transpose(1, 2).contiguous()
        key_bhld = key_states.transpose(1, 2).contiguous()
        value_bhld = value_states.transpose(1, 2).contiguous()
        attention_logits = torch.matmul(query_bhld, key_bhld.transpose(-2, -1)) * self.scale
        if attention_mask_4d is not None:
            attention_logits = attention_logits.masked_fill(attention_mask_4d.logical_not(), float("-inf"))
        attention_weights = F.softmax(attention_logits, dim=-1)
        attn_output = torch.matmul(attention_weights, value_bhld)
        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, attention_weights, s_max

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
        if output_attentions:
            return self._manual_attention(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_4d=attention_mask_4d,
                output_s_max=output_s_max,
            )

        query_bhld = query_states.transpose(1, 2).contiguous()
        key_bhld = key_states.transpose(1, 2).contiguous()
        value_bhld = value_states.transpose(1, 2).contiguous()

        if self.attention_backend == AttentionBackend.KERNELS:
            attn_output = kernels_attention_func(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask_2d=attention_mask_2d,
            ).transpose(1, 2).contiguous()
        elif self.attention_backend == AttentionBackend.FLEX:
            attn_output = flex_attention_func(
                query_states=query_bhld,
                key_states=key_bhld,
                value_states=value_bhld,
                flex_block_mask=flex_block_mask,
            )
        elif self.attention_backend == AttentionBackend.SDPA:
            attn_output = sdpa_attention_func(
                query_states=query_bhld,
                key_states=key_bhld,
                value_states=value_bhld,
                attention_mask_4d=attention_mask_4d,
            )
        else:
            raise AssertionError(f"Unsupported attention backend: {self.attention_backend}.")

        s_max = self._compute_s_max(query_states, key_states) if output_s_max else None
        return attn_output, None, s_max

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        attention_mask_4d: Optional[torch.Tensor] = None,
        flex_block_mask: Optional[BlockMask] = None,
        output_attentions: bool = False,
        output_s_max: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], list[torch.Tensor] | None]:
        query_states, key_states, value_states = self.prepare_qkv(hidden_states)
        attn_output, attention_weights, s_max = self._dispatch_attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask_2d=attention_mask_2d,
            attention_mask_4d=attention_mask_4d,
            flex_block_mask=flex_block_mask,
            output_attentions=output_attentions,
            output_s_max=output_s_max,
        )
        attn_output = attn_output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.hidden_size,
        )
        return self.out_proj(attn_output), attention_weights, s_max


class AttentionLogitsSequence(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, num_labels) -> (b, num_labels)
    """

    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = "dot"):
        super().__init__()
        self.num_labels = num_labels
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))
        self.Wx = Linear(hidden_size, hidden_size, bias=False)
        self.sim_type = sim_type

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return emb.mean(dim=1)
        return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

    def dot_product(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, p)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x_exp = x.unsqueeze(-1)
        p_exp = p.unsqueeze(1)
        dist = torch.abs(torch.norm(x_exp - p_exp, p=2, dim=2))
        return -dist

    def cosine_similarity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x * attention_mask
        x = F.normalize(x, p=2, dim=-1)
        p = F.normalize(p, p=2, dim=1)
        cos_sims = torch.matmul(x, p)
        assert cos_sims.max().item() <= 1.0 and cos_sims.min().item() >= -1.0, (
            "Cosine similarity values should be between -1 and 1."
        )
        return cos_sims

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del kwargs
        batch_size, seq_len, _ = x.size()
        p = self.Wp.expand(batch_size, -1, -1)
        x = self.Wx(x)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device, dtype=x.dtype)

        pooled_attention_mask = attention_mask.unsqueeze(-1)

        if self.sim_type == "dot":
            y = self.dot_product(x, p)
        elif self.sim_type == "euclidean":
            y = self.euclidean_distance(x, p)
        elif self.sim_type == "cosine":
            y = self.cosine_similarity(x, p, pooled_attention_mask)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")

        logits = self.mean_pooling(y, pooled_attention_mask)
        return logits, y, x


class AttentionLogitsToken(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, num_labels)
    """

    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = "dot"):
        super().__init__()
        self.num_labels = num_labels
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))
        self.Wx = Linear(hidden_size, hidden_size, bias=False)
        self.sim_type = sim_type

    def dot_product(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, p)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.norm(x - p, p=2, dim=-1)

    def cosine_similarity(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        x = F.normalize(x, p=2, dim=-1)
        p = F.normalize(p, p=2, dim=1)
        cos_sims = torch.matmul(x, p)
        assert cos_sims.max().item() <= 1.0 and cos_sims.min().item() >= -1.0, (
            "Cosine similarity values should be between -1 and 1."
        )
        return cos_sims

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, _, _ = x.size()
        p = self.Wp.expand(batch_size, -1, -1)
        x = self.Wx(x)
        if self.sim_type == "dot":
            logits = self.dot_product(x, p)
        elif self.sim_type == "euclidean":
            logits = self.euclidean_distance(x, p)
        elif self.sim_type == "cosine":
            logits = self.cosine_similarity(x, p, attention_mask)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")
        return logits
