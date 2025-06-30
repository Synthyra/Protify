import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from einops import rearrange, repeat
from typing import Optional, Tuple, Union


Linear = partial(nn.Linear, bias=False)
LayerNorm = partial(nn.LayerNorm, bias=False)


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False, _inplace=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, "s d -> s 1 (2 d)")
    sin = repeat(sin, "s d -> s 1 (2 d)")
    return torch.cat(
        [
            x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        arange = torch.arange(0, self.dim, 2, device=self.device, dtype=torch.float32)
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            freqs = torch.outer(t, inv_freq)

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        """
        self._update_cos_sin_cache(q.shape[1], device=q.device, dtype=q.dtype)
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached,
                    self._sin_cached,
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            assert False


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, rotary: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            LayerNorm(hidden_size), Linear(hidden_size, hidden_size * 3)
        )
        self.out_proj = Linear(hidden_size, hidden_size)
        self.q_ln = LayerNorm(hidden_size, bias=False)
        self.k_ln = LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        qkv = self.layernorm_qkv(x) # (bs, seq_len, d_model * 3)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (bs, seq_len, hidden_size)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask) # (bs, n_heads, seq_len, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


class PAttention(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, n_tokens) ->  (b, L, d)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int,
            dropout: float = 0.2,
    ):
        super(PAttention, self).__init__()
        self.n_tokens = n_tokens
        self.Wq = Linear(hidden_size, hidden_size)
        self.Pk = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.Pv = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, _ = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, :].expand(b, self.n_token, self.L).bool()
        
        q = self.Wq(x) # (b, L, d)
        out = F.scaled_dot_product_attention(q, self.Pk, self.Pv, attn_mask=attention_mask, is_causal=False) # (b, L, d)
        return self.dropout(out)


class AttentionLogitsSequence(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, num_labels) -> (b, num_labels)
    """
    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = 'dot'):
        super(AttentionLogitsSequence, self).__init__()
        self.num_labels = num_labels
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))
        self.Wx = Linear(hidden_size, hidden_size)
        self.sim_type = sim_type

    def mean_pooling(self, emb: torch.Tensor, attention_mask: Optional[torch.Tensor] = None): # (b, L, d) -> (b, d)
        if attention_mask is None:
            return emb.mean(dim=1)
        else:
            return (emb * attention_mask).sum(dim=1) / attention_mask.sum(dim=1) # (b, d)

    def dot_product(self, x: torch.Tensor, p: torch.Tensor): # (b, L, d) * (b, d, num_labels) -> (b, L, num_labels)
        return torch.matmul(x, p)

    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor): # (b, L, d) * (b, d, num_labels) -> (b, L, num_labels)
        # x: (b, L, d), p: (b, d, num_labels)
        x_exp = x.unsqueeze(-1)  # (b, L, d, 1)
        p_exp = p.unsqueeze(1)   # (b, 1, d, num_labels)
        dist = torch.abs(torch.norm(x_exp - p_exp, p=2, dim=2))  # (b, L, num_labels)
        return -dist

    def cosine_similarity(self, x: torch.Tensor, p: torch.Tensor): # (b, L, d) * (b, d, num_labels) -> (b, L, num_labels)
        x = F.normalize(x, p=2, dim=-1)
        p = F.normalize(p, p=2, dim=-1)
        return torch.matmul(x, p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, L, d = x.size()
        p = self.Wp.expand(b, -1, -1) # (b, d, num_labels)
        x = self.Wx(x) # (b, L, d)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, None].expand(b, L, self.num_labels) # (b, L, num_labels)

        if self.sim_type == 'dot':
            y = self.dot_product(x, p)
        elif self.sim_type == 'euclidean':
            y = self.euclidean_distance(x, p)
        elif self.sim_type == 'cosine':
            y = self.cosine_similarity(x, p)
        elif self.sim_type == 'mean':
            y = self.mean_pooling(x ,p)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")

        # y (b, L, num_labels)
        logits = self.mean_pooling(y, attention_mask) # (b, num_labels)
        return logits, y, x


class AttentionLogitsToken(nn.Module):
    """
    Cross-attention mechanism for token-parameter-attention (b, L, d) -> (b, L, num_labels)
    """
    def __init__(self, hidden_size: int, num_labels: int = 1, sim_type: str = 'dot'):
        super(AttentionLogitsToken, self).__init__()
        self.num_labels = num_labels
        self.Wp = nn.Parameter(torch.randn(1, hidden_size, num_labels))
        self.Wx = Linear(hidden_size, hidden_size)
        self.sim_type = sim_type

    def dot_product(self, x: torch.Tensor, p: torch.Tensor):
        return torch.matmul(x, p)
    
    def euclidean_distance(self, x: torch.Tensor, p: torch.Tensor):
        return torch.norm(x - p, p=2, dim=-1)
    
    def cosine_similarity(self, x: torch.Tensor, p: torch.Tensor):
        x = F.normalize(x, p=2, dim=-1)
        p = F.normalize(p, p=2, dim=-1)
        return torch.matmul(x, p)

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, L, d = x.size()
        p = self.Wp.expand(b, -1, -1) # (b, d, num_labels)
        x = self.Wx(x) # (b, L, d)
        if self.sim_type == 'dot':
            logits = self.dot_product(x, p)
        elif self.sim_type == 'euclidean':
            logits = self.euclidean_distance(x, p)
        elif self.sim_type == 'cosine':
            logits = self.cosine_similarity(x, p)
        elif self.sim_type == 'mean':
            logits = x.mean(dim=1, keepdim=True).expand(-1, x.size(1), self.num_labels)
        else:
            raise ValueError(f"Invalid similarity type: {self.sim_type}")
        return logits # (b, L, num_labels)


class MultiHeadPAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            n_heads: int,
            n_tokens: int,
            dropout: float = 0.2,
            rotary: bool = True,
            causal: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.Wq = PAttention(hidden_size, n_tokens=n_tokens, dropout=dropout)
        self.Wk = PAttention(hidden_size, n_tokens=n_tokens, dropout=dropout)
        self.Wv = PAttention(hidden_size, n_tokens=n_tokens, dropout=dropout)
        self.out_proj = Linear((hidden_size // n_heads) * n_heads, hidden_size)
        self.q_ln = LayerNorm(hidden_size)
        self.k_ln = LayerNorm(hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None
        self.causal = causal

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention mask already prepped for sdpa shape (bs, 1, seq_len, seq_len)
        b, L, _ = x.shape
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, L, L).bool()
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, seq_len, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask if not self.causal else None, is_causal=self.causal) # (bs, n_heads, seq_len, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)
