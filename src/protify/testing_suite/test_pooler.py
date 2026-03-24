import torch
import pytest

try:
    from src.protify.pooler import Pooler
except ImportError:
    try:
        from protify.pooler import Pooler
    except ImportError:
        from ..pooler import Pooler


B, L, D = 2, 4, 8


@pytest.fixture
def emb():
    torch.manual_seed(0)
    return torch.randn(B, L, D)


@pytest.fixture
def mask():
    # first sample: 3 real tokens, second: all 4
    return torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)


@pytest.fixture
def attentions():
    torch.manual_seed(0)
    n_layers = 2
    return torch.randn(B, n_layers, L, L).abs()


# ---- mean pooling ----

def test_mean_no_mask(emb):
    pooler = Pooler(['mean'])
    result = pooler.mean_pooling(emb)
    expected = emb.mean(dim=1)
    assert result.shape == (B, D)
    assert torch.allclose(result, expected)


def test_mean_with_mask(emb, mask):
    pooler = Pooler(['mean'])
    result = pooler.mean_pooling(emb, attention_mask=mask)
    assert result.shape == (B, D)
    # Manual check for first sample: mean of first 3 tokens
    manual = emb[0, :3, :].mean(dim=0)
    assert torch.allclose(result[0], manual, atol=1e-6)


# ---- max pooling ----

def test_max_no_mask(emb):
    pooler = Pooler(['max'])
    result = pooler.max_pooling(emb)
    expected = emb.max(dim=1).values
    assert result.shape == (B, D)
    assert torch.allclose(result, expected)


def test_max_with_mask(emb, mask):
    pooler = Pooler(['max'])
    result = pooler.max_pooling(emb, attention_mask=mask)
    assert result.shape == (B, D)


def test_max_with_mask_negative_values():
    """Masked positions must not win when all unmasked values are negative."""
    torch.manual_seed(0)
    emb = torch.full((1, 4, D), -5.0)
    emb[0, 0, :] = -1.0  # only first token is "less negative"
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.float32)
    pooler = Pooler(['max'])
    result = pooler.max_pooling(emb, attention_mask=mask)
    # Max of unmasked positions should be -1.0, not 0.0 from masked
    assert torch.allclose(result, torch.full((1, D), -1.0))


# ---- norm pooling ----

def test_norm_no_mask(emb):
    pooler = Pooler(['norm'])
    result = pooler.norm_pooling(emb)
    expected = emb.norm(dim=1, p=2)
    assert result.shape == (B, D)
    assert torch.allclose(result, expected)


def test_norm_with_mask(emb, mask):
    pooler = Pooler(['norm'])
    result = pooler.norm_pooling(emb, attention_mask=mask)
    assert result.shape == (B, D)


# ---- median pooling ----

def test_median_no_mask(emb):
    pooler = Pooler(['median'])
    result = pooler.median_pooling(emb)
    expected = emb.median(dim=1).values
    assert result.shape == (B, D)
    assert torch.allclose(result, expected)


# ---- var pooling ----

def test_var_no_mask(emb):
    pooler = Pooler(['var'])
    result = pooler.var_pooling(emb)
    expected = emb.var(dim=1)
    assert result.shape == (B, D)
    assert torch.allclose(result, expected)


def test_var_with_mask(emb, mask):
    pooler = Pooler(['var'])
    result = pooler.var_pooling(emb, attention_mask=mask)
    assert result.shape == (B, D)
    # Manual variance for first sample over 3 unmasked tokens (population variance)
    unmasked = emb[0, :3, :]  # (3, D)
    mean = unmasked.mean(dim=0)
    manual_var = ((unmasked - mean) ** 2).mean(dim=0)
    assert torch.allclose(result[0], manual_var, atol=1e-6)


# ---- std pooling ----

def test_std_equals_sqrt_var_with_mask(emb, mask):
    pooler = Pooler(['std'])
    std_result = pooler.std_pooling(emb, attention_mask=mask)
    var_result = pooler.var_pooling(emb, attention_mask=mask)
    assert torch.allclose(std_result, torch.sqrt(var_result), atol=1e-6)


# ---- cls pooling ----

def test_cls_returns_first_token(emb):
    pooler = Pooler(['cls'])
    result = pooler.cls_pooling(emb)
    assert result.shape == (B, D)
    assert torch.allclose(result, emb[:, 0, :])


def test_cls_ignores_mask(emb, mask):
    pooler = Pooler(['cls'])
    result_no_mask = pooler.cls_pooling(emb)
    result_with_mask = pooler.cls_pooling(emb, attention_mask=mask)
    assert torch.allclose(result_no_mask, result_with_mask)


# ---- parti pooling ----

def test_parti_shape(emb, attentions, mask):
    pooler = Pooler(['parti'])
    result = pooler._pool_parti(emb, attentions, attention_mask=mask)
    assert result.shape == (B, D)


# ---- __call__ ----

def test_call_single_type(emb):
    pooler = Pooler(['mean'])
    result = pooler(emb)
    assert result.shape == (B, D)


def test_call_multiple_types_concat(emb, mask):
    types = ['mean', 'max', 'cls']
    pooler = Pooler(types)
    result = pooler(emb, attention_mask=mask)
    assert result.shape == (B, len(types) * D)
