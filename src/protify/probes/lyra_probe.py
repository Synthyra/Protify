import math
import torch
import torch.nn as nn

from typing import Any
from einops import rearrange, repeat
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
try:
    from ..model_components.mlp import intermediate_correction_fn
except ImportError:
    try:
        from protify.model_components.mlp import intermediate_correction_fn
    except ImportError:
        from model_components.mlp import intermediate_correction_fn

try:
    from ..pooler import Pooler
except ImportError:
    try:
        from protify.pooler import Pooler
    except ImportError:
        from pooler import Pooler
from .losses import get_loss_fct


class PGC(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion_factor: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.d_model = d_model  # d
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        expanded_dim = int(d_model * expansion_factor)  # e

        self.conv = nn.Conv1d(
            expanded_dim,
            expanded_dim,
            kernel_size=3,
            padding=1,
            groups=expanded_dim,
        )

        self.in_proj = nn.Linear(d_model, int(d_model * expansion_factor * 2))
        self.out_norm = nn.RMSNorm(int(d_model), eps=1e-8)
        self.in_norm = nn.RMSNorm(expanded_dim * 2, eps=1e-8)
        self.out_proj = nn.Linear(expanded_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (b, l, d); e is the expanded width.
        projected = self.in_proj(u)  # (b, l, 2 * e)
        xv = self.in_norm(projected)  # (b, l, 2 * e)
        x, v = xv.chunk(2, dim=-1)  # each (b, l, e)
        x_channels_first = x.transpose(-1, -2)  # (b, e, l)
        x_conv = self.conv(x_channels_first).transpose(-1, -2)  # (b, l, e)
        gate = v * x_conv  # (b, l, e)
        x_out = self.out_proj(gate)  # (b, l, d)
        x_out = self.out_norm(x_out)  # (b, l, d)
        return x_out  # (b, l, d)


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True) -> None:
        """Optionally tie the dropout mask across all non-channel dimensions."""
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability has to be in [0, 1), but got {p}")
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (b, d, ...) when transposed, otherwise (b, ..., d).
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')  # (b, d, ...)
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p  # (b, d, 1, ...) or X.shape
            X = X * mask * (1.0 / (1 - self.p))  # (b, d, ...)
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')  # (b, ..., d)
            return X  # same shape as input
        return X  # same shape as input


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(
        self,
        d_model: int,
        N: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: float | None = None,
    ) -> None:
        super().__init__()
        hidden_size = d_model  # d
        log_dt = torch.rand(hidden_size) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)  # (d,)

        C = torch.randn(hidden_size, N // 2, dtype=torch.cfloat)  # (d, n / 2)
        self.C = nn.Parameter(torch.view_as_real(C))  # (d, n / 2, 2)
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(hidden_size, N // 2))  # (d, n / 2)
        A_imag = math.pi * repeat(
            torch.arange(N // 2),
            'n -> h n',
            h=hidden_size,
        )  # (d, n / 2)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L: int) -> torch.Tensor:
        # L is sequence length l; n is the configured state width.
        dt = torch.exp(self.log_dt)  # (d,)
        C = torch.view_as_complex(self.C)  # (d, n / 2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (d, n / 2)
        dtA = A * dt.unsqueeze(-1)  # (d, n / 2)
        time_steps = torch.arange(L, device=A.device)  # (l,)
        vandermonde = dtA.unsqueeze(-1) * time_steps  # (d, n / 2, l)
        C = C * (torch.exp(dtA) - 1.0) / A  # (d, n / 2)
        kernel = 2 * torch.einsum(
            'dn,dnl->dl',
            C,
            torch.exp(vandermonde),  # (d, n / 2, l)
        ).real  # (d, l)
        return kernel  # (d, l)

    def register(
        self,
        name: str,
        tensor: torch.Tensor,
        lr: float | None = None,
    ) -> None:
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = True,
        **kernel_args: Any,
    ) -> None:
        super().__init__()

        self.h = d_model  # d
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))  # (d,)
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # u: (b, d, l) when transposed, otherwise (b, l, d). Extra kwargs are ignored.
        if not self.transposed:
            u = u.transpose(-1, -2)  # (b, d, l)
        sequence_length = u.size(-1)
        kernel = self.kernel(L=sequence_length)  # (d, l)
        kernel_f = torch.fft.rfft(kernel, n=2 * sequence_length)  # (d, l + 1)
        u_f = torch.fft.rfft(u, n=2 * sequence_length)  # (b, d, l + 1)
        y = torch.fft.irfft(
            u_f * kernel_f,  # (b, d, l + 1)
            n=2 * sequence_length,
        )[..., :sequence_length]  # (b, d, l)
        y = y + u * self.D.unsqueeze(-1)  # (b, d, l)
        y = self.activation(y)  # (b, d, l)
        y = self.dropout(y)  # (b, d, l)
        y = self.output_linear(y)  # (b, d, l)
        if not self.transposed:
            y = y.transpose(-1, -2)  # (b, l, d)
        return y  # same shape as input


class LyraLayer(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.2,
        transposed: bool = False,
        **kernel_args: Any,
    ) -> None:
        super().__init__()

        self.pgc1 = PGC(d_model, expansion_factor=0.25, dropout=dropout)
        self.pgc2 = PGC(d_model, expansion_factor=2, dropout=dropout)
        self.s4d = S4D(
            d_model,
            d_state=d_state,
            dropout=dropout,
            transposed=transposed,
            **kernel_args,
        )
        self.norm = nn.RMSNorm(d_model)
        self.decoder = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The composed LyraLayer contract supports transposed=False: x is (b, l, d).
        # transposed=True remains exposed for compatibility, but PGC still emits (b, l, d)
        # while S4D expects (b, d, l), so that historical mode is layout-inconsistent.
        x = self.pgc1(x)  # (b, l, d)
        x = self.pgc2(x)  # (b, l, d)
        residual = x  # (b, l, d)
        normalized = self.norm(x)  # (b, l, d)
        state_output = self.s4d(normalized)  # (b, l, d)
        state_output = self.dropout(state_output)  # (b, l, d)
        x = state_output + residual  # (b, l, d)
        return x  # (b, l, d)


class Lyra(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.2,
        transposed: bool = False,
        n_layers: int = 1,
        **kernel_args: Any,
    ) -> None:
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList(
            [
                LyraLayer(
                    d_input=d_input,
                    d_output=d_output,
                    d_model=d_model,
                    d_state=d_state,
                    dropout=dropout,
                    transposed=transposed,
                    **kernel_args,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (b, l, d_in).
        x = self.encoder(u)  # (b, l, d)
        for layer in self.layers:
            x = layer(x)  # (b, l, d)
        return x  # (b, l, d)


class LyraConfig(PretrainedConfig):
    model_type = "lyra"

    def __init__(
        self,
        input_size: int = 29,
        hidden_size: int = 64,
        num_labels: int = 2,
        dropout: float = 0.2,
        n_layers: int = 1,
        task_type: str = 'singlelabel',
        probe_pooling_types: list[str] = ['mean'],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.task_type = task_type
        self.pooling_types = probe_pooling_types
        self.n_layers = n_layers


class LyraForSequenceClassification(PreTrainedModel):
    config_class = LyraConfig
    all_tied_weights_keys = {}

    def __init__(self, config: LyraConfig) -> None:
        super().__init__(config)
        self.lyra = Lyra(
            d_input=config.input_size,
            d_output=config.num_labels,
            d_model=config.hidden_size,
            dropout=config.dropout,
            n_layers=config.n_layers,
        )

        self.pooler = Pooler(config.pooling_types)
        classifier_size = intermediate_correction_fn(2.0, config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, classifier_size),
            nn.GELU(),
            nn.Linear(classifier_size, config.num_labels),
        )
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels  # c
        self.task_type = config.task_type

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        # embeddings: (b, l, d_in); attention_mask: (b, l) or None.
        # labels: (b,), (b, 1), or (b, c), depending on the task.
        embeddings = embeddings.to(next(self.lyra.parameters()).dtype)  # (b, l, d_in)
        hidden_states = self.lyra(embeddings)  # (b, l, d)
        # p is the configured number of pooling operations. The existing classifier contract assumes p == 1.
        pooled = self.pooler(hidden_states, attention_mask)  # (b, p * d)
        logits = self.classifier(pooled)  # (b, c), when p == 1
        if self.task_type == 'sigmoid_regression':
            logits = logits.sigmoid()  # (b, c)

        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(
                    logits.view(-1),  # (b * c,)
                    labels.view(-1).float(),  # (b * c,)
                )  # ()
            elif self.task_type == 'sigmoid_regression':
                loss = self.loss_fct(
                    logits.view(-1),  # (b * c,)
                    labels.view(-1).float(),  # (b * c,)
                )  # ()
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())  # () from (b, c), (b, c)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels),  # (b, c)
                    labels.view(-1).long(),  # (b,)
                )  # ()

        # logits: (b, c); loss: () or None.
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class LyraForTokenClassification(PreTrainedModel):
    config_class = LyraConfig
    all_tied_weights_keys = {}

    def __init__(self, config: LyraConfig) -> None:
        super().__init__(config)
        self.lyra = Lyra(
            d_input=config.input_size,
            d_output=config.num_labels,
            d_model=config.hidden_size,
            dropout=config.dropout,
            n_layers=config.n_layers,
        )
        self.loss_fct = get_loss_fct(config.task_type)
        classifier_size = intermediate_correction_fn(2.0, config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, classifier_size),
            nn.GELU(),
            nn.Linear(classifier_size, config.num_labels),
        )
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels  # c
        self.task_type = config.task_type

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TokenClassifierOutput:
        # embeddings: (b, l, d_in); attention_mask is accepted but unused.
        # labels: (b, l), (b, l, 1), or (b, l, c), depending on the task.
        embeddings = embeddings.to(next(self.lyra.parameters()).dtype)  # (b, l, d_in)
        hidden_states = self.lyra(embeddings)  # (b, l, d)
        logits = self.classifier(hidden_states)  # (b, l, c)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(
                    logits.view(-1),  # (b * l * c,)
                    labels.view(-1).float(),  # (b * l * c,)
                )  # ()
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())  # () from (b, l, c), (b, l, c)
            else:
                loss = self.loss_fct(
                    logits.view(-1, self.num_labels),  # (b * l, c)
                    labels.view(-1).long(),  # (b * l,)
                )  # ()

        # logits: (b, l, c); loss: () or None.
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


if __name__ == "__main__":
    # py -m probes.lyra_probe
    # Test sequence classification model
    print("\nTesting LyraForSequenceClassification")
    config = LyraConfig()
    seq_model = LyraForSequenceClassification(config)
    seq_model.train()
    
    # Forward pass
    batch_size = 2
    seq_length = 64
    input_size = 20
    x = torch.randint(0, 2, (batch_size, seq_length, input_size)).float()  # (b, l, d_in)
    attention_mask = torch.ones(batch_size, seq_length)  # (b, l)
    labels = torch.randint(0, 2, (batch_size,))  # (b,)

    outputs = seq_model(x, attention_mask=attention_mask, labels=labels)
    print(f"Loss: {outputs.loss.item()}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Backward pass
    outputs.loss.backward()
    print("Backward pass completed successfully")
