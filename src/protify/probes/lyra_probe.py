import torch
import math
import torch.nn as nn
from typing import Optional
from einops import rearrange, repeat
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from model_components.mlp import intermediate_correction_fn
from pooler import Pooler
from .losses import get_loss_fct


class PGC(nn.Module):
    def __init__(self, d_model, expansion_factor = 1.0, dropout = 0.0):
        super().__init__()

        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        expanded_dim = int(d_model * expansion_factor)

        self.conv = nn.Conv1d(expanded_dim,
                              expanded_dim,
                              kernel_size=3,
                              padding=1,
                              groups=expanded_dim)

        self.in_proj = nn.Linear(d_model, int(d_model * expansion_factor * 2))
        self.out_norm = nn.RMSNorm(int(d_model), eps=1e-8)
        self.in_norm = nn.RMSNorm(expanded_dim * 2, eps=1e-8)
        self.out_proj = nn.Linear(expanded_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u):
        # Input projection and normalization
        xv = self.in_norm(self.in_proj(u))

        # Split projected input into two parts: x and v
        x, v = xv.chunk(2, dim=-1)

        # Depthwise convolution on x
        x_conv = self.conv(x.transpose(-1, -2)).transpose(-1, -2)

        # Gating mechanism
        gate = v * x_conv

        # Output projection and normalization
        x_out = self.out_norm(self.out_proj(gate))

        return x_out


class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))
        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)
        # Pointwise
        self.activation = nn.GELU()
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)
        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y


class LyraLayer(nn.Module):
    def __init__(
            self,
            d_input: int,
            d_output: int,
            d_model: int,
            d_state: int = 64,
            dropout: float = 0.2,
            transposed: bool = False,
            **kernel_args,
        ):
        super().__init__()
        
        self.pgc1 = PGC(d_model, expansion_factor=0.25, dropout=dropout)
        self.pgc2 = PGC(d_model, expansion_factor=2, dropout=dropout)
        self.s4d = S4D(d_model, d_state=d_state, dropout=dropout, transposed=transposed, **kernel_args)
        self.norm = nn.RMSNorm(d_model)
        self.decoder = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pgc1(x)
        x = self.pgc2(x)
        z = x
        z = self.norm(z)
        x = self.dropout(self.s4d(z)) + x
        return x
    

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
            **kernel_args,
        ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([LyraLayer(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            transposed=transposed,
            **kernel_args
        ) for _ in range(n_layers)])

    def forward(self, u):
        x = self.encoder(u)
        for layer in self.layers:
            x = layer(x)
        return x


class LyraConfig(PretrainedConfig):
    model_type = "lyra"
    def __init__(
        self,
        input_dim: int = 29, # protein vocab
        hidden_size: int = 64,
        num_labels: int = 2,
        dropout: float = 0.2,
        n_layers: int = 1,
        task_type: str = 'singlelabel',
        probe_pooling_types: list[str] = ['mean'],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.task_type = task_type 
        self.pooling_types = probe_pooling_types
        self.n_layers = n_layers


class LyraForSequenceClassification(PreTrainedModel):
    config_class = LyraConfig
    def __init__(self, config: LyraConfig):
        super().__init__(config)
        self.lyra = Lyra(
            d_input=config.input_dim,
            d_output=config.num_labels,
            d_model=config.hidden_size,
            dropout=config.dropout,
            n_layers=config.n_layers,
        )

        self.pooler = Pooler(config.pooling_types)
        classifier_dim = intermediate_correction_fn(2.0, config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, config.num_labels),
        )
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.task_type = config.task_type

    def forward(
            self,
            embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        x = self.lyra(embeddings)
        x = self.pooler(x, attention_mask)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class LyraForTokenClassification(PreTrainedModel):
    config_class = LyraConfig
    def __init__(self, config: LyraConfig):
        super().__init__(config)
        self.lyra = Lyra(
            d_input=config.input_dim,
            d_output=config.num_labels,
            d_model=config.hidden_size,
            dropout=config.dropout,
            n_layers=config.n_layers,
        )
        self.loss_fct = get_loss_fct(config.task_type)
        classifier_dim = intermediate_correction_fn(2.0, config.num_labels)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, classifier_dim),
            nn.GELU(),
            nn.Linear(classifier_dim, config.num_labels),
        )
        self.loss_fct = get_loss_fct(config.task_type)
        self.num_labels = config.num_labels
        self.task_type = config.task_type

    def forward(
            self,
            embeddings: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
    ) -> TokenClassifierOutput:
        x = self.lyra(embeddings)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            if self.task_type == 'regression':
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.task_type == 'multilabel':
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1).long())

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
    x = torch.randint(0, 2, (batch_size, seq_length, input_size)).float()
    attention_mask = torch.ones(batch_size, seq_length)
    labels = torch.randint(0, 2, (batch_size,))
    
    outputs = seq_model(x, attention_mask=attention_mask, labels=labels)
    print(f"Loss: {outputs.loss.item()}")
    print(f"Logits shape: {outputs.logits.shape}")
    
    # Backward pass
    outputs.loss.backward()
    print("Backward pass completed successfully")
    