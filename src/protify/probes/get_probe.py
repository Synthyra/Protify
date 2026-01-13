from dataclasses import dataclass, field
from typing import List
from .linear_probe import LinearProbe, LinearProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig
from .retrievalnet import RetrievalNetForSequenceClassification, RetrievalNetForTokenClassification, RetrievalNetConfig
from .lyra_probe import LyraForSequenceClassification, LyraForTokenClassification, LyraConfig


@dataclass
class ProbeArguments:
    def __init__(
            self,
            probe_type: str = 'linear', # valid options: linear, transformer, retrievalnet
            tokenwise: bool = False,
            ### Linear Probe
            input_size: int = 960,
            hidden_size: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
            sim_type: str = 'dot',
            token_attention: bool = False,
            ### Transformer Probe
            classifier_size: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            head_size: int = 64,
            rotary: bool = True,
            probe_pooling_types: List[str] = field(default_factory=lambda: ['mean', 'cls']),
            ### RetrievalNet
            # TODO
            ### LoRA
            lora: bool = False,
            lora_r: int = 8,
            lora_alpha: float = 32.0,
            lora_dropout: float = 0.01,
            **kwargs,

    ):
        self.probe_type = probe_type
        self.tokenwise = tokenwise
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.token_attention = token_attention
        self.task_type = task_type
        self.pre_ln = pre_ln
        self.classifier_size = classifier_size
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.head_size = head_size
        if 'n_heads' in kwargs:
            raise ValueError(
                "n_heads has been deprecated. Please remove it from your config and set head_size instead "
                "(n_heads is computed as hidden_size // head_size)."
            )
        if head_size <= 0:
            raise ValueError(f"head_size must be > 0, got {head_size}")
        if hidden_size % head_size != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by head_size ({head_size}). "
                f"Got remainder {hidden_size % head_size}."
            )
        self.n_heads = hidden_size // head_size
        self.rotary = rotary
        self.pooling_types = probe_pooling_types
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout


def get_probe(args: ProbeArguments):
    if args.probe_type == 'linear' and not args.tokenwise:
        config = LinearProbeConfig(**args.__dict__)
        return LinearProbe(config)
    elif args.probe_type == 'transformer' and not args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForSequenceClassification(config)
    elif args.probe_type == 'transformer' and args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForTokenClassification(config)
    elif args.probe_type == 'retrievalnet' and not args.tokenwise:
        config = RetrievalNetConfig(**args.__dict__)
        return RetrievalNetForSequenceClassification(config)
    elif args.probe_type == 'retrievalnet' and args.tokenwise:
        config = RetrievalNetConfig(**args.__dict__)
        return RetrievalNetForTokenClassification(config)
    elif args.probe_type == 'lyra' and not args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForSequenceClassification(config)
    elif args.probe_type == 'lyra' and args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForTokenClassification(config)
    else:
        raise ValueError(f"Invalid combination of probe type and tokenwise: {args.probe_type} {args.tokenwise}")
