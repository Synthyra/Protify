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
            input_dim: int = 960,
            hidden_dim: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
            sim_type: str = 'dot',
            ### Transformer Probe
            classifier_dim: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            n_heads: int = 4,
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
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.task_type = task_type
        self.pre_ln = pre_ln
        self.classifier_dim = classifier_dim
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.n_heads = n_heads
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
