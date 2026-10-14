import importlib

from dataclasses import dataclass
from typing import Any
from transformers import PreTrainedModel

from .lyra_probe import LyraForSequenceClassification, LyraForTokenClassification, LyraConfig
from .mlp_probe import MLPProbe, MLPProbeConfig
from .projected_probe import PROJECTED_PROBE_OUTPUTS, PROJECTED_PROBE_TRUNKS, ProjectedProbe, ProjectedProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig


# The feed-forward probe was called a linear probe before it grew hidden layers. Saved
# probe configs, published repositories, and existing YAML all carry the old value, so it
# resolves to the current name rather than failing.
_RENAMED_PROBE_TYPES = {'linear': 'mlp'}


def normalize_probe_type(probe_type: str) -> str:
    return _RENAMED_PROBE_TYPES.get(probe_type, probe_type)


# External probe types can be registered here by packages on PYTHONPATH.
# Each entry maps probe_type -> {"module": "dotted.module.path", "seq_cls": "ClassName", "tok_cls": "ClassName", "config_cls": "ConfigName"}
_EXTERNAL_PROBE_REGISTRY: dict[str, dict[str, str]] = {}


def register_external_probe(probe_type: str, module: str, seq_cls: str, tok_cls: str, config_cls: str) -> None:
    _EXTERNAL_PROBE_REGISTRY[probe_type] = {
        "module": module,
        "seq_cls": seq_cls,
        "tok_cls": tok_cls,
        "config_cls": config_cls,
    }


def _load_external_probe(probe_type: str, tokenwise: bool, config_kwargs: dict[str, Any]) -> PreTrainedModel | None:
    if probe_type not in _EXTERNAL_PROBE_REGISTRY:
        return None
    entry = _EXTERNAL_PROBE_REGISTRY[probe_type]
    mod = importlib.import_module(entry["module"])
    config_cls = getattr(mod, entry["config_cls"])
    model_cls_name = entry["tok_cls"] if tokenwise else entry["seq_cls"]
    model_cls = getattr(mod, model_cls_name)
    config = config_cls(**config_kwargs)
    return model_cls(config)


@dataclass
class ProbeArguments:
    def __init__(
            self,
            probe_type: str = 'mlp', # valid options: mlp, transformer, lyra, projected, sngp, projected_linear, projected_faithful, projected_beta_nll, xgboost, lightgbm, random_forest
            tokenwise: bool = False,
            ### MLP Probe
            input_size: int = 960,
            hidden_size: int = 8192,
            dropout: float = 0.2,
            num_labels: int = 2,
            n_layers: int = 1,
            task_type: str = 'singlelabel',
            pre_ln: bool = True,
            sim_type: str = 'dot',
            use_bias: bool = False,
            add_token_ids: bool = False,
            ### Feature selection
            feature_selection: bool = False,
            feature_selection_min_size: int = 256,
            ### Transformer Probe
            classifier_size: int = 4096,
            transformer_dropout: float = 0.1,
            classifier_dropout: float = 0.2,
            head_size: int = 128,
            rotary: bool = True,
            attention_backend: str = "flex",
            output_s_max: bool = False,
            probe_pooling_types: list[str] | None = None,
            bom_k: int = 60,
            expansion_ratio: float = 8 / 3,
            max_length: int = 2048,
            ### Projected and SNGP probes
            projection_size: int = 1024,
            projected_hidden_size: int | None = None,
            spectral_bound: float = 0.95,
            input_clip: float | None = 10.0,
            branch_widths: list[int] | None = None,
            branch_sizes: list[int] | None = None,
            ### LoRA
            lora: bool = False,
            lora_r: int = 8,
            lora_alpha: float = 32.0,
            lora_dropout: float = 0.01,
            **kwargs: Any,

    ) -> None:
        self.probe_type = normalize_probe_type(probe_type)
        self.tokenwise = tokenwise
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_labels = num_labels
        self.n_layers = n_layers
        self.sim_type = sim_type
        self.add_token_ids = add_token_ids
        self.feature_selection = feature_selection
        self.feature_selection_min_size = feature_selection_min_size
        self.task_type = task_type
        self.pre_ln = pre_ln
        self.use_bias = use_bias
        self.classifier_size = classifier_size
        self.transformer_dropout = transformer_dropout
        self.classifier_dropout = classifier_dropout
        self.head_size = head_size
        legacy_n_heads = kwargs.pop("n_heads", None)
        if legacy_n_heads is not None and self.hidden_size % legacy_n_heads == 0:
            self.head_size = self.hidden_size // legacy_n_heads
        self.rotary = rotary
        self.attention_backend = attention_backend
        self.output_s_max = output_s_max
        # None leaves each probe its own default pooling: mean and cls for the transformer probe,
        # mean for Lyra. The probe configs read this as `pooling_types`, the name they store it under.
        self.pooling_types = probe_pooling_types
        self.bom_k = bom_k
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_length
        self.lora = lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # get_probe hands this whole object to the probe config, which saves every field. The
        # projected head's settings are stored only for that head, so every other probe type
        # keeps serializing exactly the fields it did before the head existed.
        if self.probe_type in PROJECTED_PROBE_OUTPUTS:
            self.projection_size = projection_size
            self.projected_trunk = PROJECTED_PROBE_TRUNKS.get(self.probe_type, 'projected')
            if projected_hidden_size is not None:
                self.projected_hidden_size = projected_hidden_size
            elif self.probe_type == 'projected':
                self.projected_hidden_size = 8192
            self.spectral_bound = spectral_bound
            self.input_clip = input_clip
            self.branch_widths = branch_widths
            self.branch_sizes = branch_sizes


def get_probe(args: ProbeArguments) -> PreTrainedModel:
    if args.pooling_types is not None and 'bom' in args.pooling_types:
        assert args.probe_type == 'transformer' and not args.tokenwise, (
            f"'bom' pooling is only supported by the transformer sequence probe "
            f"(probe_type='transformer', tokenwise=False). Got probe_type={args.probe_type!r}, "
            f"tokenwise={args.tokenwise}."
        )
    if args.probe_type == 'mlp' and not args.tokenwise:
        config = MLPProbeConfig(**args.__dict__)
        return MLPProbe(config)
    elif args.probe_type == 'transformer' and not args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForSequenceClassification(config)
    elif args.probe_type == 'transformer' and args.tokenwise:
        config = TransformerProbeConfig(**args.__dict__)
        return TransformerForTokenClassification(config)
    elif args.probe_type == 'lyra' and not args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForSequenceClassification(config)
    elif args.probe_type == 'lyra' and args.tokenwise:
        config = LyraConfig(**args.__dict__)
        return LyraForTokenClassification(config)
    elif args.probe_type in PROJECTED_PROBE_OUTPUTS and not args.tokenwise:
        config = ProjectedProbeConfig(**{
            **args.__dict__,
            'output': PROJECTED_PROBE_OUTPUTS[args.probe_type],
            'trunk': PROJECTED_PROBE_TRUNKS.get(args.probe_type, 'projected'),
        })
        return ProjectedProbe(config)
    else:
        external = _load_external_probe(args.probe_type, args.tokenwise, args.__dict__)
        if external is not None:
            return external
        raise ValueError(f"Invalid combination of probe type and tokenwise: {args.probe_type} {args.tokenwise}")


def rebuild_probe_from_saved_config(
        probe_type: str,
        tokenwise: bool,
        probe_config: dict[str, Any],
    ) -> PreTrainedModel:
    config_dict = dict(probe_config)
    if "num_labels" not in config_dict and "id2label" in config_dict:
        config_dict["num_labels"] = len(config_dict["id2label"])
    if "pooling_types" in config_dict and "probe_pooling_types" not in config_dict:
        config_dict["probe_pooling_types"] = config_dict["pooling_types"]

    probe_type = normalize_probe_type(probe_type)

    if probe_type == "mlp" and not tokenwise:
        config = MLPProbeConfig(**config_dict)
        return MLPProbe(config)
    if probe_type == "transformer" and not tokenwise:
        config = TransformerProbeConfig(**config_dict)
        return TransformerForSequenceClassification(config)
    if probe_type == "transformer" and tokenwise:
        config = TransformerProbeConfig(**config_dict)
        return TransformerForTokenClassification(config)
    if probe_type == "lyra" and not tokenwise:
        config = LyraConfig(**config_dict)
        return LyraForSequenceClassification(config)
    if probe_type == "lyra" and tokenwise:
        config = LyraConfig(**config_dict)
        return LyraForTokenClassification(config)
    if probe_type in PROJECTED_PROBE_OUTPUTS and not tokenwise:
        config = ProjectedProbeConfig(**{
            **config_dict,
            "output": PROJECTED_PROBE_OUTPUTS[probe_type],
            "trunk": config_dict.get("trunk", "linear" if probe_type == "projected_linear" else "projected"),
        })
        return ProjectedProbe(config)
    external = _load_external_probe(probe_type, tokenwise, config_dict)
    if external is not None:
        return external
    raise ValueError(f"Unsupported saved probe configuration: {probe_type} tokenwise={tokenwise}")
