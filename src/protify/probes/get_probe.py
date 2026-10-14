"""Build a probe from its arguments, or rebuild one from a saved config.

Protify defines the `mlp`, `transformer`, and `lyra` probes. Another package adds a probe type
with `register_probe_type`, after which `ProbeArguments`, `get_probe`, and
`rebuild_probe_from_saved_config` treat it as they treat Protify's own.
"""

import importlib

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from transformers import PreTrainedModel

from .lyra_probe import LyraForSequenceClassification, LyraForTokenClassification, LyraConfig
from .mlp_probe import MLPProbe, MLPProbeConfig
from .transformer_probe import TransformerForSequenceClassification, TransformerForTokenClassification, TransformerProbeConfig


# The feed-forward probe was called a linear probe before it grew hidden layers. Saved
# probe configs, published repositories, and existing YAML all carry the old value, so it
# resolves to the current name rather than failing.
_RENAMED_PROBE_TYPES = {'linear': 'mlp'}


def normalize_probe_type(probe_type: str) -> str:
    return _RENAMED_PROBE_TYPES.get(probe_type, probe_type)


# (probe type, fields or saved config, tokenwise) -> probe
ProbeBuilder = Callable[[str, dict[str, Any], bool], PreTrainedModel]


@dataclass(frozen=True)
class ProbeType:
    """A probe type another package defines.

    `build(probe_type, fields, tokenwise)` returns a new probe from the `ProbeArguments` fields,
    and `rebuild(probe_type, config, tokenwise)` one from a saved config, which may predate fields
    the type now stores; `build` serves both when no `rebuild` is given. `settings(probe_type,
    options)` returns the fields the type keeps beyond the shared ones, read from the keyword
    arguments `ProbeArguments` was given, with the type's defaults for those absent; they are saved
    with every probe of the type and with no other, so registering a type never changes what
    another type serializes.
    """

    build: ProbeBuilder
    rebuild: ProbeBuilder | None = None
    settings: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
    # Who defines the type, so the same definition registering again is told from a different one.
    origin: str = ""


_REGISTERED_PROBE_TYPES: dict[str, ProbeType] = {}


def register_probe_type(probe_type: str, build: ProbeBuilder, *, rebuild: ProbeBuilder | None = None,
                        settings: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None) -> None:
    """Add a probe type; registering a name Protify defines, or one another builder holds, raises."""
    _register(probe_type, ProbeType(build, rebuild, settings, origin=_qualified_name(build)))


def register_external_probe(probe_type: str, module: str, seq_cls: str, tok_cls: str, config_cls: str) -> None:
    """Register a probe type by the names of its module, model classes, and config class."""
    def build(_probe_type: str, fields: dict[str, Any], tokenwise: bool) -> PreTrainedModel:
        probe_module = importlib.import_module(module)
        config = getattr(probe_module, config_cls)(**fields)
        return getattr(probe_module, tok_cls if tokenwise else seq_cls)(config)

    _register(probe_type, ProbeType(build, origin=f"{module}:{seq_cls},{tok_cls},{config_cls}"))


def _register(probe_type: str, definition: ProbeType) -> None:
    if probe_type in ('mlp', 'linear', 'transformer', 'lyra'):
        raise ValueError(f"{probe_type!r} is a Protify probe type and cannot be registered")
    previous = _REGISTERED_PROBE_TYPES.get(probe_type)
    # A module imported again (a reload, or a second import path) registers the same definition anew.
    if previous is not None and previous.origin != definition.origin:
        raise ValueError(f"probe type {probe_type!r} is already registered by {previous.origin}")
    _REGISTERED_PROBE_TYPES[probe_type] = definition


def _qualified_name(function: Callable[..., Any]) -> str:
    return f"{function.__module__}.{function.__qualname__}"


def registered_probe_types() -> tuple[str, ...]:
    return tuple(sorted(_REGISTERED_PROBE_TYPES))


@dataclass
class ProbeArguments:
    def __init__(
            self,
            probe_type: str = 'mlp', # Protify's: mlp, transformer, lyra; estimators: xgboost, lightgbm, random_forest; plus any registered type
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
        # get_probe hands this whole object to the probe config, which saves every field, so a
        # registered type's own settings are stored only for that type.
        registered = _REGISTERED_PROBE_TYPES.get(self.probe_type)
        if registered is not None and registered.settings is not None:
            for name, value in registered.settings(self.probe_type, kwargs).items():
                setattr(self, name, value)


def _build_protify_probe(probe_type: str, tokenwise: bool, fields: dict[str, Any]) -> PreTrainedModel | None:
    if probe_type == 'mlp' and not tokenwise:
        return MLPProbe(MLPProbeConfig(**fields))
    if probe_type == 'transformer':
        config = TransformerProbeConfig(**fields)
        return TransformerForTokenClassification(config) if tokenwise else TransformerForSequenceClassification(config)
    if probe_type == 'lyra':
        config = LyraConfig(**fields)
        return LyraForTokenClassification(config) if tokenwise else LyraForSequenceClassification(config)
    return None


def get_probe(args: ProbeArguments) -> PreTrainedModel:
    if args.pooling_types is not None and 'bom' in args.pooling_types:
        assert args.probe_type == 'transformer' and not args.tokenwise, (
            f"'bom' pooling is only supported by the transformer sequence probe "
            f"(probe_type='transformer', tokenwise=False). Got probe_type={args.probe_type!r}, "
            f"tokenwise={args.tokenwise}."
        )
    probe = _build_protify_probe(args.probe_type, args.tokenwise, args.__dict__)
    if probe is not None:
        return probe
    if args.probe_type in _REGISTERED_PROBE_TYPES:
        return _REGISTERED_PROBE_TYPES[args.probe_type].build(args.probe_type, dict(args.__dict__), args.tokenwise)
    raise ValueError(
        f"Invalid combination of probe type and tokenwise: {args.probe_type} {args.tokenwise}. "
        f"Registered probe types: {registered_probe_types()}"
    )


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
    probe = _build_protify_probe(probe_type, tokenwise, config_dict)
    if probe is not None:
        return probe
    if probe_type in _REGISTERED_PROBE_TYPES:
        registered = _REGISTERED_PROBE_TYPES[probe_type]
        return (registered.rebuild or registered.build)(probe_type, config_dict, tokenwise)
    raise ValueError(
        f"Unsupported saved probe configuration: {probe_type} tokenwise={tokenwise}. "
        f"Registered probe types: {registered_probe_types()}"
    )
