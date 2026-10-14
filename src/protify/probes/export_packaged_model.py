"""Push a trained probe, with its backbone recipe, as a Hugging Face repository that `AutoModel` loads.

The repository holds the weights, the tokenizer, a model card, and `packaged_probe_model.py`, the
module `trust_remote_code` loads. That module imports `protify` by absolute name, so a consumer
needs Protify installed; Transformers names the missing package when it is not. The repository
carries no copy of Protify's source: a copy would sit beside the installed package unused, and
could disagree with it. `PackagedProbeConfig.protify_version` records the version that trained it.
"""

import re
import tempfile

from pathlib import Path
from typing import Any
from huggingface_hub import HfApi
from torch import nn

from .hybrid_probe import HybridProbe
from .packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
from ..base_models.supported_models import all_presets_with_paths
from ..console_output import print_message
from ..data.dataset_classes import EmbeddingStandardizer
from ..data.pair_features import DEFAULT_PAIR_FEATURE_MODE
from ..version import PROTIFY_VERSION


def _infer_probe_type(probe_model: nn.Module) -> str:
    probe_class_name = probe_model.__class__.__name__
    if probe_class_name == "MLPProbe":
        return "mlp"
    if probe_class_name in ["TransformerForSequenceClassification", "TransformerForTokenClassification"]:
        return "transformer"
    if probe_class_name in ["LyraForSequenceClassification", "LyraForTokenClassification"]:
        return "lyra"
    raise ValueError(f"Unsupported probe class for packaged export: {probe_class_name}")


def _sae_export_fields(source_model_name: str) -> dict[str, Any]:
    """The backbone path and sparse autoencoder a SAE model name resolves to.

    Returns an empty mapping for a model without one, so the caller can splat it into the
    config either way. The import is function-local because `esmc_sae` loads the vendored
    FastPLMs submodule at import time, and exporting a dense-embedding probe must not
    require it.
    """
    from ..base_models.esmc_sae import ESMC_SAE_COVERAGE, is_sae_model_name, parse_sae_model_name

    if not is_sae_model_name(source_model_name):
        return {}

    selection = parse_sae_model_name(source_model_name)
    return {
        # The packaged model loads the ESM++ backbone and attaches the codebook to it, so
        # base_model_name has to name the backbone rather than the SAE variant.
        "base_model_name": ESMC_SAE_COVERAGE[selection.backbone].model_path,
        "sae_repository": selection.repository,
        "sae_revision": selection.revision,
        "sae_layer": selection.layer,
        "sae_codebook_dim": selection.codebook_dim,
        "sae_k": selection.k,
    }


def _load_sae_backbone(source_model_name: str) -> Any:
    """Build the SAE-attached backbone using the runtime this process already has.

    PackagedProbeModel loads its backbone through AutoModel with trust_remote_code, which
    installs FastPLMs' Hub artifact. That refuses to install when a non-artifact fastplms
    is already imported, which is true of every Protify process, so the packaged model
    cannot build its own backbone at export time. Building it here with the vendored
    runtime sidesteps that: the architecture is the same, so the checkpoint this writes
    loads into the artifact the consumer builds in a clean process.
    """
    from ..base_models.esmc_sae import build_esmc_sae_model

    # build_esmc_sae_model returns the pooling wrapper. The packaged model does its own
    # pooling, so it wants the ESM++ model underneath, with the codebook already attached.
    embedder, _tokenizer = build_esmc_sae_model(source_model_name, pooling_types=('max',))
    return embedder.esm


def _is_supported_base_model(source_model_name: str) -> bool:
    # A fully qualified SAE name is generated from a registry alias rather than listed,
    # so it never appears in all_presets_with_paths.
    if _sae_export_fields(source_model_name):
        return True
    if source_model_name not in all_presets_with_paths:
        return False
    model_name_l = source_model_name.lower()
    return "random" not in model_name_l and "onehot" not in model_name_l


def _extract_sep_token_id(tokenizer: Any) -> int | None:
    try:
        tokenizer_backend = tokenizer.tokenizer
    except AttributeError:
        tokenizer_backend = tokenizer
    if tokenizer_backend.sep_token_id is not None:
        return int(tokenizer_backend.sep_token_id)
    if tokenizer_backend.eos_token_id is not None:
        return int(tokenizer_backend.eos_token_id)
    return None


# Relative imports inside the standalone module, matched at the start of a line so a
# quoted example inside a docstring cannot be rewritten.
_PARENT_RELATIVE_IMPORT = re.compile(r"^from \.\.(\S+) import ", re.MULTILINE)
_SIBLING_RELATIVE_IMPORT = re.compile(r"^from \.(\S+) import ", re.MULTILINE)


def _standalone_packaged_module_source(source: str) -> str:
    """Rewrite the packaged module's relative imports to absolute ones.

    Transformers loads a trust_remote_code module as a top-level module in its dynamic
    module cache, where a relative import has no parent package to resolve against and
    surfaces as a missing sibling file rather than as an import error. The installed protify
    package supplies the absolute paths.

    Parent imports are rewritten first, so the sibling pattern cannot match the leading
    dot of a `..` import that is already handled.
    """
    source = _PARENT_RELATIVE_IMPORT.sub(
        lambda match: f"from protify.{match.group(1)} import ", source
    )
    return _SIBLING_RELATIVE_IMPORT.sub(
        lambda match: f"from protify.probes.{match.group(1)} import ", source
    )


def _write_packaged_module(export_dir: Path) -> None:
    """Write the module `trust_remote_code` loads, its imports made absolute on the installed `protify`."""
    packaged_model_file = Path(__file__).with_name("packaged_probe_model.py")
    standalone_source = _standalone_packaged_module_source(
        packaged_model_file.read_text(encoding="utf-8")
    )
    (export_dir / "packaged_probe_model.py").write_text(standalone_source, encoding="utf-8", newline="\n")


def _build_packaged_model(
    trained_model: nn.Module,
    source_model_name: str,
    probe_args: Any,
    embedding_args: Any,
    tokenizer: Any,
    ppi: bool,
    pair_feature_mode: str,
    feature_selection: Any = None,
    embedding_standardizer: EmbeddingStandardizer | None = None,
) -> PackagedProbeModel:
    if isinstance(trained_model, HybridProbe):
        base_model = trained_model.model
        probe_model = trained_model.probe
    else:
        base_model = None
        probe_model = trained_model

    sae_fields = _sae_export_fields(source_model_name)
    if sae_fields and base_model is None:
        base_model = _load_sae_backbone(source_model_name)

    probe_type = _infer_probe_type(probe_model)
    probe_config_dict = probe_model.config.to_dict()
    sep_token_id = _extract_sep_token_id(tokenizer)
    # A probe trained on selected columns cannot read a full-width embedding, so the index
    # array has to travel with it or inference silently reads the wrong features. It
    # arrives as an argument rather than on probe_args, because get_probe splats that
    # whole object into the probe config, which has to stay JSON-serializable.
    config_fields = {
        "base_model_name": source_model_name,
        "probe_type": probe_type,
        "probe_config": probe_config_dict,
        "tokenwise": probe_args.tokenwise,
        "matrix_embed": embedding_args.matrix_embed,
        "pooling_types": embedding_args.pooling_types,
        "task_type": probe_args.task_type,
        "num_labels": probe_args.num_labels,
        "ppi": ppi,
        "pair_feature_mode": pair_feature_mode,
        "add_token_ids": probe_args.add_token_ids,
        "sep_token_id": sep_token_id,
        "feature_indices": (
            None if feature_selection is None else [int(i) for i in feature_selection.indices]
        ),
        "feature_input_size": None if feature_selection is None else feature_selection.input_size,
        # The probe trained on standardized inputs, so the packaged model standardizes with the
        # training split's statistics, held as buffers in the checkpoint.
        "standardized_width": None if embedding_standardizer is None else int(embedding_standardizer.mean.shape[0]),
        "protify_version": PROTIFY_VERSION,
    }
    # The SAE fields redirect base_model_name from the SAE variant to the ESM++ backbone
    # the codebook reads from, so they replace that entry rather than sitting beside it.
    config_fields.update(sae_fields)
    packaged_config = PackagedProbeConfig(**config_fields)
    packaged_model = PackagedProbeModel(
        config=packaged_config,
        base_model=base_model,
        probe=probe_model,
        standardizer=None if embedding_standardizer is None else (embedding_standardizer.mean, embedding_standardizer.scale),
    )
    return packaged_model.cpu()


def export_packaged_model_to_hub(
    trained_model: nn.Module,
    source_model_name: str,
    probe_args: Any,
    embedding_args: Any,
    tokenizer: Any,
    repo_id: str,
    model_card: str,
    ppi: bool = False,
    pair_feature_mode: str = DEFAULT_PAIR_FEATURE_MODE,
    private: bool = True,
    feature_selection: Any = None,
    embedding_standardizer: EmbeddingStandardizer | None = None,
) -> tuple[bool, str]:
    """Build the packaged model and upload it; Hugging Face reads `HF_TOKEN` or the stored login."""
    if not _is_supported_base_model(source_model_name):
        return False, f"Packaged export is not supported for base model: {source_model_name}"

    packaged_model = _build_packaged_model(
        trained_model=trained_model,
        source_model_name=source_model_name,
        probe_args=probe_args,
        embedding_args=embedding_args,
        tokenizer=tokenizer,
        ppi=ppi,
        pair_feature_mode=pair_feature_mode,
        feature_selection=feature_selection,
        embedding_standardizer=embedding_standardizer,
    )

    with tempfile.TemporaryDirectory(prefix="protify_packaged_model_") as temp_dir:
        export_dir = Path(temp_dir)

        packaged_model.config.auto_map = {
            "AutoConfig": "packaged_probe_model.PackagedProbeConfig",
            "AutoModel": "packaged_probe_model.PackagedProbeModel",
            # The packaged model returns SequenceClassifierOutput, so it also loads
            # through the task class people reach for first.
            "AutoModelForSequenceClassification": "packaged_probe_model.PackagedProbeModel",
        }
        packaged_model.config.architectures = ["PackagedProbeModel"]
        packaged_model.save_pretrained(str(export_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(export_dir))
        _write_packaged_module(export_dir)
        readme_path = export_dir / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            path_in_repo="",
        )

    print_message(f"Packaged model and tokenizer uploaded to Hugging Face Hub: {repo_id}")
    return True, f"Uploaded packaged model to {repo_id}"
