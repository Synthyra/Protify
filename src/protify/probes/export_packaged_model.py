import re
import shutil
import tempfile

import foundry.embedding
import foundry.layers
import interpnet

from pathlib import Path
from typing import Any
from huggingface_hub import HfApi
from torch import nn


try:
    from base_models.supported_models import all_presets_with_paths
    from data.pair_features import DEFAULT_PAIR_FEATURE_MODE
    from probes.hybrid_probe import HybridProbe
    from probes.packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
    from console_output import print_message
except ImportError:
    from ..base_models.supported_models import all_presets_with_paths
    from ..data.pair_features import DEFAULT_PAIR_FEATURE_MODE
    from .hybrid_probe import HybridProbe
    from .packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
    from ..console_output import print_message


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
    try:
        from base_models.esmc_sae import ESMC_SAE_COVERAGE, is_sae_model_name, parse_sae_model_name
    except ImportError:
        from ..base_models.esmc_sae import (
            ESMC_SAE_COVERAGE,
            is_sae_model_name,
            parse_sae_model_name,
        )

    if not is_sae_model_name(source_model_name):
        return {}

    selection = parse_sae_model_name(source_model_name)
    return {
        # The packaged model loads the ESM++ backbone and attaches the codebook to it, so
        # base_model_name has to name the backbone rather than the SAE variant.
        "base_model_name": ESMC_SAE_COVERAGE[selection.backbone].model_path,
        "sae_repository": selection.repository,
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
    try:
        from base_models.esmc_sae import build_esmc_sae_model
    except ImportError:
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
    surfaces as a missing sibling file rather than as an import error. The protify package
    copied alongside it supplies the absolute paths.

    Parent imports are rewritten first, so the sibling pattern cannot match the leading
    dot of a `..` import that is already handled.
    """
    source = _PARENT_RELATIVE_IMPORT.sub(
        lambda match: f"from protify.{match.group(1)} import ", source
    )
    return _SIBLING_RELATIVE_IMPORT.sub(
        lambda match: f"from protify.probes.{match.group(1)} import ", source
    )


def _copy_runtime_code(export_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    src_package_dir = repo_root / "src" / "protify"
    dst_package_dir = export_dir / "protify"
    excluded_directories = {
        ".cache",
        "__pycache__",
        "benchmarks",
        "fastplms",
        "logs",
        "scripts",
        "testing_suite",
    }

    for src_file in src_package_dir.rglob("*.py"):
        relative_path = src_file.relative_to(src_package_dir)
        if any(part in excluded_directories for part in relative_path.parts[:-1]):
            continue
        if not src_file.is_file():
            continue
        dst_file = dst_package_dir / relative_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    # SAE pooling is foundry.embedding's arithmetic, and interpnet's feed-forward block is
    # foundry.layers'. Both ship beside protify, so an exported probe pools and runs exactly as
    # training did without the source tree. Only these two ship, so `foundry/__init__.py` is
    # written rather than copied.
    for foundry_module in (foundry.embedding, foundry.layers):
        module_dir = Path(foundry_module.__file__).parent
        dst_module_dir = export_dir / "foundry" / module_dir.name
        dst_module_dir.mkdir(parents=True, exist_ok=True)
        for src_file in module_dir.glob("*.py"):
            shutil.copy2(src_file, dst_module_dir / src_file.name)
    (export_dir / "foundry" / "__init__.py").write_text('"""foundry"""\n', encoding="utf-8")

    # model_components imports its attention, rotary tables, and transformer stack from the
    # interpnet family, so the family's modules ship beside protify as well.
    interpnet_dir = Path(interpnet.__file__).parent
    dst_interpnet_dir = export_dir / "interpnet"
    dst_interpnet_dir.mkdir(parents=True, exist_ok=True)
    for src_file in interpnet_dir.glob("*.py"):
        shutil.copy2(src_file, dst_interpnet_dir / src_file.name)

    packaged_model_file = Path(__file__).with_name("packaged_probe_model.py")
    standalone_source = _standalone_packaged_module_source(
        packaged_model_file.read_text(encoding="utf-8")
    )
    (export_dir / "packaged_probe_model.py").write_text(standalone_source, encoding="utf-8")


def _build_packaged_model(
    trained_model: nn.Module,
    source_model_name: str,
    probe_args: Any,
    embedding_args: Any,
    tokenizer: Any,
    ppi: bool,
    pair_feature_mode: str,
    feature_selection: Any = None,
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
    }
    # The SAE fields redirect base_model_name from the SAE variant to the ESM++ backbone
    # the codebook reads from, so they replace that entry rather than sitting beside it.
    config_fields.update(sae_fields)
    packaged_config = PackagedProbeConfig(**config_fields)
    packaged_model = PackagedProbeModel(config=packaged_config, base_model=base_model, probe=probe_model)
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
    hf_token: str | None = None,
    feature_selection: Any = None,
) -> tuple[bool, str]:
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
        _copy_runtime_code(export_dir)
        readme_path = export_dir / "README.md"
        readme_path.write_text(model_card, encoding="utf-8")

        if hf_token is None:
            api = HfApi()
        else:
            api = HfApi(token=hf_token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            path_in_repo="",
        )

    print_message(f"Packaged model and tokenizer uploaded to Hugging Face Hub: {repo_id}")
    return True, f"Uploaded packaged model to {repo_id}"
