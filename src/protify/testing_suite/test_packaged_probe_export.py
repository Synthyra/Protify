import gc
import json
import os
import subprocess
import sys
import tempfile
import pytest
import torch

from pathlib import Path
from transformers import AutoModel, BertConfig, BertModel, BertTokenizerFast

from foundry.checkpoint_loading import uninitialized_memory_as_nan
from protify.probes.export_packaged_model import _write_packaged_module
from protify.probes.get_probe import ProbeArguments, get_probe
from protify.probes.mlp_probe import MLPProbe, MLPProbeConfig
from protify.probes.packaged_probe_model import PackagedProbeConfig, PackagedProbeModel
from protify.probes.transformer_probe import TransformerForSequenceClassification, TransformerProbeConfig


def _create_tiny_backbone(backbone_dir: Path) -> tuple[BertModel, BertTokenizerFast]:
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "B", "C", "D"]
    vocab_path = backbone_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
    )
    model = BertModel(config).eval()
    model.save_pretrained(str(backbone_dir))
    tokenizer.save_pretrained(str(backbone_dir))
    return model, tokenizer


def _save_and_load_with_automodel(
    packaged_model: PackagedProbeModel,
    tokenizer: BertTokenizerFast,
    model_dir: Path,
) -> AutoModel:
    packaged_model.config.auto_map = {
        "AutoConfig": "packaged_probe_model.PackagedProbeConfig",
        "AutoModel": "packaged_probe_model.PackagedProbeModel",
    }
    packaged_model.config.architectures = ["PackagedProbeModel"]
    packaged_model.save_pretrained(str(model_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(model_dir))
    _write_packaged_module(model_dir)
    return AutoModel.from_pretrained(str(model_dir), trust_remote_code=True)


def test_linear_packaged_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix="protify_linear_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "linear_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = MLPProbeConfig(
            input_size=16,
            hidden_size=32,
            dropout=0.1,
            num_labels=3,
            n_layers=1,
            task_type="singlelabel",
        )
        probe = MLPProbe(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="mlp",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=False,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=3,
            ppi=False,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C A", "B C D A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        outputs = loaded_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )  # logits: (b=2, c=3)
        assert outputs.logits.shape == (2, 3), f"Unexpected linear packaged logits shape: {outputs.logits.shape}"
        del loaded_model
        gc.collect()


def test_transformer_packaged_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix="protify_transformer_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "transformer_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = TransformerProbeConfig(
            input_size=16,
            hidden_size=16,
            classifier_size=24,
            transformer_dropout=0.1,
            classifier_dropout=0.1,
            num_labels=2,
            n_layers=1,
            head_size=8,
            task_type="singlelabel",
            rotary=False,
            attention_backend="sdpa",
            output_s_max=False,
            pre_ln=True,
            probe_pooling_types=["mean"],
            use_bias=False,
            add_token_ids=False,
        )
        probe = TransformerForSequenceClassification(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="transformer",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=True,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=2,
            ppi=False,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C D", "D C B A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        outputs = loaded_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )  # logits: (b=2, c=2)
        assert outputs.logits.shape == (2, 2), f"Unexpected transformer packaged logits shape: {outputs.logits.shape}"
        del loaded_model
        gc.collect()


def test_rotary_transformer_packaged_roundtrip_reproduces_the_trained_probe() -> None:
    """A rotary probe's tables are in no checkpoint, so loading the package has to rebuild them.

    The package is saved and loaded with every new allocation filled with NaN, so a table the
    load leaves as allocated turns the logits NaN instead of matching by chance.
    """
    with tempfile.TemporaryDirectory(prefix="protify_rotary_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "rotary_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = TransformerProbeConfig(
            input_size=16,
            hidden_size=16,
            classifier_size=24,
            transformer_dropout=0.0,
            classifier_dropout=0.0,
            num_labels=2,
            n_layers=1,
            head_size=8,
            task_type="singlelabel",
            rotary=True,
            attention_backend="sdpa",
            output_s_max=False,
            pre_ln=True,
            probe_pooling_types=["mean"],
            use_bias=False,
            add_token_ids=False,
        )
        torch.manual_seed(0)
        probe = TransformerForSequenceClassification(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="transformer",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=True,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=2,
            ppi=False,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        with uninitialized_memory_as_nan():
            loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C D", "D C B A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        with torch.inference_mode():
            expected = packaged_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits  # (b=2, c=2)
            actual = loaded_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits  # (b=2, c=2)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del loaded_model
        gc.collect()


def test_ppi_packaged_inference_with_and_without_token_type_ids() -> None:
    with tempfile.TemporaryDirectory(prefix="protify_ppi_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "ppi_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = MLPProbeConfig(
            input_size=32,
            hidden_size=24,
            dropout=0.1,
            num_labels=2,
            n_layers=1,
            task_type="singlelabel",
        )
        probe = MLPProbe(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="mlp",
            probe_config=probe.config.to_dict(),
            tokenwise=False,
            matrix_embed=False,
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=2,
            ppi=True,
            add_token_ids=False,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        pair_batch = tokenizer(
            ["A B C", "B C D"],
            ["D C B", "A C B"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l_pair=9)

        outputs_with_token_types = loaded_model(
            input_ids=pair_batch["input_ids"],
            attention_mask=pair_batch["attention_mask"],
            token_type_ids=pair_batch["token_type_ids"],
        )  # logits: (b=2, c=2)
        assert outputs_with_token_types.logits.shape == (2, 2), "PPI logits shape mismatch with token_type_ids"

        outputs_without_token_types = loaded_model(
            input_ids=pair_batch["input_ids"],
            attention_mask=pair_batch["attention_mask"],
        )  # logits: (b=2, c=2)
        assert outputs_without_token_types.logits.shape == (2, 2), "PPI logits shape mismatch without token_type_ids"
        del loaded_model
        gc.collect()


def test_packaged_probe_keeps_a_label_count_other_than_two() -> None:
    """Transformers 5 saves a config's label count only as `id2label`, and the packaged config
    overwrote it with its default of two, so a reloaded package reported two labels."""
    with tempfile.TemporaryDirectory(prefix="protify_labels_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "labels_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        probe_config = TransformerProbeConfig(
            input_size=16,
            hidden_size=16,
            classifier_size=24,
            transformer_dropout=0.0,
            classifier_dropout=0.0,
            num_labels=5,
            n_layers=1,
            head_size=8,
            rotary=False,
            attention_backend="sdpa",
            probe_pooling_types=["mean"],
        )
        torch.manual_seed(0)
        probe = TransformerForSequenceClassification(probe_config).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="transformer",
            probe_config=probe.config.to_dict(),
            matrix_embed=True,
            pooling_types=["mean"],
            num_labels=5,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C D", "D C B A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        with torch.inference_mode():
            expected = packaged_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits  # (b=2, c=5)
            actual = loaded_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits  # (b=2, c=5)
        assert loaded_model.config.num_labels == 5
        assert loaded_model.probe.config.num_labels == 5
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del loaded_model
        gc.collect()


def main() -> None:
    torch.manual_seed(0)
    test_linear_packaged_roundtrip()
    test_transformer_packaged_roundtrip()
    test_rotary_transformer_packaged_roundtrip_reproduces_the_trained_probe()
    test_packaged_probe_keeps_a_label_count_other_than_two()
    test_ppi_packaged_inference_with_and_without_token_type_ids()
    print("Packaged probe model smoke tests passed.")


if __name__ == "__main__":
    main()


def test_feature_selection_roundtrip() -> None:
    """A probe trained on selected columns must read the same columns after a reload.

    The index array lives only in the config, so this is what catches it failing to
    serialize: without it the reloaded probe would read a full-width embedding and either
    crash on the width or, worse, be fed whatever columns happened to line up.
    """
    kept_columns = [1, 3, 5, 7, 9, 11, 13, 15]
    with tempfile.TemporaryDirectory(prefix="protify_selection_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "selection_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        # The probe reads the selected width, not the backbone's 16.
        probe = MLPProbe(
            MLPProbeConfig(
                input_size=len(kept_columns),
                hidden_size=32,
                dropout=0.0,
                num_labels=3,
                n_layers=1,
                task_type="singlelabel",
            )
        ).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type="mlp",
            probe_config=probe.config.to_dict(),
            pooling_types=["mean"],
            task_type="singlelabel",
            num_labels=3,
            sep_token_id=tokenizer.sep_token_id,
            feature_indices=kept_columns,
            feature_input_size=16,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        batch = tokenizer(["A B C A", "B C D A"], padding="longest", return_tensors="pt")

        with torch.no_grad():
            before = packaged_model(**batch).logits  # (2, 3)
        loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)
        with torch.no_grad():
            after = loaded_model(**batch).logits  # (2, 3)

        assert loaded_model.config.feature_indices == kept_columns
        assert loaded_model.config.feature_input_size == 16
        assert torch.allclose(before, after, atol=1e-5)

        del loaded_model
        gc.collect()


@pytest.mark.parametrize(("probe_type", "pooling_types"), [("transformer", ["max", "var"]), ("lyra", ["max"])])
def test_packaged_probe_keeps_its_pooling(probe_type: str, pooling_types: list[str]) -> None:
    """A probe built with pooling other than its default reloads through AutoModel with that pooling.

    The packaged model rebuilds its probe from the saved probe config, so the pooling a run asks
    for has to survive get_probe, that config, and the reload.
    """
    with tempfile.TemporaryDirectory(prefix="protify_pooling_packaged_test_", ignore_cleanup_errors=True) as temp_dir:
        temp_path = Path(temp_dir)
        backbone_dir = temp_path / "backbone"
        model_dir = temp_path / "pooling_packaged_model"
        backbone_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        backbone, tokenizer = _create_tiny_backbone(backbone_dir)
        torch.manual_seed(0)
        probe = get_probe(
            ProbeArguments(
                probe_type=probe_type,
                input_size=16,
                hidden_size=16,
                classifier_size=24,
                head_size=8,
                num_labels=2,
                dropout=0.0,
                transformer_dropout=0.0,
                classifier_dropout=0.0,
                rotary=False,
                attention_backend="sdpa",
                probe_pooling_types=pooling_types,
            )
        ).eval()
        packaged_config = PackagedProbeConfig(
            base_model_name=str(backbone_dir),
            probe_type=probe_type,
            probe_config=probe.config.to_dict(),
            matrix_embed=True,
            pooling_types=["mean"],
            num_labels=2,
            sep_token_id=tokenizer.sep_token_id,
        )
        packaged_model = PackagedProbeModel(config=packaged_config, base_model=backbone, probe=probe).eval()
        with uninitialized_memory_as_nan():
            loaded_model = _save_and_load_with_automodel(packaged_model, tokenizer, model_dir)

        batch = tokenizer(
            ["A B C D", "D C B A"],
            padding="longest",
            return_tensors="pt",
        )  # input_ids, attention_mask, and token_type_ids: (b=2, l=6)
        with torch.inference_mode():
            expected = packaged_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits  # (b=2, c=2)
            actual = loaded_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits  # (b=2, c=2)
        assert loaded_model.probe.config.pooling_types == pooling_types
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        del loaded_model
        gc.collect()


def test_the_export_writes_one_module_that_imports_protify_by_name(tmp_path: Path) -> None:
    """A repository carries no copy of Protify: its module imports the installed package by absolute name."""
    _write_packaged_module(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["packaged_probe_model.py"]
    source = (tmp_path / "packaged_probe_model.py").read_text(encoding="utf-8")
    import_lines = [line for line in source.splitlines() if line.startswith(("from ", "import "))]
    assert not [line for line in import_lines if line.startswith("from .")]
    assert "from protify.probes.get_probe import rebuild_probe_from_saved_config" in import_lines
    assert "from protify.version import PROTIFY_VERSION" in import_lines
    assert not [line for line in import_lines if "interpnet" in line or "foundry" in line]


# A child whose path holds this checkout's packages but not the export, as an installed Protify would be.
_LOAD_EXPORT_SCRIPT = """
import json, sys, torch
from transformers import AutoModel
model = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True).eval()
input_ids = torch.tensor([[2, 5, 6, 7, 3], [2, 6, 7, 8, 3]])
with torch.no_grad():
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits
print(json.dumps({"module": type(model).__module__, "logits": logits.tolist()}))
"""


def test_an_exported_repository_loads_from_the_installed_package_and_standardizes(tmp_path: Path) -> None:
    """The training standardizer travels in the checkpoint, and a fresh process reproduces the logits."""
    backbone_dir = tmp_path / "backbone"
    model_dir = tmp_path / "packaged"
    backbone_dir.mkdir()
    model_dir.mkdir()
    backbone, tokenizer = _create_tiny_backbone(backbone_dir)
    torch.manual_seed(0)
    probe = MLPProbe(MLPProbeConfig(input_size=16, hidden_size=32, num_labels=2, n_layers=1, task_type="singlelabel")).eval()
    mean = torch.linspace(-1.0, 1.0, 16)  # (d,)
    scale = torch.linspace(0.5, 2.0, 16)  # (d,)
    config = PackagedProbeConfig(
        base_model_name=str(backbone_dir), probe_type="mlp", probe_config=probe.config.to_dict(), pooling_types=["mean"],
        task_type="singlelabel", num_labels=2, sep_token_id=tokenizer.sep_token_id, standardized_width=16,
        protify_version="0.1.0",
    )
    packaged = PackagedProbeModel(config=config, base_model=backbone, probe=probe, standardizer=(mean, scale)).eval()
    input_ids = torch.tensor([[2, 5, 6, 7, 3], [2, 6, 7, 8, 3]])  # (b, l)
    attention_mask = torch.ones_like(input_ids)  # (b, l)
    with torch.no_grad():
        pooled = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)  # (b, d)
        expected = probe(embeddings=(pooled - mean) / scale).logits  # (b, c)
        torch.testing.assert_close(packaged(input_ids=input_ids, attention_mask=attention_mask).logits, expected)

    _save_and_load_with_automodel(packaged, tokenizer, model_dir)
    # The child's path is this checkout's package roots, as an installed Protify's would be.
    import foundry
    import protify

    package_roots = {str(Path(protify.__file__).resolve().parents[1]), str(Path(foundry.__file__).resolve().parents[1])}
    child = subprocess.run(
        [sys.executable, "-c", _LOAD_EXPORT_SCRIPT, str(model_dir)],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": os.pathsep.join(sorted(package_roots))},
        capture_output=True, text=True, check=True,
    )
    reported = json.loads(child.stdout.strip().splitlines()[-1])

    assert reported["module"].startswith("transformers_modules")
    torch.testing.assert_close(torch.tensor(reported["logits"]), expected, rtol=1e-5, atol=1e-5)
