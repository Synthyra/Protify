"""Verify that Protify resolves the FastPLMs 1.0 submodule source contract."""

import os
import sys

import pytest


def _ensure_fastplms_on_path() -> str:
    """Reset FastPLMs imports to the vendored source tree for each contract test."""
    # FastPLMs 1.0 is a source repository rather than an installable distribution.
    fastplms_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fastplms",
        "src",
    )
    if fastplms_root in sys.path:
        sys.path.remove(fastplms_root)
    sys.path.insert(0, fastplms_root)
    if "fastplms" in sys.modules:
        for module_name in list(sys.modules):
            if module_name == "fastplms" or module_name.startswith("fastplms."):
                del sys.modules[module_name]
    return fastplms_root


FASTPLMS_ROOT = _ensure_fastplms_on_path()


@pytest.fixture(autouse=True)
def _reset_fastplms_import_path() -> None:
    _ensure_fastplms_on_path()


def test_fastplms_root_exists():
    assert os.path.isdir(FASTPLMS_ROOT), f"FastPLMs root not found: {FASTPLMS_ROOT}"
    assert os.path.isdir(os.path.join(FASTPLMS_ROOT, "fastplms")), (
        "fastplms/ package directory missing inside submodule root"
    )
    assert os.path.isfile(os.path.join(FASTPLMS_ROOT, "fastplms", "models.toml"))


def test_fastplms_version_and_registry():
    import fastplms
    from fastplms import get_model_registry

    assert fastplms.__version__ == "1.0.0"
    registry = get_model_registry()
    assert registry.families["dplm2"].attention == ("sdpa",)


def test_import_esm2():
    from fastplms.models.esm2.modeling_fastesm import (
        FastEsmModel,
        FastEsmForMaskedLM,
        FastEsmForSequenceClassification,
        FastEsmForTokenClassification,
    )
    for cls in (
        FastEsmModel,
        FastEsmForMaskedLM,
        FastEsmForSequenceClassification,
        FastEsmForTokenClassification,
    ):
        assert cls is not None


def test_import_esm_plusplus():
    from fastplms.models.esm_plusplus.modeling_esm_plusplus import (
        ESMplusplusModel,
        ESMplusplusForMaskedLM,
        ESMplusplusForSequenceClassification,
        ESMplusplusForTokenClassification,
    )
    for cls in (
        ESMplusplusModel,
        ESMplusplusForMaskedLM,
        ESMplusplusForSequenceClassification,
        ESMplusplusForTokenClassification,
    ):
        assert cls is not None


def test_import_dplm():
    from fastplms.models.dplm.modeling_dplm import (
        DPLMForMaskedLM,
        DPLMForSequenceClassification,
        DPLMForTokenClassification,
    )
    for cls in (DPLMForMaskedLM, DPLMForSequenceClassification, DPLMForTokenClassification):
        assert cls is not None


def test_import_dplm2():
    from fastplms.models.dplm2.modeling_dplm2 import (
        DPLM2ForMaskedLM,
        DPLM2ForSequenceClassification,
        DPLM2ForTokenClassification,
    )
    for cls in (DPLM2ForMaskedLM, DPLM2ForSequenceClassification, DPLM2ForTokenClassification):
        assert cls is not None


def test_import_e1():
    from fastplms.models.e1.modeling_e1 import (
        E1Model,
        E1ForMaskedLM,
        E1ForSequenceClassification,
        E1ForTokenClassification,
        E1BatchPreparer,
        E1MaskedLMOutputWithPast,
        DataPrepConfig,
        get_context,
        KVCache,
    )
    for obj in (
        E1Model,
        E1ForMaskedLM,
        E1ForSequenceClassification,
        E1ForTokenClassification,
        E1BatchPreparer,
        E1MaskedLMOutputWithPast,
        DataPrepConfig,
        get_context,
        KVCache,
    ):
        assert obj is not None


def test_import_attention():
    from fastplms.attention import (
        AttentionBackend,
        VALID_ATTENTION_BACKENDS,
        resolve_attention_backend,
    )
    assert AttentionBackend is not None
    assert isinstance(VALID_ATTENTION_BACKENDS, (list, tuple))
    assert callable(resolve_attention_backend)


def test_import_embedding_mixin():
    from fastplms.embeddings import (
        EmbeddingMixin,
        EmbeddingResult,
        Pooler,
        embed_dataset,
        parse_fasta,
    )
    for obj in (Pooler, EmbeddingMixin, EmbeddingResult, embed_dataset, parse_fasta):
        assert obj is not None


def test_import_dplm2_tokenizer():
    from fastplms.models.dplm2.tokenization_dplm2 import DPLM2Tokenizer

    assert DPLM2Tokenizer is not None


def test_base_models_import_esm2():
    try:
        from src.protify.base_models.esm2 import FastEsmForMaskedLM, build_esm2_model
    except ImportError:
        try:
            from protify.base_models.esm2 import FastEsmForMaskedLM, build_esm2_model
        except ImportError:
            from base_models.esm2 import FastEsmForMaskedLM, build_esm2_model
    assert FastEsmForMaskedLM is not None
    assert callable(build_esm2_model)


def test_base_models_import_esmc():
    try:
        from src.protify.base_models.esmc import ESMplusplusForMaskedLM, build_esmc_model
    except ImportError:
        try:
            from protify.base_models.esmc import ESMplusplusForMaskedLM, build_esmc_model
        except ImportError:
            from base_models.esmc import ESMplusplusForMaskedLM, build_esmc_model
    assert ESMplusplusForMaskedLM is not None
    assert callable(build_esmc_model)


def test_base_models_import_dplm():
    try:
        from src.protify.base_models.dplm import DPLMForMaskedLM, build_dplm_model
    except ImportError:
        try:
            from protify.base_models.dplm import DPLMForMaskedLM, build_dplm_model
        except ImportError:
            from base_models.dplm import DPLMForMaskedLM, build_dplm_model
    assert DPLMForMaskedLM is not None
    assert callable(build_dplm_model)


def test_base_models_import_dplm2():
    try:
        from src.protify.base_models.dplm2 import DPLM2ForMaskedLM, build_dplm2_model
    except ImportError:
        try:
            from protify.base_models.dplm2 import DPLM2ForMaskedLM, build_dplm2_model
        except ImportError:
            from base_models.dplm2 import DPLM2ForMaskedLM, build_dplm2_model
    assert DPLM2ForMaskedLM is not None
    assert callable(build_dplm2_model)


def test_base_models_import_e1():
    try:
        from src.protify.base_models.e1 import E1ForMaskedLM, build_e1_model
    except ImportError:
        try:
            from protify.base_models.e1 import E1ForMaskedLM, build_e1_model
        except ImportError:
            from base_models.e1 import E1ForMaskedLM, build_e1_model
    assert E1ForMaskedLM is not None
    assert callable(build_e1_model)
