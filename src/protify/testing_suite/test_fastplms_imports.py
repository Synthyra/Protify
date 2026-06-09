"""Verify that FastPLMs submodule imports resolve correctly after the rehaul."""

import sys
import os
import pytest


def _ensure_fastplms_on_path():
    """Mirror the _FASTPLMS sys.path logic used by base_models/*.py."""
    # From base_models: one level up from testing_suite, then into fastplms
    fastplms_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "fastplms",
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
def _reset_fastplms_import_path():
    _ensure_fastplms_on_path()


def test_fastplms_root_exists():
    assert os.path.isdir(FASTPLMS_ROOT), f"FastPLMs root not found: {FASTPLMS_ROOT}"
    assert os.path.isdir(os.path.join(FASTPLMS_ROOT, "fastplms")), (
        "fastplms/ package directory missing inside submodule root"
    )


def test_import_esm2():
    from fastplms.esm2.modeling_fastesm import (
        FastEsmModel,
        FastEsmForMaskedLM,
        FastEsmForSequenceClassification,
        FastEsmForTokenClassification,
    )
    for cls in (FastEsmModel, FastEsmForMaskedLM, FastEsmForSequenceClassification, FastEsmForTokenClassification):
        assert cls is not None


def test_import_esm_plusplus():
    from fastplms.esm_plusplus.modeling_esm_plusplus import (
        ESMplusplusModel,
        ESMplusplusForMaskedLM,
        ESMplusplusForSequenceClassification,
        ESMplusplusForTokenClassification,
    )
    for cls in (ESMplusplusModel, ESMplusplusForMaskedLM, ESMplusplusForSequenceClassification, ESMplusplusForTokenClassification):
        assert cls is not None


def test_import_dplm():
    from fastplms.dplm.modeling_dplm import (
        DPLMForMaskedLM,
        DPLMForSequenceClassification,
        DPLMForTokenClassification,
    )
    for cls in (DPLMForMaskedLM, DPLMForSequenceClassification, DPLMForTokenClassification):
        assert cls is not None


def test_import_dplm2():
    from fastplms.dplm2.modeling_dplm2 import (
        DPLM2ForMaskedLM,
        DPLM2ForSequenceClassification,
        DPLM2ForTokenClassification,
    )
    for cls in (DPLM2ForMaskedLM, DPLM2ForSequenceClassification, DPLM2ForTokenClassification):
        assert cls is not None


def test_import_e1():
    from fastplms.e1.modeling_e1 import (
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
    for obj in (E1Model, E1ForMaskedLM, E1ForSequenceClassification, E1ForTokenClassification,
                E1BatchPreparer, E1MaskedLMOutputWithPast, DataPrepConfig, get_context, KVCache):
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
    from fastplms.embedding_mixin import Pooler, EmbeddingMixin, ProteinDataset, parse_fasta, build_collator
    for obj in (Pooler, EmbeddingMixin, ProteinDataset, parse_fasta, build_collator):
        assert obj is not None


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
