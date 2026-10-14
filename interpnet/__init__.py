"""InterpNet: attention probes that read protein language model residue embeddings."""

from .interpnet import (
    InterpNetConfig,
    InterpNetForSequenceClassification,
    InterpNetForTokenClassification,
    InterpNetOutput,
)


__all__ = [
    "InterpNetConfig",
    "InterpNetForSequenceClassification",
    "InterpNetForTokenClassification",
    "InterpNetOutput",
]
