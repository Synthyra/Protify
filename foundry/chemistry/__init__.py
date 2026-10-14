"""One canonical SMILES and SELFIES string per molecule, for every model that reads ligands.

The SELFIES text names, from `selfies_text`, need neither RDKit nor selfies. Every other name
loads them on first use, and raises `ImportError` there when either is missing or too old, so a
consumer that only counts or cuts SELFIES symbols runs where neither is installed.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from .selfies_text import (
    SELFIES_SYMBOL,
    SELFIES_TOKENS,
    compact_selfies,
    selfies_symbols,
    space_selfies,
    truncate_selfies,
)


if TYPE_CHECKING:
    from .conversion import (
        AmbiguousNotationError,
        InvalidSelfiesError,
        SelfiesRepresentationError,
        build_smiles_to_selfies,
        convert_smiles_to_selfies,
        is_selfies,
        map_smiles_list_to_selfies,
        safe_smiles_to_selfies,
        safe_to_selfies,
        selfies_to_smiles,
        smiles_to_selfies,
        to_selfies,
        to_smiles,
    )
    from .smiles import (
        ChemistryError,
        InvalidSmilesError,
        canonical_kekule_smiles,
        canonicalize_smiles,
        normalize_smiles_for_rdkit,
        parse_smiles,
        standardize_molecule,
    )


# The module that defines each name needing RDKit or selfies.
MODULE_OF = {
    **dict.fromkeys(
        (
            "AmbiguousNotationError",
            "InvalidSelfiesError",
            "SelfiesRepresentationError",
            "build_smiles_to_selfies",
            "convert_smiles_to_selfies",
            "is_selfies",
            "map_smiles_list_to_selfies",
            "safe_smiles_to_selfies",
            "safe_to_selfies",
            "selfies_to_smiles",
            "smiles_to_selfies",
            "to_selfies",
            "to_smiles",
        ),
        ".conversion",
    ),
    **dict.fromkeys(
        (
            "ChemistryError",
            "InvalidSmilesError",
            "canonical_kekule_smiles",
            "canonicalize_smiles",
            "normalize_smiles_for_rdkit",
            "parse_smiles",
            "standardize_molecule",
        ),
        ".smiles",
    ),
}


def __getattr__(name: str) -> object:
    if name in MODULE_OF:
        return getattr(import_module(MODULE_OF[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SELFIES_SYMBOL",
    "SELFIES_TOKENS",
    "AmbiguousNotationError",
    "ChemistryError",
    "InvalidSelfiesError",
    "InvalidSmilesError",
    "SelfiesRepresentationError",
    "build_smiles_to_selfies",
    "canonical_kekule_smiles",
    "canonicalize_smiles",
    "compact_selfies",
    "convert_smiles_to_selfies",
    "is_selfies",
    "map_smiles_list_to_selfies",
    "normalize_smiles_for_rdkit",
    "parse_smiles",
    "safe_smiles_to_selfies",
    "safe_to_selfies",
    "selfies_symbols",
    "selfies_to_smiles",
    "smiles_to_selfies",
    "space_selfies",
    "standardize_molecule",
    "to_selfies",
    "to_smiles",
    "truncate_selfies",
]
