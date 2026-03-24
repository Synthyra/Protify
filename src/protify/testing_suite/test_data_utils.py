"""Tests for data/utils.py constants and translation mappings."""

import torch
import pytest

try:
    from src.protify.data.utils import (
        AA_SET, DNA_SET, RNA_SET, CODON_SET, NONCANONICAL_AMINO_ACIDS,
        AMINO_ACID_TO_HUMAN_CODON, AA_TO_CODON_TOKEN, CODON_TO_AA,
        DNA_CODON_TO_AA, RNA_CODON_TO_AA, pad_and_concatenate_dimer,
    )
except ImportError:
    try:
        from protify.data.utils import (
            AA_SET, DNA_SET, RNA_SET, CODON_SET, NONCANONICAL_AMINO_ACIDS,
            AMINO_ACID_TO_HUMAN_CODON, AA_TO_CODON_TOKEN, CODON_TO_AA,
            DNA_CODON_TO_AA, RNA_CODON_TO_AA, pad_and_concatenate_dimer,
        )
    except ImportError:
        from ..data.utils import (
            AA_SET, DNA_SET, RNA_SET, CODON_SET, NONCANONICAL_AMINO_ACIDS,
            AMINO_ACID_TO_HUMAN_CODON, AA_TO_CODON_TOKEN, CODON_TO_AA,
            DNA_CODON_TO_AA, RNA_CODON_TO_AA, pad_and_concatenate_dimer,
        )


CANONICAL_20 = set("ACDEFGHIKLMNPQRSTVWY")


def test_aa_set_contains_all_canonical() -> None:
    assert CANONICAL_20.issubset(AA_SET)


def test_aa_set_contains_noncanonical() -> None:
    for aa in NONCANONICAL_AMINO_ACIDS:
        assert aa in AA_SET


def test_dna_set_exact() -> None:
    assert DNA_SET == {"A", "T", "C", "G"}


def test_rna_set_exact() -> None:
    assert RNA_SET == {"A", "U", "C", "G"}


def test_codon_set_matches_codon_to_aa_keys() -> None:
    assert set(CODON_TO_AA.keys()) == CODON_SET


def test_amino_acid_to_human_codon_covers_canonical() -> None:
    assert set(AMINO_ACID_TO_HUMAN_CODON.keys()) == CANONICAL_20


def test_amino_acid_to_human_codon_values_are_dna() -> None:
    for aa, codon in AMINO_ACID_TO_HUMAN_CODON.items():
        assert len(codon) == 3, f"Codon for {aa} is not 3 chars: {codon}"
        for base in codon:
            assert base in DNA_SET, f"Non-DNA base '{base}' in codon for {aa}"


def test_aa_to_codon_token_covers_canonical() -> None:
    assert set(AA_TO_CODON_TOKEN.keys()) == CANONICAL_20


def test_codon_to_aa_values_in_aa_set() -> None:
    for codon, aa in CODON_TO_AA.items():
        assert aa in AA_SET, f"Codon '{codon}' maps to '{aa}' not in AA_SET"


def test_dna_codon_to_aa_has_64_entries() -> None:
    assert len(DNA_CODON_TO_AA) == 64


def test_rna_codon_to_aa_has_64_entries() -> None:
    assert len(RNA_CODON_TO_AA) == 64


def test_rna_codon_to_aa_is_t_to_u_of_dna() -> None:
    for dna_codon, aa in DNA_CODON_TO_AA.items():
        rna_codon = dna_codon.replace("T", "U")
        assert rna_codon in RNA_CODON_TO_AA
        assert RNA_CODON_TO_AA[rna_codon] == aa


def test_pad_and_concatenate_dimer_shapes() -> None:
    torch.manual_seed(0)
    batch, L, d = 2, 4, 8
    A = torch.randn(batch, L, d)
    B = torch.randn(batch, L, d)
    a_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float)
    b_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float)
    combined, combined_mask = pad_and_concatenate_dimer(A, B, a_mask, b_mask)
    # Max combined length: max(3+2, 2+3) = 5
    assert combined.shape == (2, 5, d)
    assert combined_mask.shape == (2, 5)
    # First sample: 3+2 = 5 valid tokens
    assert combined_mask[0].sum().item() == 5
    # Second sample: 2+3 = 5 valid tokens
    assert combined_mask[1].sum().item() == 5


def test_pad_and_concatenate_dimer_no_masks() -> None:
    torch.manual_seed(0)
    batch, L, d = 2, 3, 4
    A = torch.randn(batch, L, d)
    B = torch.randn(batch, L, d)
    combined, combined_mask = pad_and_concatenate_dimer(A, B)
    # All tokens valid: max length = L + L = 6
    assert combined.shape == (2, 6, d)
    assert combined_mask.sum().item() == 12
