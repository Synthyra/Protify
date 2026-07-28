import torch
from typing import Optional, Tuple


AA_SET: set[str] = set('LAGVSERTIPDKQNFYMHWCXBUOZ*')
CODON_SET: set[str] = set('aA@bB#$%rRnNdDcCeEqQ^G&ghHiIj+MmlJLkK(fFpPoO=szZwSXTtxWyYuvUV]})')
DNA_SET: set[str] = set('ATCG')
RNA_SET: set[str] = set('AUCG')
NONCANONICAL_AMINO_ACIDS: set[str] = set('XBUOZ*')
AMINO_ACID_TO_HUMAN_CODON: dict[str, str] = {
    'A': 'GCC',
    'R': 'CGC',
    'N': 'AAC',
    'D': 'GAC',
    'C': 'TGC',
    'Q': 'CAG',
    'E': 'GAG',
    'G': 'GGC',
    'H': 'CAC',
    'I': 'ATC',
    'L': 'CTG',
    'K': 'AAG',
    'M': 'ATG',
    'F': 'TTC',
    'P': 'CCC',
    'S': 'AGC',
    'T': 'ACC',
    'W': 'TGG',
    'Y': 'TAC',
    'V': 'GTG',
}
NONCANONICAL_ALANINE_CODON = 'GCT'

AA_TO_CODON_TOKEN: dict[str, str] = {
    'A': 'A',
    'R': 'B',
    'N': 'N',
    'D': 'D',
    'C': 'C',
    'Q': 'Q',
    'E': 'E',
    'G': 'G',
    'H': 'H',
    'I': 'I',
    'L': 'L',
    'K': 'K',
    'M': '(',
    'F': 'F',
    'P': 'P',
    'S': 'S',
    'T': 'T',
    'W': 'W',
    'Y': 'Y',
    'V': 'V',
}
CODON_TO_AA: dict[str, str] = {
    'a':'A',
    'A':'A',
    '@':'A',
    'b':'A',
    'B':'R',
    '#':'R',
    '$':'R',
    '%':'R',
    'r':'R',
    'R':'R',
    'n':'N',
    'N':'N',
    'd':'D',
    'D':'D',
    'c':'C',
    'C':'C',
    'e':'E',
    'E':'E',
    'q':'Q',
    'Q':'Q',
    '^':'G',
    'G':'G',
    '&':'G',
    'g':'G',
    'h':'H',
    'H':'H',
    'i':'I',
    'I':'I',
    'j':'I',
    '+':'L',
    'M':'L',
    'm':'L',
    'l':'L',
    'J':'L',
    'L':'L',
    'k':'K',
    'K':'K',
    '(':'M',
    'f':'F',
    'F':'F',
    'p':'P',
    'P':'P',
    'o':'P',
    'O':'P',
    '=':'S',
    's':'S',
    'z':'S',
    'Z':'S',
    'w':'S',
    'S':'S',
    'X':'S',
    'T':'T',
    't':'T',
    'x':'T',
    'W':'T',
    'y':'Y',
    'Y':'Y',
    'u':'V',
    'v':'V',
    'U':'V',
    'V':'V',
    ']':'*',
    '}':'*',
    ')':'*',
}
DNA_CODON_TO_AA: dict[str, str] = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

RNA_CODON_TO_AA: dict[str, str] = {
    codon.replace('T', 'U'): aa for codon, aa in DNA_CODON_TO_AA.items()
}


def pad_and_concatenate_dimer(
    A: torch.Tensor,
    B: torch.Tensor,
    a_mask: Optional[torch.Tensor] = None,
    b_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate each pair of valid sequence embeddings into a padded batch."""
    # A: (b, l_a, d); B: (b, l_b, d)
    batch_size, a_seq_len, hidden_size = A.size()  # b, l_a, d
    if a_mask is None:
        a_mask = torch.ones(batch_size, a_seq_len, device=A.device)  # (b, l_a)
    if b_mask is None:
        # The historical no-mask path assumes l_b == l_a.
        b_mask = torch.ones(batch_size, a_seq_len, device=A.device)  # (b, l_a)

    # a_mask: (b, l_a); b_mask: (b, l_b) when supplied by the caller
    max_len = max(  # l_out = max_i(l_a_i + l_b_i)
        int(a_mask[i].sum().item() + b_mask[i].sum().item())
        for i in range(batch_size)
    )
    combined = torch.zeros(batch_size, max_len, hidden_size, device=A.device)  # (b, l_out, d)
    combined_mask = torch.zeros(batch_size, max_len, device=A.device)  # (b, l_out)

    for i in range(batch_size):
        a_len = int(a_mask[i].sum().item())  # l_a_i
        b_len = int(b_mask[i].sum().item())  # l_b_i
        combined[i, :a_len] = A[i, :a_len]  # (a_len, d) <- (a_len, d)
        combined[i, a_len:a_len + b_len] = B[i, :b_len]  # (b_len, d) <- (b_len, d)
        combined_mask[i, :a_len + b_len] = 1  # (a_len + b_len,)

    return combined, combined_mask  # (b, l_out, d), (b, l_out)
