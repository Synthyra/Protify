"""SELFIES as text: its symbols, its two serializations, and a cut to a symbol budget.

Nothing here parses a molecule, so this module imports neither RDKit nor selfies. A consumer that
only handles SELFIES strings, such as Atlas cutting a ligand to its encoder's budget, can use it
where neither is installed.
"""

from __future__ import annotations

import re


# One SELFIES symbol, and a whole compact SELFIES string: symbols, with a bare "." between the
# fragments of a salt, as in "[Na+1].[O-1][C]...". Spaces are removed before matching.
SELFIES_SYMBOL = re.compile(r"\[[^\[\]]+\]")
SELFIES_TOKENS = re.compile(rf"^(?:{SELFIES_SYMBOL.pattern})+(?:\.(?:{SELFIES_SYMBOL.pattern})+)*$")


def compact_selfies(selfies: str) -> str:
    """'[C] [C]' -> '[C][C]'."""
    return "".join(selfies.split())


def space_selfies(selfies: str) -> str:
    """'[C][C]' -> '[C] [C]'. A fragment separator stays attached to the symbol after it."""
    return compact_selfies(selfies).replace("][", "] [")


def selfies_symbols(selfies: str) -> list[str]:
    """The symbols of a SELFIES string, spaced or compact, in order and without separators."""
    return SELFIES_SYMBOL.findall(selfies)


def truncate_selfies(selfies: str, max_symbols: int) -> str:
    """The first `max_symbols` symbols of a SELFIES string, cut after a symbol, never inside one.

    An encoder's `max_length` counts tokens, and a SELFIES-TED token is one symbol: 1,024
    characters of spaced SELFIES hold about 170 symbols, and a cut by characters can end inside
    one, as in '[=Bra'. SELFIES decodes any run of symbols, so the cut string still names a
    molecule. It keeps its serialization, and a string within the budget comes back unchanged.
    Raises `ValueError` for a string that is not bracketed symbols or a budget below one.
    """
    if max_symbols < 1:
        raise ValueError(f"max_symbols must be positive, got {max_symbols}")
    if not SELFIES_TOKENS.match(compact_selfies(selfies)):
        raise ValueError(f"not a SELFIES string: {selfies[:80]!r}")
    ends = [match.end() for match in SELFIES_SYMBOL.finditer(selfies)]
    if len(ends) <= max_symbols:
        return selfies
    return selfies[: ends[max_symbols - 1]]
