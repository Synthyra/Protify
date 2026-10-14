"""SMILES and SELFIES, translated through the canonical SMILES string.

SELFIES here means the grammar of selfies 2 under that library's default semantic constraints,
with strict encoding. A SELFIES string is written spaced, `[C] [C] [O]`, the form the SELFIES-TED
tokenizers read; a compact `[C][C][O]` is one unknown token to them. A fragment separator stays
attached to the next symbol, as in `[C] [O-1].[Na+1]`, which they read as `[C]`, `[O-1]`,
`.[Na+1]`, so their token count equals `len(selfies_symbols(selfies))`. That text handling is in
`selfies_text`, which needs neither RDKit nor selfies.

`smiles_to_selfies` guarantees the round trip: for every SMILES it accepts,
`selfies_to_smiles(smiles_to_selfies(smiles)) == canonicalize_smiles(smiles)`. A molecule SELFIES
cannot write faithfully raises `SelfiesRepresentationError` rather than coming back as a different
molecule: a wildcard atom, a quadruple bond, an atom over its default bond capacity (such as
hypervalent iodine), and any molecule whose SELFIES decodes to something else.

`to_smiles` and `to_selfies` take either notation and write one string per molecule. `to_smiles`
says how the notation is read, and what happens where the two readings name different molecules.
"""

from __future__ import annotations

import importlib.metadata
import selfies as sf

from tqdm.auto import tqdm

from .selfies_text import SELFIES_TOKENS, compact_selfies, selfies_symbols, space_selfies
from .smiles import ChemistryError, InvalidSmilesError, canonicalize_smiles


# selfies 1 wrote other symbols, such as [Branch1_1], which the SELFIES-TED vocabularies lack.
MINIMUM_SELFIES = (2, 0)
# The constraints are global in selfies. Under any others an encoder accepts different molecules
# and a decoder returns different ones, so every call checks them.
DEFAULT_CONSTRAINTS = sf.get_preset_constraints("default")

if tuple(int(part) for part in importlib.metadata.version("selfies").split(".")[:2]) < MINIMUM_SELFIES:
    raise ImportError(
        f"foundry.chemistry needs selfies 2 or later, whose symbols the SELFIES-TED vocabularies "
        f"use; this environment has selfies {importlib.metadata.version('selfies')}"
    )


class InvalidSelfiesError(ChemistryError):
    """A string that is not SELFIES, or SELFIES that names no molecule."""


class SelfiesRepresentationError(ChemistryError):
    """A molecule that SELFIES cannot write faithfully under the default constraints."""


class AmbiguousNotationError(ChemistryError):
    """One symbol that names one molecule as SMILES and another as SELFIES, such as '[S]'."""


def _require_default_constraints() -> None:
    if sf.get_semantic_constraints() != DEFAULT_CONSTRAINTS:
        raise RuntimeError(
            "selfies' semantic constraints differ from its default preset; foundry.chemistry "
            "encodes and decodes under the default only"
        )


def is_selfies(text: str) -> bool:
    """Whether a string is SELFIES: bracketed symbols only, and the decoder accepts them.

    Containing a bracket is not enough, since SMILES has bracket atoms: '[C@@H]', '[N+]', '[nH]',
    '[Na+]'. Sent to the decoder, those raise. A SMILES made only of bracket atoms that are also
    SELFIES symbols, such as '[C][C]' or '[CH4]', is SELFIES too; `to_smiles` says which reading
    it gets.
    """
    compact = compact_selfies(text)
    if not SELFIES_TOKENS.match(compact):
        return False
    _require_default_constraints()
    try:
        sf.decoder(compact)
    except sf.DecoderError:
        return False
    return True


def selfies_to_smiles(selfies: str) -> str:
    """The canonical SMILES of the molecule a SELFIES string names. Raises `InvalidSelfiesError`."""
    compact = compact_selfies(selfies)
    if not SELFIES_TOKENS.match(compact):
        raise InvalidSelfiesError(f"not a SELFIES string: {selfies!r}")
    _require_default_constraints()
    try:
        decoded = sf.decoder(compact)
    except sf.DecoderError as error:
        raise InvalidSelfiesError(f"the SELFIES decoder rejects {selfies!r}: {error}") from error
    if not decoded:
        raise InvalidSelfiesError(f"the SELFIES names no atoms: {selfies!r}")
    try:
        return canonicalize_smiles(decoded)
    except InvalidSmilesError as error:
        raise InvalidSelfiesError(f"{selfies!r} decodes to {decoded!r}, which is not a molecule: {error}") from error


def smiles_to_selfies(smiles: str) -> str:
    """The spaced SELFIES of the canonical SMILES.

    Raises `InvalidSmilesError` for a string that names no molecule, and
    `SelfiesRepresentationError` for a molecule SELFIES cannot write faithfully.
    """
    canonical = canonicalize_smiles(smiles)
    _require_default_constraints()
    try:
        encoded = sf.encoder(canonical, strict=True)
    except sf.EncoderError as error:
        reason = str(error).splitlines()[0]
        raise SelfiesRepresentationError(f"SELFIES cannot write {canonical}: {reason}") from error
    decoded = sf.decoder(encoded)
    try:
        round_trip = canonicalize_smiles(decoded)
    except InvalidSmilesError as error:
        raise SelfiesRepresentationError(
            f"the SELFIES of {canonical} decodes to {decoded}, which is not a molecule"
        ) from error
    if round_trip != canonical:
        raise SelfiesRepresentationError(f"the SELFIES of {canonical} decodes to a different molecule, {round_trip}")
    return space_selfies(encoded)


def _refuse_an_ambiguous_symbol(selfies: str) -> None:
    """Raise `AmbiguousNotationError` for one symbol whose two readings are different molecules."""
    if len(selfies_symbols(selfies)) != 1:
        return
    try:
        as_smiles = canonicalize_smiles(selfies)
    except InvalidSmilesError:
        return
    as_selfies = selfies_to_smiles(selfies)
    if as_smiles != as_selfies:
        raise AmbiguousNotationError(
            f"{selfies!r} names two molecules: {as_smiles} as SMILES and {as_selfies} as SELFIES. "
            f"Send {smiles_to_selfies(as_smiles)!r} for {as_smiles} or {as_selfies!r} for "
            f"{as_selfies}, neither of which is ambiguous."
        )


def to_smiles(ligand: str) -> str:
    """SMILES or SELFIES in, canonical SMILES out.

    A string of bracketed symbols that the SELFIES decoder accepts is read as SELFIES (see
    `is_selfies`), and anything else as SMILES. Where the two readings name different molecules:

    - One symbol raises `AmbiguousNotationError`. Nine symbols do: '[B]', '[Br]', '[C]', '[F]',
      '[I]', '[N]', '[O]', '[P]', and '[S]'. As SMILES each is a lone atom with no hydrogens, as
      RDKit writes elemental sulfur; as SELFIES each is that atom's hydride, as `smiles_to_selfies`
      writes hydrogen sulfide. The error names both readings and a spelling of each that reads one
      way: '[SH0]' is the sulfur atom in either notation, and 'S' is hydrogen sulfide as SMILES.
    - Two or more symbols keep the SELFIES reading. As SMILES such a string is bracket atoms joined
      by single bonds, at least one of them short of its valence with no hydrogen written: a
      radical, such as dicarbon, '[C][C]'. No drug or compound measured in the README is written so.
    """
    ligand = ligand.strip()
    if is_selfies(ligand):
        _refuse_an_ambiguous_symbol(ligand)
        return selfies_to_smiles(ligand)
    return canonicalize_smiles(ligand)


def to_selfies(ligand: str) -> str:
    """SMILES or SELFIES in, canonical spaced SELFIES out: `smiles_to_selfies(to_smiles(ligand))`.

    Every spelling of a molecule in either notation gives one string, and `to_smiles` says how the
    notation is read. to_selfies(to_selfies(x)) == to_selfies(x) for every molecule but the nine
    hydrides whose SELFIES is a symbol `to_smiles` refuses, such as hydrogen sulfide, '[S]'.
    """
    return smiles_to_selfies(to_smiles(ligand))


def safe_to_selfies(ligand: str) -> str | None:
    """`to_selfies`, or None, with the reason printed, when foundry.chemistry rejects the ligand."""
    try:
        return to_selfies(ligand)
    except ChemistryError as error:
        print(f"Error converting ligand to SELFIES:\n{error}")
        return None


def safe_smiles_to_selfies(smiles: str) -> str | None:
    """`smiles_to_selfies`, or None, with the reason printed, when foundry.chemistry rejects it."""
    try:
        return smiles_to_selfies(smiles)
    except ChemistryError as error:
        print(f"Error converting SMILES to SELFIES:\n{error}")
        return None


def convert_smiles_to_selfies(smiles_list: list[str]) -> tuple[list[str], list[int]]:
    """The SELFIES of every SMILES that converts, and those SMILES' indices."""
    selfies_list: list[str] = []
    valid_indices: list[int] = []
    for index, smiles in enumerate(tqdm(smiles_list, desc="Converting SMILES to SELFIES", unit="SMILES")):
        selfies = safe_smiles_to_selfies(smiles)
        if selfies is not None:
            selfies_list.append(selfies)
            valid_indices.append(index)
    return selfies_list, valid_indices


def build_smiles_to_selfies(smiles_set: set[str]) -> dict[str, str]:
    """SMILES to SELFIES for every SMILES in the set that converts."""
    smiles_list = sorted(smiles_set)
    selfies_list, valid_indices = convert_smiles_to_selfies(smiles_list)
    return {smiles_list[index]: selfies for index, selfies in zip(valid_indices, selfies_list, strict=True)}


def map_smiles_list_to_selfies(smiles_list: list[str], smiles_to_selfies: dict[str, str]) -> tuple[list[str], list[int]]:
    """Look each SMILES up in a mapping built by `build_smiles_to_selfies`; skip the missing ones."""
    selfies_list: list[str] = []
    valid_indices: list[int] = []
    for index, smiles in enumerate(smiles_list):
        if smiles in smiles_to_selfies:
            selfies_list.append(smiles_to_selfies[smiles])
            valid_indices.append(index)
    return selfies_list, valid_indices
