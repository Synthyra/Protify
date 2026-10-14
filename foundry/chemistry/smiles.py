"""One canonical SMILES string per molecule: a standardization policy, then one way of writing it.

`canonicalize_smiles` runs these steps, in this order:

1. Text. Surrounding whitespace is removed. Whitespace inside is rejected: RDKit reads a space as
   the end of the SMILES and would silently drop the rest, so "CC O" would become ethane.
2. Parsing. RDKit parses and sanitizes the string, then hydrogens written as atoms are removed
   (see `parse_smiles`). If RDKit rejects the string, and a spelling in `RDKIT_REWRITES` occurs in
   it, the rewritten string is tried once. A string RDKit cannot parse, a molecule it cannot
   sanitize (a pentavalent carbon, an aromatic ring it cannot kekulize), and a molecule with no
   atoms are rejected.
3. Atom map numbers are removed. They label atoms for reactions and are not part of the molecule.
4. Cleanup, `rdMolStandardize.Cleanup` with its defaults: bonds to metals are broken into ions,
   functional groups are normalized (a nitro group is written charge separated), and charges are
   moved to the most acidic sites.
5. Charges, `rdMolStandardize.Uncharger`: acids and bases are neutralized where the net charge
   allows. A counterion keeps its partner charged and a quaternary nitrogen keeps its charge, so
   glycine written as a zwitterion is neutralized and betaine is not.
6. Stereo. Tetrahedral centers and double-bond geometry are kept as given, and unspecified stereo
   stays unspecified. Any other chiral tag, such as square planar or allene, is removed: RDKit does
   not canonicalize those independently of atom order, and SELFIES cannot write them.
7. Writing. The molecule is written as canonical isomeric Kekulé SMILES.

Steps 2 to 7 then run again on that string until it stops changing, so a canonical string
canonicalizes to itself. One pass settles every FDA-approved drug and every ExpansionRx compound;
a few unusual charged molecules need up to three more, and one that needs more than
`MAXIMUM_PASSES` is rejected.

Fragments, isotopes, and tautomers are kept as given: salts are not stripped, deuterium stays, and
two tautomers of one compound give two strings.

A rejected string raises `InvalidSmilesError`, a `ValueError` whose message says why. So does a
molecule that steps 4 to 6 leave RDKit unable to sanitize, such as B#[P-]#B, whose phosphorus
the uncharger gives too high a valence. Nothing returns None, and nothing falls back to a
different reading of the input.

The string is canonical only on RDKit 2026.03 or later. From that release, `Kekulize()` no longer
depends on atom and bond order, so a Kekulé SMILES places its double bonds the same way whichever
spelling the molecule came from. Under RDKit 2025.9.6 the same writer gave 1,522 of 2,838
FDA-approved drugs more than one string, so importing this module on an earlier RDKit raises.
"""

from __future__ import annotations

from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import MolSanitizeException


# The first RDKit release whose Kekulé structures do not depend on atom and bond order.
MINIMUM_RDKIT = (2026, 3)
# Standardization passes after the first, at most, before a molecule whose string keeps changing
# is rejected. Random molecules from SELFIES' alphabet needed up to three.
MAXIMUM_PASSES = 5
# Tetrahedral stereo is kept. Any other chiral tag, such as allene or square planar, is dropped.
KEPT_CHIRAL_TAGS = frozenset(
    {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }
)
# Invalid spellings found in source data, each rewritten to the molecule it was meant to be. They
# apply only to a string RDKit rejects as written, in this order; a later rewrite also applies to
# what an earlier one wrote.
RDKIT_REWRITES = (
    ("[PH2](=O)(O)O", "P(=O)(O)O"),
    ("[PH](=O)(=O)O", "P(=O)(=O)O"),
    ("P(=O)(=O)O", "P(=O)(O)O"),
    ("[N+2](=O)=O", "[N+](=O)[O-]"),
    ("[S+](=N)(=N)=O", "S(=N)(=N)=O"),
    ("S(=N)(=N)=O", "S(=O)(=N)N"),
)

if tuple(int(part) for part in rdBase.rdkitVersion.split(".")[:2]) < MINIMUM_RDKIT:
    raise ImportError(
        f"foundry.chemistry needs RDKit 2026.03 or later, whose Kekulé structures do not depend on "
        f"atom order; this environment has RDKit {rdBase.rdkitVersion}"
    )


class ChemistryError(ValueError):
    """A ligand string foundry.chemistry rejects. The message says why."""


class InvalidSmilesError(ChemistryError):
    """A SMILES string that names no molecule RDKit can parse and sanitize."""


def normalize_smiles_for_rdkit(smiles: str) -> str:
    """Apply every rewrite in `RDKIT_REWRITES`, in order."""
    for before, after in RDKIT_REWRITES:
        smiles = smiles.replace(before, after)
    return smiles


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse and sanitize one SMILES string exactly as written, or raise `InvalidSmilesError`.

    Hydrogens written as atoms are removed after parsing rather than during it, and the bond
    directions around each double bond are then set again from its stereo. RDKit's parser, removing
    them itself, reads a ring double bond whose direction is written on a hydrogen as the other
    isomer: "[H]/C1=C/CCCCCC1" is cis-cyclooctene and it reads trans.
    """
    if not smiles or any(character.isspace() for character in smiles):
        raise InvalidSmilesError(f"a SMILES string must be non-empty and hold no whitespace: {smiles!r}")
    params = Chem.SmilesParserParams()
    params.removeHs = False
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles, params)
        if molecule is None:
            params.sanitize = False
            unsanitized = Chem.MolFromSmiles(smiles, params)
            if unsanitized is None:
                raise InvalidSmilesError(f"RDKit failed to parse SMILES: {smiles}")
            problems = "; ".join(problem.Message() for problem in Chem.DetectChemistryProblems(unsanitized))
            raise InvalidSmilesError(f"RDKit failed to sanitize SMILES {smiles}: {problems}")
        molecule = Chem.RemoveHs(molecule)
    if molecule.GetNumAtoms() == 0:
        raise InvalidSmilesError(f"the SMILES names no atoms: {smiles!r}")
    # RemoveHs moves a hydrogen's bond direction onto a ring closure bond with the wrong sense, and
    # the writer would copy it. Directions set from the stereo are right.
    for bond in molecule.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(molecule)
    return molecule


def _parse_with_rewrites(smiles: str) -> Chem.Mol:
    """Parse as written; only a string RDKit rejects gets the rewrites, and one more try."""
    try:
        return parse_smiles(smiles)
    except InvalidSmilesError as error:
        rewritten = normalize_smiles_for_rdkit(smiles)
        if rewritten == smiles:
            raise
        try:
            return parse_smiles(rewritten)
        except InvalidSmilesError:
            raise error from None


def _standardize_once(smiles: str) -> Chem.Mol:
    molecule = _parse_with_rewrites(smiles.strip())
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        with rdBase.BlockLogs():
            molecule = rdMolStandardize.Cleanup(molecule)
            molecule = rdMolStandardize.Uncharger().uncharge(molecule)
            for atom in molecule.GetAtoms():
                if atom.GetChiralTag() not in KEPT_CHIRAL_TAGS:
                    atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            Chem.SanitizeMol(molecule)
    except MolSanitizeException as error:
        raise InvalidSmilesError(f"RDKit's standardization rejects SMILES {smiles.strip()}: {error}") from error
    return molecule


def standardize_molecule(smiles: str) -> Chem.Mol:
    """Parse one SMILES string and apply steps 1 to 6 of the policy above until its string settles.

    Each further pass starts from the canonical string of the last one's molecule. Raises
    `InvalidSmilesError`.
    """
    molecule = _standardize_once(smiles)
    written = canonical_kekule_smiles(molecule)
    for _ in range(MAXIMUM_PASSES):
        molecule = _standardize_once(written)
        rewritten = canonical_kekule_smiles(molecule)
        if rewritten == written:
            return molecule
        written = rewritten
    raise InvalidSmilesError(f"the standardization of SMILES {smiles.strip()} does not settle in {MAXIMUM_PASSES} more passes")


def canonical_kekule_smiles(molecule: Chem.Mol) -> str:
    """Canonical isomeric Kekulé SMILES, the one string every model input is written as."""
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True, kekuleSmiles=True)


def canonicalize_smiles(smiles: str) -> str:
    """The canonical SMILES of the standardized molecule. Raises `InvalidSmilesError`."""
    return canonical_kekule_smiles(standardize_molecule(smiles))
