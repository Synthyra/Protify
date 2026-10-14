---
id: foundry.chemistry
kind: library_module
status: active
aliases: [foundry.chemistry]
---
# foundry.chemistry

One string per molecule, for every model that reads a ligand as text. A ligand model scores
the string, not the molecule: Atlas-PLI gives the same drug, spelled two ways, scores that
differ by a median 0.03 and by up to 0.6 on one protein. So every path that hands a ligand to a
model writes it with `canonicalize_smiles`, and SELFIES come from that string.

This module is the workspace's one implementation of canonical SMILES and of translation between
SMILES and SELFIES. A consumer that repeats a piece of it has a test that fails when the copy
drifts; see the consumers table below.

| Name | Does |
|---|---|
| `canonicalize_smiles(smiles)` | The standardization policy below, then `canonical_kekule_smiles` |
| `standardize_molecule(smiles)` | The policy, as an RDKit molecule |
| `parse_smiles(smiles)` | Parse and sanitize as written, hydrogens removed; no standardization |
| `canonical_kekule_smiles(molecule)` | Canonical isomeric Kekulé SMILES, the one string every model input is written as |
| `smiles_to_selfies`, `selfies_to_smiles` | Conversions through the canonical SMILES, with the round trip checked |
| `to_selfies(ligand)`, `to_smiles(ligand)` | Either notation in, one canonical string per molecule out. `to_selfies(x)` is `smiles_to_selfies(to_smiles(x))`. One symbol that names two molecules raises (see SELFIES) |
| `is_selfies(text)` | True only for bracketed symbols that the decoder accepts, so `[C@@H]` or `[Na+]` in a SMILES is not SELFIES |
| `compact_selfies`, `space_selfies`, `selfies_symbols` | The two serializations of one SELFIES string, and its symbols |
| `truncate_selfies(selfies, max_symbols)` | The first `max_symbols` symbols, cut after a whole symbol: the cut to an encoder's token budget |
| `SELFIES_SYMBOL`, `SELFIES_TOKENS` | The patterns of one symbol and of a whole compact SELFIES string |
| `normalize_smiles_for_rdkit(smiles)` | The rewrites of `RDKIT_REWRITES`, for spellings RDKit rejects |
| `safe_to_selfies`, `safe_smiles_to_selfies`, `convert_smiles_to_selfies`, `build_smiles_to_selfies`, `map_smiles_list_to_selfies` | Batch conversion that skips, and prints, what foundry rejects |

## Canonical SMILES

`canonicalize_smiles` applies these steps in this order. The docstring of `smiles.py` is the
full statement.

1. **Text.** Surrounding whitespace is removed. Whitespace inside is rejected, because RDKit
   reads a space as the end of a SMILES: `CC O` would become ethane.
2. **Parsing.** RDKit parses and sanitizes the string as written, then removes hydrogens written
   as atoms and sets each double bond's neighbor directions again from its stereo. RDKit's own
   removal, during parsing, reads a ring double bond whose direction sits on a hydrogen as the
   other isomer: `[H]/C1=C/CCCCCC1` is cis-cyclooctene, and it reads trans. The rewrites in
   `RDKIT_REWRITES`, for invalid spellings found in source data, apply only when RDKit rejects the
   string as written.
3. **Atom maps** are removed.
4. **Cleanup,** RDKit's `rdMolStandardize.Cleanup` with its defaults: bonds to metals are broken
   into ions, functional groups are normalized (nitro is written charge separated), and charges
   move to the most acidic sites.
5. **Charges,** RDKit's `Uncharger`: acids and bases are neutralized where the net charge allows.
   A counterion keeps its partner charged and a quaternary nitrogen keeps its charge, so glycine
   written as a zwitterion is neutralized and levocarnitine is not.
6. **Stereo.** Tetrahedral centers and double-bond geometry are kept as given, and unspecified
   stereo stays unspecified. Every other chiral tag (allene, square planar, octahedral) is removed.
7. **Writing,** as canonical isomeric Kekulé SMILES. Steps 2 to 7 run again on that string until
   it stops changing, so a canonical string canonicalizes to itself. Every FDA-approved drug and
   ExpansionRx compound settles in one pass; some unusual charged molecules need up to three more,
   and one that needs more than five is rejected.

Kept as given: fragments (salts are not stripped), isotopes (deuterium stays), and tautomers
(two tautomers of one compound give two strings).

## What is rejected, and how a caller learns

Nothing returns None or a different molecule. Each rejection raises an error whose message says
why. The four chemistry errors derive from `ChemistryError`, a `ValueError`.

| Error | Raised for |
|---|---|
| `InvalidSmilesError` | An empty string, whitespace inside, a string RDKit cannot parse, a molecule it cannot sanitize (a pentavalent carbon, an aromatic ring with no Kekulé structure), no atoms, a molecule the standardization breaks (such as `B#[P-]#B`), or one that does not settle |
| `InvalidSelfiesError` | A string that is not bracketed symbols, one the decoder rejects, SELFIES that decode to no atoms (`[Branch1]`), or to a SMILES RDKit rejects |
| `SelfiesRepresentationError` | A valid molecule SELFIES cannot write under its default constraints: a wildcard atom, a quadruple bond, an atom over its default bond capacity (hypervalent iodine), or any molecule whose SELFIES decodes to another |
| `AmbiguousNotationError` | One symbol, sent to `to_smiles` or `to_selfies`, that names one molecule as SMILES and another as SELFIES, such as `[S]`. The message names both readings and a spelling of each that reads one way |
| `RuntimeError` | selfies' global semantic constraints no longer equal its default preset |
| `ImportError` | RDKit older than 2026.03, or selfies older than 2, on first use of a name that needs them |

The `safe_*` and batch helpers return None or skip an input only for a `ChemistryError`, and print
its message. Any other exception is a bug and propagates.

## SELFIES

- **Version and constraints.** The grammar of selfies 2, under the library's default semantic
  constraints, with the strict encoder. The research environment holds selfies 2.2.0; that
  release's `selfies.__version__` still says 2.1.1, so the version check reads the distribution's
  metadata.
- **Round trip.** For every SMILES `smiles_to_selfies` accepts,
  `selfies_to_smiles(smiles_to_selfies(smiles)) == canonicalize_smiles(smiles)`. Stereo and the
  charges the policy keeps survive it. The check runs on every conversion, so a molecule SELFIES
  would write as another raises instead.
- **Serialization.** SELFIES are written spaced, `[C] [C] [O]`, the form the SELFIES-TED
  tokenizers read; to them a compact string is one unknown token. The fragment separator stays
  attached to the next symbol, `[C] [O-1].[Na+1]`, because their vocabulary holds tokens such as
  `.[Na+1]`. Their token count is `len(selfies_symbols(selfies))`.
- **Either notation, one string.** `to_selfies(x)` is `smiles_to_selfies(to_smiles(x))`, so a
  SELFIES input is decoded and written again: every spelling of a molecule, in either notation,
  gives the one SELFIES its canonical SMILES gives. The stored SELFIES of the FDA-approved drugs
  are already that string and come back unchanged, but for the three below.
- **Ambiguous strings.** A string made only of bracket atoms that are also SELFIES symbols is
  SELFIES to `is_selfies`. For most such strings both readings name one molecule (`[O-2].[Zn+2]`,
  `[Mg+2]`). Where they differ, `to_smiles` and `to_selfies` decide by the number of symbols:
  - **One symbol raises `AmbiguousNotationError`.** Nine do: `[B]`, `[Br]`, `[C]`, `[F]`, `[I]`,
    `[N]`, `[O]`, `[P]`, `[S]`. As SMILES each is a lone atom with no hydrogens; as SELFIES, that
    atom's hydride. The error names both readings and a spelling of each that reads one way:
    `[SH0]` is the sulfur atom in either notation, and `S` is hydrogen sulfide as SMILES. Among the
    FDA-approved drugs this refuses the SMILES of sulfur, `[S]`, and the stored SELFIES of water,
    strong ammonia solution, and hydrofluoric acid, `[O]`, `[N]`, and `[F]`, whose SMILES `O`,
    `N`, and `F` are accepted.
  - **Two or more symbols keep the SELFIES reading.** As SMILES such a string is bracket atoms
    joined by single bonds, at least one short of its valence with no hydrogen written: a radical,
    such as dicarbon, `[C][C]`. No FDA-approved drug or ExpansionRx compound is written so.

  When the notation is known, call `canonicalize_smiles`, `smiles_to_selfies`, or
  `selfies_to_smiles` directly; none of them reads the other notation.
- **Idempotence, with nine exceptions.** `to_selfies(to_selfies(x)) == to_selfies(x)` for every
  molecule but the nine hydrides, whose SELFIES is one of the nine symbols: `to_selfies("S")` is
  `[S]`, which `to_selfies` refuses. Their explicit spellings, such as `[OH2]`, are not tokens of
  the SELFIES-TED vocabularies, but for `[SH2]`, so the canonical SELFIES stays the string selfies'
  encoder writes and the models read. Pass such a string to `selfies_to_smiles`.
- **A budget.** An encoder's `max_length` counts tokens, and a SELFIES-TED token is one symbol.
  `truncate_selfies(selfies, max_symbols)` keeps the first `max_symbols` symbols and cuts after a
  whole one, never inside; synth cuts with it when serving and Atlas when embedding. A spaced
  symbol takes a median 5.9 characters in the FDA SELFIES, so a cut of 1,024 characters kept a
  median 168 symbols of the 54 that exceed 1,024 characters, and ended inside a symbol for 29 of
  them. None of the 2,840 exceeds 1,022 symbols, so none is cut at 1,024.
- **Without RDKit.** `SELFIES_SYMBOL`, `SELFIES_TOKENS`, `compact_selfies`, `space_selfies`,
  `selfies_symbols`, and `truncate_selfies` come from `selfies_text`, which imports neither RDKit
  nor selfies. The package loads the other names on first use, so a consumer that only counts or
  cuts symbols, as Atlas does, runs where neither is installed.

## Limitations

- Tautomers are not canonicalized.
- A coordination compound comes out as its ions, and its geometry is lost: cisplatin and
  transplatin written with `@SP` tags give one string.
- The policy neutralizes charges that protonation at an assay pH adds, so a protonated molecule
  and its neutral form give one string. sae_selfies, which protonates, keeps its own conversion
  for that reason.

## Consumers

| Consumer | Uses | Status |
|---|---|---|
| serving_api (synth) | `synth.utilities.chemistry` re-exports the module under its old names and adds `ligands_to_selfies` and `require_ligands_as_selfies`, which write every PLI ligand with `to_selfies`: scoring and embedding (`capabilities/scoring.py`, `capabilities/embedding.py`), batches (`serving/runtime.py`), and networks (`capabilities/network.py`). `predict/truncation.py` cuts with `truncate_selfies`; `utilities/pubchem.py`; `scripts/workflows/selfies_to_structure.py` counts tokens with `selfies_symbols` | Imports it. A ligand this module refuses fails its own batch item or network ligand; a scoring or embedding request raises a `ValueError` that names each one |
| phenomics | `data/chemistry.py` (`canonicalize_smiles`, `canonical_kekule_smiles`, `to_selfies`, `selfies_symbols`); `atlas/adapter_pli.py` (`to_selfies`) | Imports it, and converts as serving does, so a compound serving refuses, such as sulfur's `[S]`, is marked incompatible with Atlas. The pinned remote and run scripts (`remote_prepare_vector_holdout.py`, `remote_prepare_chembl500k.py`, `run_priority0.py`, `run_pli_bridge_audit.py`) keep their own token counts and conversions, as recorded runs used them; its README says what `MANIFEST.sha256` records |
| dataset_builders | `dataset_dev_utils/chemistry.py` re-exports it; `fda/classify.py`, `enzyme_kinetics_data/kinetics_pairs.py`, `annotation_vocab/plinder_to_selfies.py` | Imports it |
| predecessor_synteract | `utils/chemistry.py` re-exports it | Imports it |
| sae_selfies | Its own `selfies_sae.chem`: the raw encoder with IBM's fallback, and dimorphite-dl protonation at the assay pH, as Polaris preprocesses | Keeps its copy, decided 2026-09-24. Its recorded embeddings and SAEs were made from its strings, and foundry's standardization neutralizes what protonation charges: on foundry, SELFIES-TED's input would change for 6,823 of the 7,608 ExpansionRx molecules at pH 7.4 and name a different molecule for 2,191. `tests/test_chem.py` fails if its spacing or token count differs from this module's |
| atlas (production and research) | Stored SELFIES from `Synthyra/PLI` and `Synthyra/atlas_pli_data`, cut to the B encoder's budget with `truncate_selfies` in `production/preembed.py` (PLI pre-embedding, through `preembed_dual.py`), `production/analysis/drug_repurposing/screen_proteome.py`, and `research/eval/predictor.py` | Converts nothing. Imports only the SELFIES text names, so it needs neither RDKit nor selfies |
| folding_service, fastplms | Pass SMILES to ESMFold2 and Biohub as given | Must not canonicalize: the model reads the caller's string |
| tmap | Parses SMILES for fingerprints and non-isomeric Murcko scaffolds | Writes no model input string |

## RDKit 2026.03 or later

From RDKit 2026.03.1, `Kekulize()` no longer depends on atom and bond order, so a Kekulé SMILES
places its double bonds the same way whichever spelling the molecule came from. On an earlier
RDKit the first name that needs RDKit raises `ImportError`. Measured on the 2,838 FDA-approved
drugs in `Synthyra/FDA-Approved-Drugs`:

| | RDKit 2025.9.6 | RDKit 2026.3.6 |
|---|---|---|
| drugs with more than one string, over the study's 48,811 spellings | 1,522 | 19 |
| drugs with more than one string, over 48,768 spellings written by RDKit 2026.3.6 | | 0 |
| stored strings written back unchanged | 2,793 | 2,838 |

The 19 are not a canonicalization fault. The study's spellings were written by RDKit 2025.9.6's
randomized writer, which wrote some of those drugs, such as romidepsin, as a different
stereoisomer: their InChI and CIP labels differ from the stored molecule's. RDKit 2026.3.6 writes
every spelling from the same seeds with the stored stereo, and every drug then has one string.

The rewrite of this module on 2026-09-23, which made it reject what it had passed through and
added the checks above, changes no string on any of those inputs, nor on the 2,838 stored SELFIES
or 136,775 spellings of the ExpansionRx compounds. The decisions of 2026-09-24 change strings on
purpose: `to_selfies` now rewrites a SELFIES that is not canonical, which 44,956 of 48,766 spellings
encoded by selfies directly are, and refuses the nine ambiguous symbols, among them three of the
2,838 stored SELFIES. [One canonical SMILES](../../docs/decisions/foundry/chemistry/2026-09-23_one_canonical_smiles.md)
has the study, what it showed about how much Atlas-PLI's scores depend on spelling, and those
decisions.

Tests: `tests/tier1_unit/test_chemistry.py`, a curated set of 72 molecules with frozen strings, the
policy step by step, every rejection, and Hypothesis properties: one string for every atom order,
Kekulé structure, and hydrogen spelling, whether it arrives as SMILES or as SELFIES; idempotence
and its nine exceptions; the round trip; stereo and charges surviving it; arbitrary text and
arbitrary SELFIES failing loudly or coming back canonical. The nine ambiguous symbols are found by
trying every atom symbol of selfies' alphabet and 560 spellings of elements with hydrogens, charges,
and isotopes; a tenth among them would fail a test. The cut keeps whole symbols on arbitrary SELFIES,
and a subprocess imports the text names with RDKit and selfies blocked. Each of 14 faults, put back
into a copy of the module, fails at least one test. The tooling environment has no RDKit and skips
them; `ws test --projects foundry` runs them under the research interpreter.
