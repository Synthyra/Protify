---
id: foundry.downloads.string_db
kind: library_module
status: active
aliases: [foundry.downloads.string_db]
---
# foundry.downloads.string_db

STRING database files and proteomes. STRING publishes, for each organism, its interaction links,
protein sequences, and aliases, named by the organism's NCBI taxonomy identifier; this module
downloads them from stringdb-downloads.org at `STRING_DB_VERSION` (v12.0) and reads them.

| Name | Does |
|---|---|
| `ensure_string_{links,links_detailed,physical_links,fasta,alias}_file(org_id, data_dir, version)` | The local file under `data_dir` (default `data/string_db`), downloaded first when it is missing |
| `string_*_filename`, `string_*_url`, `string_*_path` | The name, download address, and local path of each file |
| `load_string_links_df`, `load_string_links_detailed_df`, `load_string_physical_links_df` | Links as a data frame, `protein1` and `protein2` renamed `IdA` and `IdB` unless `rename_cols=False`. The detailed file adds each evidence channel's score; the physical file holds physical-channel pairs only |
| `load_string_fasta_dict`, `load_string_alias_df`, `load_string_alias_dict`, `load_string_data` | Sequences, aliases, and all three at once |
| `build_string_to_uniprot_mapping`, `build_string_to_uniprot_mapping_all` | STRING identifier to one UniProt accession, a six-character one preferred, or to all of them |
| `build_uniprot_interaction_lookup(org_id, min_score=0)` | The highest `combined_score` of each accession pair |
| `Organism`, `load_from_fasta`, `load_from_string_db`, `load_organism`, `combine_organisms` | A proteome, from a FASTA file, a taxid, or a name in `ORG_NAME_TO_ID`; several combined under `<taxid>:` prefixes |
| `ORG_NAME_TO_ID`, `ID_TO_ORG_NAME`, `resolve_string_org_id` | Organism names and taxids, with ProteomeLM's pathogen panel; the taxid STRING uses where UniProt's strain taxid differs |

`Organism.subset(n, seed)` seeds Python's global `random`, as every copy did, so a caller's later
draws from `random` follow from `seed`.

Promoted on 2026-09-24 from the `string_db_utils.py` that DatasetDev, Atlas, and synth each held.
DatasetDev's and synth's differed only in imports. Atlas's added the pathogen panel and the
detailed and physical link files, which this module keeps, so DatasetDev and synth gain those
names in `ORG_NAME_TO_ID` and nothing else changes for them. Each copy is now a module that
imports its names from here.

Tests: `tests/tier1_unit/test_string_db.py`, with Atlas's `tests/test_utils_string_db.py` as the
guard on the names it imports. Nothing here reaches the network in a test.
