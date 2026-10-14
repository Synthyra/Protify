---
id: foundry.downloads.uniprot
kind: library_module
status: active
aliases: [foundry.downloads.uniprot]
---
# foundry.downloads.uniprot

UniProt access: identifier parsing, proteome metadata and FASTA downloads, and ID mapping.

| Name | Does |
|---|---|
| `parse_uniprot_id(raw_id)`, `parse_id_fields(raw_id)` | The accession, and the gene symbol, in `sp|P08238|HS90B_HUMAN` or a plain identifier |
| `ProteomeInfo`, `DEFAULT_PROTEOMES`, `HUMAN_PROTEOME_ID`, `HUMAN_ORGANISM_ID` | A proteome's identifier, organism, protein count, and domain; the eight reference proteomes synth serves |
| `fetch_proteome_metadata`, `load_proteome_metadata_json`, `parse_proteome_tsv` | Proteome metadata from the REST API, a JSON file, or UniProt's proteome table |
| `download_proteome_fasta`, `download_uniprot_proteome` | A reference proteome's FASTA from the FTP site; failing that, the reviewed entries from the REST stream; failing that, all of them |
| `download_human_proteome`, `load_human_proteome` | The human reference proteome, downloaded once into a cache directory |
| `UniProtConfig`, `uniprot_idmapping`, `resolve_uniprot_sequences` | Accessions mapped to primary accessions, sequences, and organisms, in parallel chunks cached as pickles |
| `request_with_retry` | One HTTP request retried on errors and 429 or 5xx statuses with capped exponential backoff |

Promoted on 2026-09-24 from the `uniprot.py` that DatasetDev and synth each held, which
differed only in imports and synth's public name for `request_with_retry`, and from Atlas's
`utils/uniprot.py`, which held the two identifier parsers. Each is now a module that imports its
names from here. Atlas's `utils/uniprot_proteome.py`, a different proteome fetcher with one
consumer, stays in Atlas.

FASTA downloads use the shared HTTPS downloader. They resume partial transfers within
one source's retries, validate the staged FASTA, then replace the destination. Failed
downloads leave an existing destination untouched. Each fallback source has a separate
staging directory; partial bytes are removed when its retries finish, so later sources
and later calls start independently.

## Known issues

These are behaviors every copy had, kept as they were.

- `uniprot_idmapping` maps nothing, without an error, when UniProt's results are empty,
  unparsable, or lack their `From` and `Entry` columns.
- `resolve_uniprot_sequences` reports a failed chunk and skips it, so its accessions stay
  unmapped until a rerun asks for them again.
- `download_proteome_fasta` and `download_uniprot_proteome` report failure by returning False,
  and print their progress rather than logging it.

Tests: `tests/tier1_unit/test_uniprot.py`, with UniProt stood in for; nothing reaches the network.
