---
id: foundry.downloads
kind: library_module
status: active
aliases: [foundry.downloads]
---

# foundry.downloads

Every way the workspace fetches a file it does not hold, in one package, one submodule per kind of
source. Import the submodule; the package itself imports nothing, so a caller pays only for the
dependencies of the source it uses.

| Submodule | Source | Main names |
|---|---|---|
| [`https`](https/README.md) | Any public HTTPS file | `download_file` (resumable, renamed into place only when whole, digest checked), `probe_headers`, `recorded_headers` |
| [`hub`](hub/README.md) | One Hugging Face Hub file at a pinned commit | `HubFile(repo_id, filename, revision, sha256).path()` |
| [`uniprot`](uniprot/README.md) | UniProt REST and FTP | `download_uniprot_proteome`, `uniprot_idmapping`, `resolve_uniprot_sequences`, `request_with_retry` |
| [`string_db`](string_db/README.md) | stringdb-downloads.org | `ensure_string_*_file`, `load_string_data`, `load_organism` |

Logan asked on 2026-09-26 for downloads to have one module shared across the workspace, for the
Hub, UniProt, and the rest. `hub` was `foundry.hub_files`, `uniprot` was `foundry.uniprot`, and `string_db` was
`foundry.string_db`; they moved here that day unchanged, apart from `string_db`, which now
downloads through `https`. Until then it wrote each STRING file straight to its final path with
`urllib.request.urlretrieve`, so an interrupted download left a truncated file that every later
call read as present. `https` is phenomics' resumable downloader with its provenance manifest left
behind in phenomics, since only phenomics records license and attribution per source.

Publishing to the Hub is not here: that is [`foundry.datasets`](../datasets/README.md).

## Known issues

- TODO(downloads_adoption): other code fetches files its own way, and each copy that also resumes,
  checks a digest, or renames into place when whole should call `https.download_file` or
  `hub.HubFile` instead. Remaining candidates found on 2026-09-26: Atlas's `utils/uniprot_proteome.py`,
  tmap's `utils/proteins.py`, protein_screening's `acquisition.py` and `reference_acquisition.py`,
  and dataset_builders' `ec_data/uniprot_cluster_splits.py`, `go_data/go_multilabel_dataset.py`, and `taxon_data`. Those in
  trees marked `converted: false` wait for their conversion pass.

UniProt's FASTA downloader adopted `https.download_file` on 2026-10-14, with separate
staging per source and FASTA validation before replacement.

Tests: `tests/tier1_unit/test_downloads_https.py`, `test_hub_files.py`, `test_uniprot.py`, and
`test_string_db.py`. None reaches the network.
