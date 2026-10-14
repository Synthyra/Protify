"""Getting files from outside the workspace, one submodule per kind of source.

- `foundry.downloads.https`: any public HTTPS file, resumable, renamed into place only when whole.
- `foundry.downloads.hub`: one Hugging Face Hub file at a pinned commit, checked against its digest.
- `foundry.downloads.uniprot`: UniProt identifiers, proteomes, and ID mapping.
- `foundry.downloads.string_db`: STRING's per-organism links, sequences, and aliases.

Nothing is imported here, so a caller pays only for the submodule it uses: `hub` needs
`huggingface_hub`, and `uniprot` and `string_db` need pandas and Biopython.
"""
