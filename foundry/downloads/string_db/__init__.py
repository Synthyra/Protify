"""STRING database files and proteomes: download, local paths, and loaders.

STRING publishes, for each organism, its interaction links, protein sequences, and aliases,
named by the organism's NCBI taxonomy identifier. Each `ensure_*` function downloads one file
from stringdb-downloads.org at `STRING_DB_VERSION` into `data_dir` when it is missing and returns
its path; each `load_*` function reads one. `Organism` holds a proteome, read from STRING or from
any FASTA file.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
import pandas as pd

from dataclasses import dataclass, field
from pathlib import Path
from Bio import SeqIO

from foundry.compression import open_text_maybe_gzip
from foundry.downloads.https import download_file


STRING_DB_VERSION = "v12.0"
DEFAULT_STRING_DB_DIR = "data/string_db"

ORG_NAME_TO_ID: dict[str, int] = {
    "homo_sapiens": 9606,
    "human": 9606,
    "mus_musculus": 10090,
    "mouse": 10090,
    "arabidopsis_thaliana": 3702,
    "saccharomyces_cerevisiae": 4932,
    "yeast": 4932,
    "drosophila_melanogaster": 7227,
    "fruit_fly": 7227,
    "danio_rerio": 7955,
    "zebrafish": 7955,
    "caenorhabditis_elegans": 6239,
    "c_elegans": 6239,
    "escherichia_coli_k12": 511145,
    "e_coli": 511145,
    "escherichia_coli_o157_h7": 155864,
    "rattus_norvegicus": 10116,
    "rat": 10116,
    "pseudomonas_aeruginosa_pao1": 208964,
    "plasmodium_falciparum_3d7": 36329,
    "plasmodium_vivax_sal1": 126793,
    "plasmodium_berghei_anka": 5823,
    "plasmodium_yoelii": 73239,
    "plasmodium_knowlesi_strain_h": 5851,
    # ProteomeLM's pathogen panel (Bitbol-Lab, Fig. 3E): the taxids of its repository's
    # `experiments/interactomes/pathogens.ipynb`, cell 26. E. coli and P. aeruginosa are above.
    "legionella_pneumophila": 272624,
    "listeria_monocytogenes": 169963,
    "neisseria_meningitidis": 122586,
    "acinetobacter_pittii": 871585,
    "francisella_tularensis": 177416,
    # Three species take the strain STRING files instead of ProteomeLM's, whose identifiers
    # return 404 from stringdb-downloads.org at v12.0: C. trachomatis 272561 (ProteomeLM
    # 315277), B. henselae 38323 (283166), and Y. pestis CO92 214092 (632).
    "chlamydia_trachomatis": 272561,
    "clostridioides_difficile": 272563,
    "mycoplasma_genitalium": 243273,
    "streptococcus_pneumoniae": 171101,
    "staphylococcus_aureus": 93061,
    "bacteroides_fragilis": 295405,
    "bartonella_henselae": 38323,
    "helicobacter_pylori": 85962,
    "yersinia_pestis": 214092,
    "vibrio_cholerae": 243277,
    "mycobacterium_tuberculosis": 83332,
    "salmonella_enterica": 99287,
}

# The last name listed for a taxid names it, so 9606 is "human".
ID_TO_ORG_NAME: dict[int, str] = {org_id: name for name, org_id in ORG_NAME_TO_ID.items()}

# UniProt files some organisms under a strain taxid that STRING files under another.
_UNIPROT_TO_STRING_ORG_ID: dict[int, int] = {
    83333: 511145,  # E. coli K-12
    559292: 4932,  # S. cerevisiae S288C
}

STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/stream/protein.links.{version}/"
    "{org_id}.protein.links.{version}.onlyAB.csv.gz"
)
STRING_LINKS_DETAILED_URL = (
    "https://stringdb-downloads.org/download/protein.links.detailed.{version}/"
    "{org_id}.protein.links.detailed.{version}.txt.gz"
)
STRING_PHYSICAL_LINKS_URL = (
    "https://stringdb-downloads.org/download/protein.physical.links.{version}/"
    "{org_id}.protein.physical.links.{version}.txt.gz"
)
STRING_FASTA_URL = (
    "https://stringdb-downloads.org/download/protein.sequences.{version}/"
    "{org_id}.protein.sequences.{version}.fa.gz"
)
STRING_ALIAS_URL = (
    "https://stringdb-downloads.org/download/protein.aliases.{version}/"
    "{org_id}.protein.aliases.{version}.txt.gz"
)

_SWISSPROT_RE = re.compile(r"^[A-Z][0-9][A-Z0-9]{3}[0-9]$")


def resolve_string_org_id(org_id: int) -> int:
    """The taxid STRING files an organism under, given UniProt's."""
    return _UNIPROT_TO_STRING_ORG_ID.get(org_id, org_id)


@dataclass
class Organism:
    """One proteome: protein identifiers and their sequences, in the same order."""

    name: str
    org_id: int | None = None
    ids: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    source: str = "unknown"

    def __len__(self) -> int:
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"Organism(name='{self.name}', n_sequences={len(self)}, source='{self.source}')"

    def get_sequence(self, protein_id: str) -> str:
        assert protein_id in self.ids, f"Protein ID '{protein_id}' not found"
        return self.sequences[self.ids.index(protein_id)]

    def subset(self, n: int, seed: int = 42) -> Organism:
        """`n` proteins drawn at random. Seeds Python's global `random` with `seed`, as it always has."""
        random.seed(seed)
        indices = random.sample(range(len(self)), min(n, len(self)))
        return Organism(
            name=self.name,
            org_id=self.org_id,
            ids=[self.ids[i] for i in indices],
            sequences=[self.sequences[i] for i in indices],
            source=self.source,
        )

    def filter_by_length(self, min_length: int = 0, max_length: int = 100000) -> Organism:
        """The proteins whose length is within [min_length, max_length]."""
        kept = [
            (protein_id, sequence)
            for protein_id, sequence in zip(self.ids, self.sequences, strict=True)
            if min_length <= len(sequence) <= max_length
        ]
        return Organism(
            name=self.name,
            org_id=self.org_id,
            ids=[protein_id for protein_id, _ in kept],
            sequences=[sequence for _, sequence in kept],
            source=self.source,
        )


def string_links_filename(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return f"{_normalize_org_id(org_id)}.protein.links.{version}.onlyAB.csv.gz"


def string_links_detailed_filename(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return f"{_normalize_org_id(org_id)}.protein.links.detailed.{version}.txt.gz"


def string_physical_links_filename(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return f"{_normalize_org_id(org_id)}.protein.physical.links.{version}.txt.gz"


def string_fasta_filename(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return f"{_normalize_org_id(org_id)}.protein.sequences.{version}.fa.gz"


def string_alias_filename(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return f"{_normalize_org_id(org_id)}.protein.aliases.{version}.txt.gz"


def string_links_url(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return STRING_LINKS_URL.format(org_id=_normalize_org_id(org_id), version=version)


def string_links_detailed_url(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return STRING_LINKS_DETAILED_URL.format(org_id=_normalize_org_id(org_id), version=version)


def string_physical_links_url(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return STRING_PHYSICAL_LINKS_URL.format(org_id=_normalize_org_id(org_id), version=version)


def string_fasta_url(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return STRING_FASTA_URL.format(org_id=_normalize_org_id(org_id), version=version)


def string_alias_url(org_id: int | str, version: str = STRING_DB_VERSION) -> str:
    return STRING_ALIAS_URL.format(org_id=_normalize_org_id(org_id), version=version)


def string_links_path(org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION) -> str:
    return os.path.join(data_dir, string_links_filename(org_id, version))


def string_links_detailed_path(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return os.path.join(data_dir, string_links_detailed_filename(org_id, version))


def string_physical_links_path(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return os.path.join(data_dir, string_physical_links_filename(org_id, version))


def string_fasta_path(org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION) -> str:
    return os.path.join(data_dir, string_fasta_filename(org_id, version))


def string_alias_path(org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION) -> str:
    return os.path.join(data_dir, string_alias_filename(org_id, version))


def ensure_string_links_file(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return _ensure_file(
        string_links_path(org_id, data_dir, version), string_links_url(org_id, version), "links"
    )


def ensure_string_links_detailed_file(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return _ensure_file(
        string_links_detailed_path(org_id, data_dir, version),
        string_links_detailed_url(org_id, version),
        "detailed links",
    )


def ensure_string_physical_links_file(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return _ensure_file(
        string_physical_links_path(org_id, data_dir, version),
        string_physical_links_url(org_id, version),
        "physical links",
    )


def ensure_string_fasta_file(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return _ensure_file(string_fasta_path(org_id, data_dir, version), string_fasta_url(org_id, version), "FASTA")


def ensure_string_alias_file(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> str:
    return _ensure_file(string_alias_path(org_id, data_dir, version), string_alias_url(org_id, version), "alias")


def load_string_links_df(
    org_id: int | str,
    data_dir: str = DEFAULT_STRING_DB_DIR,
    version: str = STRING_DB_VERSION,
    rename_cols: bool = True,
) -> pd.DataFrame:
    """The links file: `protein1`, `protein2` (renamed `IdA`, `IdB`), and `combined_score`, 0 to 1000."""
    link_path = ensure_string_links_file(org_id, data_dir=data_dir, version=version)
    with open_text_maybe_gzip(link_path, mode="rt") as handle:
        link_df = pd.read_csv(handle)
    if rename_cols:
        link_df = link_df.rename(columns={"protein1": "IdA", "protein2": "IdB"})
    return link_df


def load_string_links_detailed_df(
    org_id: int | str,
    data_dir: str = DEFAULT_STRING_DB_DIR,
    version: str = STRING_DB_VERSION,
    rename_cols: bool = True,
) -> pd.DataFrame:
    """The links with each evidence channel's score, 0 to 1000, beside `combined_score`.

    The channels are `neighborhood`, `neighborhood_transferred`, `fusion`, `cooccurence`,
    `homology`, `coexpression`, `coexpression_transferred`, `experiments`,
    `experiments_transferred`, `database`, `database_transferred`, `textmining`, and
    `textmining_transferred`. FlashPPI's M. genitalium screen takes its positives from
    `experiments`. The file is larger than the links file, and STRING hosts it for every species.
    """
    link_path = ensure_string_links_detailed_file(org_id, data_dir=data_dir, version=version)
    with open_text_maybe_gzip(link_path, mode="rt") as handle:
        link_df = pd.read_csv(handle, sep=" ")
    if rename_cols:
        link_df = link_df.rename(columns={"protein1": "IdA", "protein2": "IdB"})
    return link_df


def load_string_physical_links_df(
    org_id: int | str,
    data_dir: str = DEFAULT_STRING_DB_DIR,
    version: str = STRING_DB_VERSION,
    rename_cols: bool = True,
) -> pd.DataFrame:
    """The links with physical-channel evidence only: `protein1`, `protein2`, `combined_score`.

    FlashPPI's E. coli breakdown calls a predicted pair physical when it is here, functional when
    it is only in the links file, and novel when it is in neither.
    """
    link_path = ensure_string_physical_links_file(org_id, data_dir=data_dir, version=version)
    with open_text_maybe_gzip(link_path, mode="rt") as handle:
        link_df = pd.read_csv(handle, sep=" ")
    if rename_cols:
        link_df = link_df.rename(columns={"protein1": "IdA", "protein2": "IdB"})
    return link_df


def load_string_fasta_dict(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> dict[str, str]:
    fasta_path = ensure_string_fasta_file(org_id, data_dir=data_dir, version=version)
    with open_text_maybe_gzip(fasta_path, mode="rt") as handle:
        return {record.id: str(record.seq) for record in SeqIO.parse(handle, "fasta")}


def load_string_alias_df(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> pd.DataFrame:
    alias_path = ensure_string_alias_file(org_id, data_dir=data_dir, version=version)
    with open_text_maybe_gzip(alias_path, mode="rt", encoding="utf-8") as handle:
        return pd.read_csv(handle, sep="\t")


def load_string_alias_dict(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> dict[str, str]:
    """Each alias mapped to its STRING protein identifier. An alias listed twice keeps the last."""
    alias_df = load_string_alias_df(org_id, data_dir=data_dir, version=version)
    return dict(zip(alias_df["alias"], alias_df["#string_protein_id"], strict=True))


def load_string_data(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """The renamed links, the sequences, and the alias map of one organism."""
    link_df = load_string_links_df(org_id, data_dir=data_dir, version=version, rename_cols=True)
    sequence_by_id = load_string_fasta_dict(org_id, data_dir=data_dir, version=version)
    string_id_by_alias = load_string_alias_dict(org_id, data_dir=data_dir, version=version)
    return link_df, sequence_by_id, string_id_by_alias


def load_from_fasta(fasta_path: str | Path, organism_name: str | None = None, org_id: int | None = None) -> Organism:
    """A proteome from a FASTA file, plain or gzipped, named for the file unless `organism_name` is given."""
    fasta_path = Path(fasta_path)
    assert fasta_path.exists(), f"FASTA file not found: {fasta_path}"

    if organism_name is None:
        organism_name = fasta_path.stem
        for extension in (".fasta", ".fa", ".gz"):
            organism_name = organism_name.replace(extension, "")

    ids: list[str] = []
    sequences: list[str] = []
    with open_text_maybe_gzip(fasta_path, mode="rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))
    return Organism(name=organism_name, org_id=org_id, ids=ids, sequences=sequences, source=str(fasta_path))


def load_from_string_db(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> Organism:
    """An organism's STRING proteome, named from `ID_TO_ORG_NAME` or else `org_<taxid>`."""
    org_id_str = _normalize_org_id(org_id)
    assert org_id_str.isdigit(), f"STRING DB org_id must be numeric, got: {org_id_str}"
    org_id_int = int(org_id_str)

    fasta_file = ensure_string_fasta_file(org_id_str, data_dir=data_dir, version=version)
    org_name = ID_TO_ORG_NAME.get(org_id_int, f"org_{org_id_str}")
    return load_from_fasta(fasta_file, organism_name=org_name, org_id=org_id_int)


def load_organism(
    organism: Organism | int | str | Path,
    data_dir: str = DEFAULT_STRING_DB_DIR,
    version: str = STRING_DB_VERSION,
) -> Organism:
    """An `Organism` as given, or loaded from a taxid, a name in `ORG_NAME_TO_ID`, or a FASTA path."""
    if isinstance(organism, Organism):
        return organism
    if isinstance(organism, int):
        return load_from_string_db(organism, data_dir=data_dir, version=version)

    organism_str = str(organism).strip()
    assert organism_str != "", "organism must be a non-empty Organism, int, or string."

    normalized_name = organism_str.lower().replace(" ", "_")
    if normalized_name in ORG_NAME_TO_ID:
        return load_from_string_db(ORG_NAME_TO_ID[normalized_name], data_dir=data_dir, version=version)
    if organism_str.isdigit():
        return load_from_string_db(int(organism_str), data_dir=data_dir, version=version)

    organism_path = Path(organism_str)
    assert organism_path.exists(), (
        f"Unknown organism source '{organism_str}'. Expected a known organism name, STRING taxid, or FASTA path."
    )
    return load_from_fasta(organism_path)


def combine_organisms(organisms: list[Organism], name: str = "combined") -> Organism:
    """One proteome from several, each identifier prefixed `<taxid>:`, or `<name>:` without a taxid."""
    assert len(organisms) > 0, "organisms must be a non-empty list."
    combined_ids: list[str] = []
    combined_sequences: list[str] = []
    combined_sources: list[str] = []

    for organism in organisms:
        assert isinstance(organism, Organism), "combine_organisms expects Organism instances."
        prefix = organism.name if organism.org_id is None else f"{organism.org_id}"
        combined_sources.append(organism.source)
        for protein_id, sequence in zip(organism.ids, organism.sequences, strict=True):
            combined_ids.append(f"{prefix}:{protein_id}")
            combined_sequences.append(sequence)

    return Organism(
        name=name,
        org_id=None,
        ids=combined_ids,
        sequences=combined_sequences,
        source=";".join(combined_sources),
    )


def build_string_to_uniprot_mapping(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> dict[str, str]:
    """Each STRING protein identifier mapped to one of its `UniProt_AC` aliases.

    A six-character accession, the format of most reviewed Swiss-Prot entries, is preferred to a
    longer one; among accessions of one kind, the first seen stays.
    """
    alias_df = load_string_alias_df(org_id, data_dir=data_dir, version=version)
    uniprot_aliases = alias_df[alias_df["source"] == "UniProt_AC"]

    string_to_uniprot: dict[str, str] = {}
    is_swissprot: dict[str, bool] = {}
    for _, row in uniprot_aliases.iterrows():
        string_id = row["#string_protein_id"]
        accession = row["alias"]
        accession_is_swissprot = _is_swissprot_format(accession)
        if string_id not in string_to_uniprot:
            string_to_uniprot[string_id] = accession
            is_swissprot[string_id] = accession_is_swissprot
        elif accession_is_swissprot and not is_swissprot[string_id]:
            string_to_uniprot[string_id] = accession
            is_swissprot[string_id] = True
    return string_to_uniprot


def build_string_to_uniprot_mapping_all(
    org_id: int | str, data_dir: str = DEFAULT_STRING_DB_DIR, version: str = STRING_DB_VERSION
) -> dict[str, list[str]]:
    """Each STRING protein identifier mapped to every one of its `UniProt_AC` aliases, in file order."""
    alias_df = load_string_alias_df(org_id, data_dir=data_dir, version=version)
    uniprot_aliases = alias_df[alias_df["source"] == "UniProt_AC"]

    string_to_uniprot: dict[str, list[str]] = {}
    for _, row in uniprot_aliases.iterrows():
        string_to_uniprot.setdefault(row["#string_protein_id"], []).append(row["alias"])
    return string_to_uniprot


def build_uniprot_interaction_lookup(
    org_id: int | str,
    data_dir: str = DEFAULT_STRING_DB_DIR,
    version: str = STRING_DB_VERSION,
    min_score: int = 0,
) -> dict[tuple[str, str], int]:
    """The highest `combined_score` of each accession pair, keyed in sorted order, at `min_score` or above."""
    started = time.perf_counter()
    links_df = load_string_links_df(org_id, data_dir=data_dir, version=version, rename_cols=False)
    string_to_uniprot = build_string_to_uniprot_mapping(org_id, data_dir=data_dir, version=version)
    # A proteome holds up to millions of links, walked here row by row.
    print(
        f"STRING {org_id}: building the UniProt interaction lookup from {len(links_df):,} links",
        file=sys.stderr,
        flush=True,
    )
    lookup: dict[tuple[str, str], int] = {}
    for _, row in links_df.iterrows():
        score = int(row["combined_score"])
        if score < min_score:
            continue
        protein_a, protein_b = row["protein1"], row["protein2"]
        if protein_a not in string_to_uniprot or protein_b not in string_to_uniprot:
            continue
        accession_a = string_to_uniprot[protein_a]
        accession_b = string_to_uniprot[protein_b]
        key = (min(accession_a, accession_b), max(accession_a, accession_b))
        if key not in lookup or score > lookup[key]:
            lookup[key] = score
    print(
        f"STRING {org_id}: {len(lookup):,} accession pairs at score {min_score} or above "
        f"in {time.perf_counter() - started:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return lookup


def _normalize_org_id(org_id: int | str) -> str:
    org_id_str = str(org_id).strip()
    assert org_id_str != "", "org_id must be a non-empty string or int."
    return org_id_str


def _is_swissprot_format(accession: str) -> bool:
    """Whether `accession` has the six-character UniProt format. Newer accessions have ten."""
    return bool(_SWISSPROT_RE.match(accession))


def _ensure_file(file_path: str, url: str, description: str) -> str:
    """`file_path`, downloaded from `url` first when it is missing.

    `download_file` renames the file into place only once it is whole, so an interrupted
    download resumes on the next call instead of leaving a truncated file that reads as present.
    """
    missing = not os.path.isfile(file_path)
    started = time.perf_counter()
    if missing:
        print(f"STRING: downloading the {description} file {url}", file=sys.stderr, flush=True)
    download = download_file(url, file_path)
    if missing:
        print(
            f"STRING: {download.size_bytes:,} bytes to {file_path} in {time.perf_counter() - started:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    assert os.path.exists(file_path), f"STRING DB {description} file not found: {file_path}"
    return file_path


__all__ = [
    "DEFAULT_STRING_DB_DIR",
    "ID_TO_ORG_NAME",
    "ORG_NAME_TO_ID",
    "STRING_ALIAS_URL",
    "STRING_DB_VERSION",
    "STRING_FASTA_URL",
    "STRING_LINKS_DETAILED_URL",
    "STRING_LINKS_URL",
    "STRING_PHYSICAL_LINKS_URL",
    "Organism",
    "build_string_to_uniprot_mapping",
    "build_string_to_uniprot_mapping_all",
    "build_uniprot_interaction_lookup",
    "combine_organisms",
    "ensure_string_alias_file",
    "ensure_string_fasta_file",
    "ensure_string_links_detailed_file",
    "ensure_string_links_file",
    "ensure_string_physical_links_file",
    "load_from_fasta",
    "load_from_string_db",
    "load_organism",
    "load_string_alias_df",
    "load_string_alias_dict",
    "load_string_data",
    "load_string_fasta_dict",
    "load_string_links_detailed_df",
    "load_string_links_df",
    "load_string_physical_links_df",
    "resolve_string_org_id",
    "string_alias_filename",
    "string_alias_path",
    "string_alias_url",
    "string_fasta_filename",
    "string_fasta_path",
    "string_fasta_url",
    "string_links_detailed_filename",
    "string_links_detailed_path",
    "string_links_detailed_url",
    "string_links_filename",
    "string_links_path",
    "string_links_url",
    "string_physical_links_filename",
    "string_physical_links_path",
    "string_physical_links_url",
]
