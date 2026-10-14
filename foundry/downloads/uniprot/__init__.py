"""UniProt access: identifier parsing, proteome metadata and FASTA downloads, and ID mapping.

Proteome FASTA files come from UniProt's FTP reference proteomes, one protein per gene, with the
REST stream as the fallback: reviewed entries first, then the whole proteome. ID mapping sends
accessions to UniProt's `idmapping` service in chunks, on parallel sessions, and caches what it
resolves as pickles, so a rerun asks only for accessions it has not mapped.
"""

from __future__ import annotations

import io
import os
import tempfile
import time
import urllib.parse
import pandas as pd
import requests

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from Bio import SeqIO
from tqdm.auto import tqdm

from foundry.compression import open_text_maybe_gzip
from foundry.downloads.https import download_file
from foundry.retry import retry
from foundry.serialization import read_json_dict, read_pickle_dict, write_pickle_dict_atomic


UNIPROT_BASE = "https://rest.uniprot.org"
REFERENCE_PROTEOMES_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes"
)
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class UniProtConfig:
    """The ID mapping databases, the retry count, the timeout and backoff in seconds, and the worker threads."""

    idmapping_from: str = "UniProtKB_AC-ID"
    idmapping_to: str = "UniProtKB"
    timeout_s: float = 120.0
    max_retries: int = 8
    backoff_base_s: float = 1.5
    backoff_max_s: float = 60.0
    max_workers: int = 8


@dataclass
class ProteomeInfo:
    """One UniProt proteome: its identifier, organism, protein count, and taxonomic domain."""

    proteome_id: str
    organism_name: str
    organism_id: int
    protein_count: int
    domain: str = "Eukaryota"

    def __repr__(self) -> str:
        return f"Proteome({self.proteome_id}, {self.organism_name}, {self.protein_count} proteins, {self.domain})"


HUMAN_PROTEOME_ID = "UP000005640"
HUMAN_ORGANISM_ID = 9606

DEFAULT_PROTEOMES = [
    ProteomeInfo(
        proteome_id="UP000000625",
        organism_name="Escherichia coli (strain K12) (K12 / MG1655 / ATCC 47076)",
        organism_id=83333,
        protein_count=0,
        domain="Bacteria",
    ),
    ProteomeInfo(
        proteome_id="UP000002311",
        organism_name="Saccharomyces cerevisiae (strain ATCC 204508 / S288c) (Baker's yeast) (ATCC 204508 / S288c)",
        organism_id=559292,
        protein_count=0,
        domain="Eukaryota",
    ),
    ProteomeInfo(
        proteome_id="UP000000589",
        organism_name="Mus musculus (Mouse) (C57BL/6J)",
        organism_id=10090,
        protein_count=0,
        domain="Eukaryota",
    ),
    ProteomeInfo(
        proteome_id="UP000000803",
        organism_name="Drosophila melanogaster (Fruit fly) (Berkeley)",
        organism_id=7227,
        protein_count=0,
        domain="Eukaryota",
    ),
    ProteomeInfo(
        proteome_id="UP000001570",
        organism_name="Bacillus subtilis (strain 168) (168)",
        organism_id=224308,
        protein_count=0,
        domain="Bacteria",
    ),
    ProteomeInfo(
        proteome_id="UP000005640",
        organism_name="Homo sapiens (Human)",
        organism_id=9606,
        protein_count=0,
        domain="Eukaryota",
    ),
    ProteomeInfo(
        proteome_id="UP000006548",
        organism_name="Arabidopsis thaliana (Mouse-ear cress) (cv. Columbia)",
        organism_id=3702,
        protein_count=0,
        domain="Eukaryota",
    ),
    ProteomeInfo(
        proteome_id="UP000001940",
        organism_name="Caenorhabditis elegans (Bristol N2)",
        organism_id=6239,
        protein_count=0,
        domain="Eukaryota",
    ),
]


def parse_uniprot_id(raw_id: str | None) -> str:
    """The accession in `sp|P08238|HS90B_HUMAN`, or the stripped identifier when it has no `|`; None gives ""."""
    if raw_id is None:
        return ""
    raw = str(raw_id).strip()
    if "|" in raw:
        parts = raw.split("|")
        if len(parts) > 1:
            return parts[1].strip()
    return raw


def parse_id_fields(raw_id: str) -> tuple[str, str]:
    """The accession and upper-case gene symbol in `sp|P08238|HS90B_HUMAN` or `tr|...`.

    Without `|`, the accession is the whole identifier and the gene symbol its first word.
    """
    raw = str(raw_id)
    accession = raw
    gene_symbol = raw

    if "|" in raw:
        parts = raw.split("|")
        if len(parts) > 1:
            accession = parts[1].strip()
        gene_field = parts[-1].strip()
        gene_symbol = gene_field.rsplit("_", 1)[0] if "_" in gene_field else gene_field
    else:
        gene_symbol = raw.split()[0]

    return accession.strip(), gene_symbol.strip().upper()


def fetch_proteome_metadata(proteome_id: str) -> ProteomeInfo:
    """A proteome's metadata from UniProt's REST API, its domain from its taxonomic lineage."""
    metadata = _request_json(f"{UNIPROT_BASE}/proteomes/{proteome_id}.json")

    organism_name = None
    if "organism" in metadata:
        if isinstance(metadata["organism"], dict):
            if "scientificName" in metadata["organism"]:
                organism_name = metadata["organism"]["scientificName"]
            elif "name" in metadata["organism"]:
                organism_name = metadata["organism"]["name"]
        elif isinstance(metadata["organism"], str):
            organism_name = metadata["organism"]
    if organism_name is None and "organism_name" in metadata:
        organism_name = metadata["organism_name"]
    if organism_name is None and "taxonomy" in metadata and isinstance(metadata["taxonomy"], dict):
        taxonomy = metadata["taxonomy"]
        if "scientificName" in taxonomy:
            organism_name = taxonomy["scientificName"]
        elif "name" in taxonomy:
            organism_name = taxonomy["name"]
    assert organism_name is not None, f"Missing organism name for {proteome_id}"

    organism_id = None
    if "organism" in metadata and isinstance(metadata["organism"], dict) and "taxonId" in metadata["organism"]:
        organism_id = int(metadata["organism"]["taxonId"])
    if organism_id is None:
        if "organism_id" in metadata:
            organism_id = int(metadata["organism_id"])
        elif "taxonId" in metadata:
            organism_id = int(metadata["taxonId"])
        elif "taxonomyId" in metadata:
            organism_id = int(metadata["taxonomyId"])
        elif "taxonomy" in metadata and isinstance(metadata["taxonomy"], dict) and "taxonId" in metadata["taxonomy"]:
            organism_id = int(metadata["taxonomy"]["taxonId"])
    assert organism_id is not None, f"Missing organism id for {proteome_id}"

    protein_count = 0
    if "protein_count" in metadata:
        protein_count = int(metadata["protein_count"])
    elif "proteinCount" in metadata:
        protein_count = int(metadata["proteinCount"])

    lineage = _extract_lineage_from_metadata(metadata)
    if lineage == "":
        lineage = _fetch_taxonomy_lineage(organism_id)
    assert lineage != "", f"Missing lineage for {proteome_id}; provide a metadata JSON mapping."

    return ProteomeInfo(
        proteome_id=proteome_id,
        organism_name=organism_name,
        organism_id=organism_id,
        protein_count=protein_count,
        domain=_extract_domain_from_lineage(lineage),
    )


def load_proteome_metadata_json(path: str) -> dict[str, ProteomeInfo]:
    """Proteome metadata from a JSON file mapping each proteome identifier to its fields.

    Each entry needs `organism_id` and `organism_name`, and `domain` or a `lineage` to take it
    from; without either, the lineage is fetched from UniProt's taxonomy endpoint.
    """
    entries = read_json_dict(path)
    assert isinstance(entries, dict), "Proteome metadata JSON must be a dict"

    proteomes: dict[str, ProteomeInfo] = {}
    for proteome_id, entry in entries.items():
        assert isinstance(entry, dict), f"Metadata for {proteome_id} must be a dict"
        assert "organism_id" in entry, f"Missing organism_id for {proteome_id}"
        assert "organism_name" in entry, f"Missing organism_name for {proteome_id}"

        organism_id = int(entry["organism_id"])
        if "domain" in entry:
            domain = entry["domain"]
        else:
            lineage = _normalize_lineage(entry["lineage"]) if "lineage" in entry else ""
            if lineage == "":
                lineage = _fetch_taxonomy_lineage(organism_id)
            assert lineage != "", f"Missing lineage for {proteome_id}; add lineage or domain to JSON."
            domain = _extract_domain_from_lineage(lineage)

        proteomes[proteome_id] = ProteomeInfo(
            proteome_id=proteome_id,
            organism_name=entry["organism_name"],
            organism_id=organism_id,
            protein_count=int(entry["protein_count"]) if "protein_count" in entry else 0,
            domain=domain,
        )
    return proteomes


def parse_proteome_tsv(tsv_path: str) -> list[ProteomeInfo]:
    """Proteomes from UniProt's proteome table export, with each domain taken from its lineage."""
    proteome_table = pd.read_csv(tsv_path, sep="\t")

    proteomes: list[ProteomeInfo] = []
    for _, row in proteome_table.iterrows():
        lineage = row.get("Taxonomic lineage", "")
        proteomes.append(
            ProteomeInfo(
                proteome_id=row["Proteome Id"],
                organism_name=row["Organism"],
                organism_id=int(row["Organism Id"]),
                protein_count=int(row["Protein count"]),
                domain=_extract_domain_from_lineage(str(lineage)),
            )
        )
    return proteomes


def download_proteome_fasta(
    proteome_id: str,
    organism_id: int,
    domain: str,
    output_path: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> bool:
    """Download a reference proteome's FASTA, one canonical protein per gene, from UniProt's FTP.

    The file is `<domain>/<proteome_id>/<proteome_id>_<organism_id>.fasta.gz`. Each failed attempt,
    an empty file included, is retried after a delay that doubles; the result says whether one
    succeeded.
    """
    url = f"{REFERENCE_PROTEOMES_URL}/{domain}/{proteome_id}/{proteome_id}_{organism_id}.fasta.gz"
    return _download_fasta(
        url,
        None,
        output_path,
        f"{proteome_id} from {domain}",
        "sequences (one per gene)",
        max_retries,
        retry_delay,
        url_printed=True,
    )


def download_uniprot_proteome(
    proteome_id: str,
    output_path: str,
    organism_id: int | None = None,
    domain: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> bool:
    """Download a proteome's FASTA: the reference proteome, else its reviewed entries, else all of it.

    A missing `organism_id` or `domain` is taken from the proteome's metadata.
    """
    if organism_id is None or domain is None:
        metadata = fetch_proteome_metadata(proteome_id)
        if organism_id is None:
            organism_id = metadata.organism_id
        if domain is None:
            domain = metadata.domain

    assert organism_id is not None, f"Missing organism_id for {proteome_id}"
    assert domain is not None, f"Missing domain for {proteome_id}"

    if download_proteome_fasta(
        proteome_id=proteome_id,
        organism_id=organism_id,
        domain=domain,
        output_path=output_path,
        max_retries=max_retries,
        retry_delay=retry_delay,
    ):
        return True

    print("  Reference proteome download failed; falling back to REST stream")
    if _download_uniprotkb_stream(
        proteome_id=proteome_id,
        query=f"(proteome:{proteome_id} AND reviewed:true)",
        output_path=output_path,
        max_retries=max_retries,
        retry_delay=retry_delay,
        label="REST reviewed",
    ):
        return True

    print("  Reviewed proteome unavailable; downloading full proteome")
    return _download_uniprotkb_stream(
        proteome_id=proteome_id,
        query=f"(proteome:{proteome_id})",
        output_path=output_path,
        max_retries=max_retries,
        retry_delay=retry_delay,
        label="REST full",
    )


def download_human_proteome(cache_dir: str) -> str:
    """The human reference proteome's gzipped FASTA in `cache_dir`, downloaded once."""
    os.makedirs(cache_dir, exist_ok=True)
    fasta_path = os.path.join(cache_dir, f"{HUMAN_PROTEOME_ID}.fasta.gz")
    if os.path.exists(fasta_path):
        print(f"Human proteome FASTA already exists: {fasta_path}")
        return fasta_path

    url = f"{REFERENCE_PROTEOMES_URL}/Eukaryota/{HUMAN_PROTEOME_ID}/{HUMAN_PROTEOME_ID}_{HUMAN_ORGANISM_ID}.fasta.gz"
    print("Downloading human proteome from UniProt...")
    print(f"  URL: {url}")

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()
    with open(fasta_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=8192):
            handle.write(chunk)

    record_count = _count_fasta_records(fasta_path)
    assert record_count > 0, "Downloaded FASTA is empty"
    print(f"  Downloaded {record_count} protein sequences")
    return fasta_path


def load_human_proteome(cache_dir: str) -> tuple[list[str], list[str]]:
    """The human reference proteome's identifiers and sequences, downloaded once into `cache_dir`."""
    fasta_path = download_human_proteome(cache_dir)

    ids: list[str] = []
    sequences: list[str] = []
    with open_text_maybe_gzip(fasta_path, mode="rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            ids.append(record.id)
            sequences.append(str(record.seq))

    print(f"Loaded {len(sequences)} human proteome sequences")
    return ids, sequences


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout_s: float,
    max_retries: int,
    backoff_base_s: float,
    backoff_max_s: float,
    **kwargs: Any,
) -> requests.Response:
    """One request, retried up to `max_retries` times on any error or a 429 or 5xx status.

    Attempt `k`, counting from 0, is followed by a wait of `min(backoff_max_s, backoff_base_s**k)`
    seconds, or what a `Retry-After` header asks within that cap (`foundry.retry`). The last failure
    is raised.
    """

    @retry(attempts=max_retries + 1, cap_seconds=backoff_max_s, when=lambda error: True,
           backoff=lambda attempt: backoff_base_s**attempt, sleep=lambda seconds: time.sleep(seconds))
    def attempt() -> requests.Response:
        response = session.request(method, url, timeout=timeout_s, **kwargs)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raise requests.HTTPError(f"Retryable HTTP {response.status_code}: {response.text[:500]}", response=response)
        response.raise_for_status()
        return response

    return attempt()


def uniprot_idmapping(
    session: requests.Session,
    ids: list[str],
    *,
    cfg: UniProtConfig,
    include_organism_name: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Map `ids` through UniProt's ID mapping service.

    Returns input accession to primary accession, primary accession to sequence, and, with
    `include_organism_name`, primary accession to organism name. Results UniProt returns empty,
    unparsable, or without its `From` and `Entry` columns map nothing.
    """
    if len(ids) == 0:
        return {}, {}, {}

    retry = {
        "timeout_s": cfg.timeout_s,
        "max_retries": cfg.max_retries,
        "backoff_base_s": cfg.backoff_base_s,
        "backoff_max_s": cfg.backoff_max_s,
    }
    submitted = request_with_retry(
        session,
        "POST",
        f"{UNIPROT_BASE}/idmapping/run",
        data={"from": cfg.idmapping_from, "to": cfg.idmapping_to, "ids": ",".join(ids)},
        headers={"Accept": "application/json"},
        **retry,
    )
    job_id = submitted.json()["jobId"]

    status_url = f"{UNIPROT_BASE}/idmapping/status/{job_id}"
    while True:
        status_response = request_with_retry(
            session,
            "GET",
            status_url,
            headers={"Accept": "application/json"},
            allow_redirects=False,
            **retry,
        )
        if status_response.status_code == 303:
            break
        job = status_response.json()
        status = job["jobStatus"]
        if status in ("RUNNING", "NEW"):
            time.sleep(2.0)
            continue
        if status == "FINISHED":
            break
        raise RuntimeError(f"UniProt ID mapping failed: {job}")

    fields = "accession,sequence,organism_name" if include_organism_name else "accession,sequence"
    table_response = request_with_retry(
        session,
        "GET",
        f"{UNIPROT_BASE}/idmapping/uniprotkb/results/stream/{job_id}",
        params={"format": "tsv", "fields": fields},
        headers={"Accept": "text/plain"},
        **{**retry, "timeout_s": cfg.timeout_s * 2},
    )

    text = table_response.text
    if not text.strip():
        return {}, {}, {}
    try:
        mapping_table = pd.read_csv(io.StringIO(text), sep="\t")
    except Exception:  # noqa: broad-except - preserve the public empty-mapping result for unparsable tables
        return {}, {}, {}

    mapping_table.columns = [column.lower() for column in mapping_table.columns]
    if "from" not in mapping_table.columns or "entry" not in mapping_table.columns:
        return {}, {}, {}

    org_col = None
    if include_organism_name:
        if "organism" in mapping_table.columns:
            org_col = "organism"
        elif "organism_name" in mapping_table.columns:
            org_col = "organism_name"

    id_to_primary: dict[str, str] = {}
    primary_to_seq: dict[str, str] = {}
    primary_to_org: dict[str, str] = {}
    for _, row in mapping_table.iterrows():
        primary = str(row["entry"])
        id_to_primary[str(row["from"])] = primary
        if "sequence" in mapping_table.columns:
            sequence = str(row["sequence"])
            if sequence and sequence != "nan":
                primary_to_seq[primary] = sequence
        if org_col is not None:
            organism = str(row[org_col])
            if organism and organism != "nan":
                primary_to_org[primary] = organism

    return id_to_primary, primary_to_seq, primary_to_org


def resolve_uniprot_sequences(
    all_ids: set[str],
    *,
    cfg: UniProtConfig,
    idmapping_chunk_size: int,
    cache_dir: Path,
    seq_chunk_size: int | None = None,
    include_organism_name: bool = False,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Map every identifier in `all_ids` not yet in the cache, in chunks on `cfg.max_workers` threads.

    Returns original identifier to primary accession, primary accession to sequence, and, with
    `include_organism_name`, primary accession to organism name, each merged into and saved back
    to its pickle in `cache_dir`. A chunk that fails is reported and skipped, so its identifiers
    stay unmapped until a rerun. `seq_chunk_size` is accepted and unused.
    """
    id_map_path = cache_dir / "id_to_primary.pkl"
    seq_path = cache_dir / "primary_to_seq.pkl"
    org_path = cache_dir / "primary_to_org.pkl"

    id_to_primary: dict[str, str] = _cache_load(id_map_path, default={})
    primary_to_seq: dict[str, str] = _cache_load(seq_path, default={})
    primary_to_org: dict[str, str] = _cache_load(org_path, default={}) if include_organism_name else {}

    missing_ids = [accession for accession in sorted(all_ids) if accession not in id_to_primary]
    if not missing_ids:
        return id_to_primary, primary_to_seq, primary_to_org

    print(f"Resolving {len(missing_ids):,} missing accessions via UniProt ID Mapping...")
    chunks = [missing_ids[i : i + idmapping_chunk_size] for i in range(0, len(missing_ids), idmapping_chunk_size)]
    cache_lock = Lock()

    def process_chunk(chunk: list[str]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        return uniprot_idmapping(requests.Session(), chunk, cfg=cfg, include_organism_name=include_organism_name)

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="UniProt ID Mapping"):
            try:
                chunk_map, chunk_seqs, chunk_orgs = future.result()
            except Exception as error:  # noqa: broad-except - failed chunks remain unmapped for a later retry
                print(f"Warning: Chunk failed with {error}")
                continue
            with cache_lock:
                id_to_primary.update(chunk_map)
                primary_to_seq.update(chunk_seqs)
                if include_organism_name:
                    primary_to_org.update(chunk_orgs)

    _cache_save(id_map_path, id_to_primary)
    _cache_save(seq_path, primary_to_seq)
    if include_organism_name:
        _cache_save(org_path, primary_to_org)
    return id_to_primary, primary_to_seq, primary_to_org


def _extract_domain_from_lineage(lineage: str) -> str:
    """Eukaryota, Bacteria, Archaea, or Viruses, from a lineage string; Eukaryota, with a warning, if none match."""
    lineage_lower = lineage.lower()
    if "eukaryota" in lineage_lower:
        return "Eukaryota"
    if "bacteria" in lineage_lower:
        return "Bacteria"
    if "archaea" in lineage_lower:
        return "Archaea"
    if "virus" in lineage_lower or "viridae" in lineage_lower:
        return "Viruses"
    print("  Warning: Could not determine domain from lineage, defaulting to Eukaryota")
    return "Eukaryota"


def _request_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    document = response.json()
    assert isinstance(document, dict), f"Expected JSON object from {url}, got {type(document)}"
    return document


def _normalize_lineage(lineage_field: Any) -> str:
    """A lineage as one semicolon-separated string, from a string, a list of names, or a list of taxa."""
    if isinstance(lineage_field, str):
        return lineage_field
    if isinstance(lineage_field, list):
        if len(lineage_field) == 0:
            return ""
        first = lineage_field[0]
        if isinstance(first, str):
            return ";".join(str(item) for item in lineage_field)
        if isinstance(first, dict):
            names: list[str] = []
            for item in lineage_field:
                if "scientificName" in item:
                    names.append(item["scientificName"])
                elif "name" in item:
                    names.append(item["name"])
                else:
                    raise KeyError("Lineage entry missing scientificName/name")
            return ";".join(names)
    raise TypeError(f"Unsupported lineage format: {type(lineage_field)}")


def _extract_lineage_from_metadata(metadata: dict[str, Any]) -> str:
    """The lineage from any of the places UniProt's JSON shapes put it, or "" when none holds one."""
    if "lineage" in metadata:
        return _normalize_lineage(metadata["lineage"])
    for container in ("organism", "taxonomy"):
        if container in metadata and isinstance(metadata[container], dict):
            section = metadata[container]
            if "lineage" in section:
                return _normalize_lineage(section["lineage"])
            if "taxonLineage" in section:
                return _normalize_lineage(section["taxonLineage"])
    if "taxonLineage" in metadata:
        return _normalize_lineage(metadata["taxonLineage"])
    return ""


def _fetch_taxonomy_lineage(organism_id: int) -> str:
    return _extract_lineage_from_metadata(_request_json(f"{UNIPROT_BASE}/taxonomy/{organism_id}.json"))


def _count_fasta_records(fasta_path: str) -> int:
    with open_text_maybe_gzip(fasta_path, mode="rt") as handle:
        return sum(1 for _ in SeqIO.parse(handle, "fasta"))


def _download_uniprotkb_stream(
    proteome_id: str,
    query: str,
    output_path: str,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    label: str | None = None,
) -> bool:
    """Download the canonical entries matching `query` from UniProtKB's REST stream as gzipped FASTA."""
    params = {
        "compressed": "true",
        "download": "true",
        "format": "fasta",
        "includeIsoform": "false",
        "query": query,
    }
    source = label if label is not None else "REST API"
    return _download_fasta(
        f"{UNIPROT_BASE}/uniprotkb/stream",
        params,
        output_path,
        f"{proteome_id} from {source}",
        "sequences",
        max_retries,
        retry_delay,
        url_printed=False,
    )


def _download_fasta(
    url: str,
    params: dict[str, str] | None,
    output_path: str,
    description: str,
    count_label: str,
    max_retries: int,
    retry_delay: float,
    *,
    url_printed: bool,
) -> bool:
    """Fetch and validate a FASTA before replacing `output_path`, retrying with backoff.

    Partial transfers resume within these retries. Each source gets its own staging
    directory, so a fallback cannot append bytes from a different source.
    """
    if max_retries <= 0:
        return False
    destination = Path(output_path)
    request_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    delay = retry_delay
    try:
        with tempfile.TemporaryDirectory(prefix=".uniprot-", dir=destination.parent) as staging:
            candidate = Path(staging) / destination.name
            for attempt in range(max_retries):
                try:
                    print(f"Downloading {description} (attempt {attempt + 1}/{max_retries})...")
                    if url_printed:
                        print(f"  URL: {url}")
                    download_file(request_url, candidate, timeout_seconds=300)
                    record_count = _count_fasta_records(str(candidate))
                    if record_count == 0:
                        raise ValueError("Empty FASTA file")
                    candidate.replace(destination)
                    print(f"  Downloaded {record_count} {count_label}")
                    return True
                except Exception as error:  # noqa: broad-except - preserve the public boolean failure contract
                    # The public API reports exhausted transport or FASTA errors as False.
                    # Reject complete but invalid candidates; retain resumable partial bytes.
                    print(f"  Download failed: {error}")
                    candidate.unlink(missing_ok=True)
                    if attempt < max_retries - 1:
                        print(f"  Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        print(f"  Failed after {max_retries} attempts")
    except OSError as error:
        print(f"  Download failed: {error}")
    return False


def _cache_load(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    return read_pickle_dict(path)


def _cache_save(path: Path, payload: dict[str, Any]) -> None:
    assert isinstance(payload, dict), "Cache payload must be a dict."
    write_pickle_dict_atomic(path, payload)


__all__ = [
    "DEFAULT_PROTEOMES",
    "HUMAN_ORGANISM_ID",
    "HUMAN_PROTEOME_ID",
    "ProteomeInfo",
    "UNIPROT_BASE",
    "UniProtConfig",
    "download_human_proteome",
    "download_proteome_fasta",
    "download_uniprot_proteome",
    "fetch_proteome_metadata",
    "load_human_proteome",
    "load_proteome_metadata_json",
    "parse_id_fields",
    "parse_proteome_tsv",
    "parse_uniprot_id",
    "request_with_retry",
    "resolve_uniprot_sequences",
    "uniprot_idmapping",
]
