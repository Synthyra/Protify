"""The pinned checkpoints behind the stores of the non-protein bases: a Hub revision and every file's SHA-256.

A protein base reads its pins from the FastPLMs model registry. The ligand and annotation models are not FastPLMs
models, so their pins sit here, in the registry's own `CheckpointSource` form, and `dependencies.fetch_snapshot` and
`verified_snapshot` download and check them exactly as they do an ESMC checkpoint: every file is hashed against this table
before anything loads, and a revision that is not 40 hexadecimal characters is refused. The model and the tokenizer
are one repository here, so the tokenizer pin is the subset of its files that make the vocabulary. An annotation model has no
tokenizer: its vocabulary is a release of the annotation datasets, pinned as a `VocabularyPin`.

How a pin is made: take the repository's commit hash from `GET /api/models/<repo>/revision/<revision>`, and for each file
the SHA-256 of its bytes (`lfs.oid` of the tree listing for a large file, `sha256sum` of the file at that revision for a
small one). Never a branch name, never a digest copied from a different revision.

Importing this module needs only the standard library; the registry types load inside the functions.
"""

from __future__ import annotations

import hashlib

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from fastplms.registry import CheckpointSource


# repository, 40-character revision, then each pinned file's SHA-256. The model weights are `model.safetensors`.
SELFIES_TED_LARGE = (
    "lhallee/SELFIES-TED-LARGE",
    "259f46e840edca8fbc6963b17972f0cd004469e3",  # the repository's head on 2026-01-24, the last commit
    {
        "config.json": "edc5e04f4e75902b79269bbfa2bb6fdf410f80e02e1652e81d77db3a7ab1f2d6",
        "model.safetensors": "76833148f76c743bcfedb4f3de6aab482f7a591f5a1ad181d6beeee4727f6bb3",
        "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
        "tokenizer.json": "29b3568f4721f2c8635f8c8468e3c5bacddc58e5970f7a7a323d549c296e23c8",
        "tokenizer_config.json": "6ee1227289cc9f28cd89f06a5f78809941e2cff96ce8f2f1a5407cb149b583b5",
    },
)
# The root checkpoint of Synthyra/AT-experimental-v1 is its best by validation loss (step 6,500). A later revision that only
# changes the README leaves these digests, so this revision stays the pin.
AT_EXPERIMENTAL_V1 = (
    "Synthyra/AT-experimental-v1",
    "7f946290492c7eebcba42672756069dbb89a4452",
    {
        "config.json": "725ac1743b600d18b4e7d4f3d1e8dd0df97d81b5718a9a644bb3602bb2e14324",
        "model.safetensors": "1b7ba323df801a6700d60e63452836a9fd303b5b03d9a335647f142ebe7df4a6",
    },
)
# The root checkpoint of Synthyra/AT-experimental-v2 is its best by validation loss (step 11,000), trained on the v2 release without disease.
AT_EXPERIMENTAL_V2 = (
    "Synthyra/AT-experimental-v2",
    "9e975bb1f55b660fd3554afd9d72c148ba6e8db8",
    {
        "config.json": "9a62ce2865b2ef007eea3caf2f9fd54105d68f9af227b3f05477b222683901f3",
        "model.safetensors": "add5ccb8b2cce34e14c756039722a3ba1360cdc8eef6d5375589c0e54ede3fb1",
    },
)
TOKENIZER_FILES = frozenset({"special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"})

# The pinned checkpoint of each non-protein base, by base id.
PINNED_CHECKPOINTS: dict[str, tuple[str, str, dict[str, str]]] = {
    "selfies_ted_large": SELFIES_TED_LARGE, "at_experimental_v1": AT_EXPERIMENTAL_V1, "at_experimental_v2": AT_EXPERIMENTAL_V2,
}


@dataclass(frozen=True, slots=True)
class VocabularyPin:
    """A vocabulary release (a dataset repository) that numbers an annotation model's terms.

    ``file_sha256`` is the digest of the release's `vocabulary.parquet` and ``listing_sha256`` the digest the release itself
    names the vocabulary by: SHA-256 of its `token<TAB>term_id` lines for the terms, in token order, joined by newlines. Both
    are checked, the second from the parquet's own rows. ``specials`` are the ids after the terms.
    """

    repository: str
    revision: str
    file: str
    file_sha256: str
    terms: int
    specials: dict[str, int]
    listing_sha256: str


# The vocabulary of `Synthyra/AT-experimental-v1`: release `protein_annotation_experimental_v1`, config `default`.
ANNOTATION_EXPERIMENTAL_V1_VOCABULARY = VocabularyPin(
    "Synthyra/annotation_experimental_v1", "ee05eb945d86bf506a8ee4753d5d4465abd69a34", "vocabulary.parquet",
    "425dc2eb11842c972c54b4ccceffa0d204fdebc0b0e18a92b84fbe65c6be1ec2", 22464,
    {"<pad>": 22464, "<bos>": 22465, "<eos>": 22466, "<mask>": 22467, "<unk>": 22468},
    "6c3588874d6c1615fca214c05e1691723837697e8933f02a8cedbbe0a35001cf",
)
# The vocabulary of `Synthyra/AT-experimental-v2`: release `protein_annotation_experimental_v2`, its second revision (no disease).
ANNOTATION_EXPERIMENTAL_V2_VOCABULARY = VocabularyPin(
    "Synthyra/annotation_experimental_v2", "b8882cf4d18139fec3e959040b9f0bf96d75dc1c", "vocabulary.parquet",
    "558e8aee7e2429704f4baaa50cdee6281a47c114599cd2cdaa3699242746a53d", 25119,
    {"<pad>": 25119, "<bos>": 25120, "<eos>": 25121, "<mask>": 25122, "<unk>": 25123},
    "2a68c81c39072b86fd917927b9f0c57f03a56b01bff828a86469f9d6eb4ac25a",
)
PINNED_VOCABULARIES: dict[str, VocabularyPin] = {
    "at_experimental_v1": ANNOTATION_EXPERIMENTAL_V1_VOCABULARY, "at_experimental_v2": ANNOTATION_EXPERIMENTAL_V2_VOCABULARY,
}


def pinned_vocabulary(base_id: str) -> VocabularyPin:
    if base_id not in PINNED_VOCABULARIES:
        raise ValueError(f"No pinned vocabulary for {base_id!r}; the pinned bases are {sorted(PINNED_VOCABULARIES)}.")
    return PINNED_VOCABULARIES[base_id]


def vocabulary_path(pin: VocabularyPin, cache: Path) -> Path:
    """Where the Hub cache layout keeps the release's file under ``cache``."""
    return cache / ("datasets--" + pin.repository.replace("/", "--")) / "snapshots" / pin.revision / pin.file


def fetch_vocabulary(pin: VocabularyPin, cache: Path) -> Path:
    """Download the release's vocabulary file at the pinned revision (a private dataset needs HF_TOKEN in the environment)."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=pin.repository, repo_type="dataset", revision=pin.revision, cache_dir=str(cache), allow_patterns=[pin.file])
    return vocabulary_path(pin, cache)


def verified_vocabulary(pin: VocabularyPin, cache: Path) -> dict[str, object]:
    """Check the cached file's digest and the token-to-term listing it holds, and return the evidence the descriptors record.

    A table whose tokens are not 0 to ``terms`` - 1 then the pinned special ids, or whose listing hashes to another digest,
    fails here, so the ids a store is keyed by mean what the release says.
    """
    import pyarrow.parquet as parquet

    path = vocabulary_path(pin, cache)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; fetch {pin.repository}@{pin.revision} {pin.file}.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != pin.file_sha256:
        raise ValueError(f"{pin.file} of {pin.repository}@{pin.revision} hashes to {digest}, not the pinned {pin.file_sha256}.")
    table = parquet.read_table(path, columns=["token", "term_id"])
    rows = sorted(zip(table.column("token").to_pylist(), table.column("term_id").to_pylist(), strict=True))
    terms = rows[: pin.terms]
    if [token for token, _ in terms] != list(range(pin.terms)):
        raise ValueError(f"The first {pin.terms} tokens of {pin.file} are not 0 to {pin.terms - 1}.")
    if {term: token for token, term in rows[pin.terms:]} != pin.specials:
        raise ValueError(f"The special tokens of {pin.file} are not the pinned {pin.specials}.")
    listing = hashlib.sha256("\n".join(f"{token}\t{term}" for token, term in terms).encode("utf-8")).hexdigest()
    if listing != pin.listing_sha256:
        raise ValueError(f"The token to term listing of {pin.file} hashes to {listing}, not the pinned {pin.listing_sha256}.")
    return {"files": {pin.file: digest}, "listing_sha256": listing, "terms": pin.terms, "specials": dict(pin.specials)}


def pinned_source(base_id: str, *, files: frozenset[str] | None = None) -> CheckpointSource:
    """The registry-form source of ``base_id``'s checkpoint, limited to ``files`` when given (the tokenizer's subset)."""
    from fastplms.registry import CheckpointSource, FileDigest

    if base_id not in PINNED_CHECKPOINTS:
        raise ValueError(f"No pinned checkpoint for {base_id!r}; the pinned bases are {sorted(PINNED_CHECKPOINTS)}.")
    repository, revision, digests = PINNED_CHECKPOINTS[base_id]
    chosen = {path: digest for path, digest in digests.items() if files is None or path in files}
    if files is not None and set(chosen) != set(files):
        raise ValueError(f"{repository} does not pin {sorted(set(files) - set(chosen))}.")
    return CheckpointSource(repository, revision, tuple(FileDigest(path, "sha256", digest) for path, digest in sorted(chosen.items())))


__all__ = [
    "ANNOTATION_EXPERIMENTAL_V1_VOCABULARY", "ANNOTATION_EXPERIMENTAL_V2_VOCABULARY", "AT_EXPERIMENTAL_V1", "AT_EXPERIMENTAL_V2", "PINNED_CHECKPOINTS", "PINNED_VOCABULARIES", "SELFIES_TED_LARGE", "TOKENIZER_FILES",
    "VocabularyPin", "fetch_vocabulary", "pinned_source", "pinned_vocabulary", "verified_vocabulary", "vocabulary_path",
]
