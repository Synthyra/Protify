---
id: foundry.downloads.hub
kind: library_module
status: active
aliases:
- foundry.downloads.hub
---
# foundry.downloads.hub

One file from the Hugging Face Hub at a pinned revision, checked against its digest. The mirror
of [`foundry.datasets`](../../datasets/README.md)'s publishing: that module puts a folder on the Hub
and returns the revision, this one reads a single file back at a revision the code names.

It exists for files that are neither source nor build output and are too large for the tree:
runtime data a service loads, an evidence table a report draws on, a generated lookup module.
Before it, each of those was an unmanaged artifact that only Logan's machine held, and the tests
that read them skipped.

```python
from foundry.downloads.hub import HubFile

ANNOTATION_NAMES = HubFile(
    repo_id="Synthyra/annotation_vocabulary_names",
    filename="annotation_names.json",
    revision="0d9b1f...",   # a commit hash, 40 hex characters
    sha256="1c1f0a...",
)

path = ANNOTATION_NAMES.path()          # the cache's copy, downloaded once
cached = ANNOTATION_NAMES.cached_path() # the cache's copy, or None, without the network
```

| Name | Does |
|---|---|
| `HubFile(repo_id, filename, revision, sha256, repo_type="dataset")` | One file of one repository at one immutable revision. A revision that is not 40 hex characters is refused at construction, because a branch moves and an offline cache cannot resolve one |
| `HubFile.path()` | The file in the Hugging Face cache, downloaded once if the cache lacks it, verified either way. Raises `HubFileUnavailable` naming `hf auth login` when it is neither cached nor downloadable |
| `HubFile.cached_path()` | The cache's copy, verified, or None. Answers without the network, so an offline suite runs against a file it holds and skips when it does not |
| `HubFile.verify(path)`, `digest(path)` | The digest check on its own, and the SHA-256 of a file read in chunks, which is [`foundry.digests`](../../digests/README.md)' `sha256_file` under this name |
| `HubFileError`, `DigestMismatch`, `HubFileUnavailable` | The failures. A mismatch means the bytes on disk are not the bytes the pin names |
| `hub_copy_differences(api, repo, root, files, digests=True)` | What a dataset repository lacks or holds differently from a pushed folder's files, by name, size and digest, downloading nothing: the Hub's LFS SHA-256 or git blob id against the local bytes. Empty when the repository is a copy. Moved from `production_suite.hub_copy` when the embedding stores' push (`foundry.embedding.hub`) became its second consumer |
| `file_digests(path)` | A file's SHA-256 and git blob id from one read |
| `upload_resuming(api, *, repo_id, folder_path, repo_type, sleep=time.sleep, **options)` | `api.upload_large_folder(...)` with `options` unchanged, run again after it gives up: huggingface_hub 1.33 gives up after eight failed requests in a row, which the account's rate limit causes once an upload of new data has spent the quota. A RuntimeError (its giving up) or a passing HTTP failure starts the next run after `UPLOAD_PAUSE_SECONDS` (330, the 5-minute window and a margin), up to `UPLOAD_ROUNDS` (12) runs; each run resumes the progress the last saved under the folder's `.cache/huggingface/`. Any other failure raises at once |
| `hub_call(call, *args, sleep=time.sleep, when=transient, **kwargs)` | One Hub request, made again while it fails for a reason `when` accepts, up to `HUB_CALL_ATTEMPTS` (10) calls. A 429 waits the reset the Hub's `RateLimit` header names (`foundry.retry.retry_after`), any other failure a full-jitter backoff from `HUB_CALL_BACKOFF_SECONDS` (15 s, capped at 300 s). Wrap the requests after a large upload in it, since that is when the quota is spent; build a paginated listing inside the call (`lambda: list(api.list_repo_tree(...))`) |
| `rate_limited(error)` | True only for the 429: the `when` for a write (`create_commit`, `delete_repo`, `move_repo`, `super_squash_history`), since a 429 means the Hub did nothing, while after a server error the write may have landed |

Authentication belongs to `huggingface_hub`, through `HF_TOKEN` or the token `hf auth login`
stored. No function here accepts, reads, or returns a token, per
[credentials](../../../docs/conventions/credentials.md).

Consumers: serving_api's translator and mechanism services (`annotation_names.json`),
annotation_vocabulary's report builder (`local_association.parquet`), dsm's `synthyra_api`
(`annotation_mapping.py`), and confounders' BioGRID splits (the `processed_datasets` archives).
A consumer that publishes a projection lists this module under `github.includes`, so
`ws publish` vendors it. The four pinned repositories are recorded in
[the approvals record](../../../docs/decisions/workspace/approvals_waiting_on_logan.evidence.md).

Since 2026-09-25 `digest` is `foundry.digests.sha256_file`, which also takes a `str` path and a
chunk size, and `CHUNK_BYTES`, which only `digest` and its test used, is gone; that module's README
lists what changed.

The account's rate limit (3,000 API requests in 5 minutes on the Team plan) counts every process and
machine of the account, other agents' included. A 3.1 TB embedding push died on it on 2026-10-07, so
`foundry.embedding.hub` gained `upload_resuming`, `hub_call` and `rate_limited` that night; they moved
here on 2026-10-08 for the other uploaders: the legacy embedding push (`foundry.embedding.job`),
production_suite's Atlas profile store and ESMC archive, the AtlasFold-Data builder and the pH benchmark's
dataset upload. Call `upload_large_folder` only through `upload_resuming`.

Tests: `tests/tier1_unit/test_hub_files.py`, `test_hub_copies.py` and `test_hub_rate_limit.py`, against a
stand-in Hub. None needs the network.
