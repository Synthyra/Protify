---
id: foundry.retry
kind: library_module
status: active
aliases: [foundry.retry]
---
# foundry.retry

One way to call a network service again after a failure that passes: a rate limit, a server error,
a dropped or timed-out connection. Standard library only, so any project can vendor it.

```python
from foundry.retry import retry

@retry
def fetch(accession: str) -> dict: ...

@retry(attempts=8, cap_seconds=300)
def submit(job: Job) -> str: ...
```

| Name | Does |
|---|---|
| `retry(function)`, `retry(*, attempts=5, base_seconds=1.0, cap_seconds=120.0, when=transient, backoff=None, sleep=time.sleep)` | Calls the function up to `attempts` times in all while `when(error)` holds, waiting what the response asks (`retry_after`), else `backoff(k)` after attempt `k`, by default `uniform(0, base_seconds * 2**k)`, either capped at `cap_seconds`. Raises the last failure unchanged. Prints one stderr line per retry with the function, error type and status, never the message, which can hold a keyed URL |
| `transient(error)` | True for HTTP 408, 425, 429 and 5xx, and for connection and timeout errors of `requests`, `urllib3`, `httpx`, `http.client` and the standard library, matched by class name |
| `status_of(error)`, `retry_after(error)` | The HTTP status an error carries, and the seconds its response asks to wait: a `Retry-After` header, as a number or an HTTP date, else for a 429 the reset its `RateLimit` header names |
| `quota_reset(value)` | The seconds until a spent quota resets, from a `RateLimit` header (IETF draft): each member names a quota with its remaining requests `r` and reset `t`, as the Hugging Face Hub's `"api";r=0;t=55`; the latest reset of the spent quotas, or None when none is spent. Waiting it, a request outlasts the Hub's rate limit in one pause instead of a backoff's guesses (added 2026-10-08) |

An async or generator function is refused, since it returns before it can fail. Decorate the smallest call that can fail alone, a request rather than the loop of requests, so a
retry repeats one call. `when=lambda error: True` retries every failure, and
`backoff=lambda k: 2.0**k` keeps a fixed schedule, as `foundry.downloads.uniprot.request_with_retry` does.

Promoted on 2026-10-02 from the retry loops written one by one: UniProt's `request_with_retry` now
uses it. Copies that wait to move, each in a project another session held: the `Retry-After`
handling in FastPLMs' E1 retrieval, folding_service's Biohub client, synth's enrichment cache and
the Atlas fold case-study script, and taxon_data's `_request_with_retry`.

Tests: `tests/tier1_unit/test_retry.py`.
