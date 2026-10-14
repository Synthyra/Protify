"""Hub requests and uploads that outlast the account's rate limit.

The Hub answers every API request of an account with HTTP 429 once the account has made its quota of requests in the
window (3,000 in 5 minutes on the Team plan), whichever process or machine sends it. A push of new data with
`upload_large_folder` spends that in about 5 minutes: the commits and preupload calls of its batches are refused while
its Xet chunk uploads go on, and huggingface_hub 1.33 gives up after eight refusals in a row (`MAX_CONSECUTIVE_ERRORS`),
raising RuntimeError with its progress saved under the folder's `.cache/huggingface/`. A 3.1 TB embedding push died that
way on 2026-10-07; `foundry.embedding.hub` gained these three that night, and they moved here for their other callers.

- `upload_resuming` runs `upload_large_folder` again after `UPLOAD_PAUSE_SECONDS`, longer than the window; the saved
  progress means a later run hashes and sends nothing twice and commits what was sent.
- `hub_call` sends one request again while it fails for a passing reason (`foundry.retry.transient`): a 429 waits the
  reset the Hub's `RateLimit` header names (`foundry.retry.retry_after`), any other failure a full-jitter backoff.
- `rate_limited` accepts only the 429, the condition for repeating a write (a commit, a deletion, a rename, a squash): a
  429 means the Hub did nothing, while after a server error the write may have landed.
"""

from __future__ import annotations

import sys
import time

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from foundry.retry import retry, status_of, transient


R = TypeVar("R")

RATE_LIMITED_STATUS = 429  # the Hub's answer once the account has spent its requests of the window
UPLOAD_ROUNDS = 12  # runs of `upload_large_folder` for one upload, each resuming the progress the last one saved
UPLOAD_PAUSE_SECONDS = 330.0  # between two runs: the Hub's 5-minute rate-limit window and a margin
HUB_CALL_ATTEMPTS = 10  # calls of one request while it fails for a passing reason
HUB_CALL_BACKOFF_SECONDS = (15.0, 300.0)  # base and cap of the full-jitter pauses between them, about 20 minutes in all


def hub_call(call: Callable[..., R], *args: Any, sleep: Callable[[float], None] = time.sleep,
             when: Callable[[BaseException], bool] = transient, **kwargs: Any) -> R:
    """``call(*args, **kwargs)``, made again while it fails for a reason ``when`` accepts, up to `HUB_CALL_ATTEMPTS` calls.

    By default the reason is a passing one (`foundry.retry.transient`), above all the 429 every request of the account
    gets once an upload has spent the quota. A 429 waits the reset its `RateLimit` header names; any other failure waits
    a full-jitter backoff from `HUB_CALL_BACKOFF_SECONDS`, long enough in all to outlast the window. A call that reads a
    paginated listing must build the listing inside ``call`` (``lambda: list(api.list_repo_tree(...))``), since a
    generator a failed attempt began yields nothing to the next.
    """
    base, cap = HUB_CALL_BACKOFF_SECONDS
    return retry(attempts=HUB_CALL_ATTEMPTS, base_seconds=base, cap_seconds=cap, when=when, sleep=sleep)(call)(*args, **kwargs)


def rate_limited(error: BaseException) -> bool:
    """Whether the Hub refused a request for the account's quota, so the request did nothing and may be sent again.

    The only failure a write is repeated on (``hub_call(api.create_commit, ..., when=rate_limited)``), since after a
    server error the write may have landed.
    """
    return status_of(error) == RATE_LIMITED_STATUS


def upload_resuming(api: Any, *, repo_id: str, folder_path: str | Path, repo_type: str,
                    sleep: Callable[[float], None] = time.sleep, **options: Any) -> None:
    """``api.upload_large_folder(repo_id=..., folder_path=..., repo_type=..., **options)``, run again after it gives up.

    A RuntimeError, which the library raises when it gives up, or a passing HTTP failure (`foundry.retry.transient`)
    starts the next run after `UPLOAD_PAUSE_SECONDS`, up to `UPLOAD_ROUNDS` runs in all. Each run reads the progress the
    last one saved under the folder's `.cache/huggingface/`, so the folder must stay where it is between runs. Any other
    failure, such as a missing folder or a refused token, raises at once.
    """
    for round_number in range(1, UPLOAD_ROUNDS + 1):
        try:
            api.upload_large_folder(repo_id=repo_id, folder_path=folder_path, repo_type=repo_type, **options)
            return
        except Exception as error:  # the library raises RuntimeError when it gives up; a request's own error otherwise
            if round_number == UPLOAD_ROUNDS or not (isinstance(error, RuntimeError) or transient(error)):
                raise
            status = status_of(error)
            print(f"upload round {round_number} of {UPLOAD_ROUNDS} into {repo_id} stopped ({type(error).__name__}"
                  f"{f' {status}' if status is not None else ''}); resuming in {UPLOAD_PAUSE_SECONDS:.0f} s",
                  file=sys.stderr, flush=True)
            sleep(UPLOAD_PAUSE_SECONDS)


__all__ = [
    "HUB_CALL_ATTEMPTS", "HUB_CALL_BACKOFF_SECONDS", "RATE_LIMITED_STATUS", "UPLOAD_PAUSE_SECONDS", "UPLOAD_ROUNDS",
    "hub_call", "rate_limited", "upload_resuming",
]
