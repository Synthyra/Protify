"""`@retry`: call a network function again when it fails for a reason that passes.

    @retry
    def fetch(accession: str) -> dict: ...

    @retry(attempts=8, cap_seconds=300)
    def submit(job: Job) -> str: ...

A failure is transient (`transient`) when it carries an HTTP status of 408, 425, 429 or 5xx, or
when the connection itself failed or timed out, in `requests`, `httpx`, `urllib` and the standard
library alike, without importing any of them. Between attempts it waits what the response asks:
a `Retry-After` header, or for a 429 the reset of the spent quota a `RateLimit` header names (the
IETF draft the Hugging Face Hub follows); else `backoff(k)` after attempt `k`, by default a
full-jitter exponential backoff, `uniform(0, base * 2**k)`; all capped at `cap_seconds`. The last
failure is raised as it was. Each retry prints one line to stderr naming
the function, the error's type and its status, never its message, which can hold a URL whose
query carries a key.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import random
import sys
import time

from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import ParamSpec, TypeVar, overload


P = ParamSpec("P")
R = TypeVar("R")

TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
TOO_MANY_REQUESTS = 429  # the status whose `RateLimit` header says when the spent quota resets
# Connection and timeout errors by class name, so no HTTP library is imported: requests and urllib3
# (`ConnectionError`, `Timeout`, `ChunkedEncodingError`, `ProtocolError`), httpx (`NetworkError`,
# `TimeoutException`, `RemoteProtocolError`) and http.client (`RemoteDisconnected`, `IncompleteRead`).
TRANSPORT_ERRORS = frozenset({
    "ConnectionError", "Timeout", "ChunkedEncodingError", "ProtocolError", "NetworkError", "TimeoutException",
    "RemoteProtocolError", "RemoteDisconnected", "IncompleteRead", "TimeoutError",
})


@overload
def retry(function: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def retry(
    *, attempts: int = 5, base_seconds: float = 1.0, cap_seconds: float = 120.0,
    when: Callable[[BaseException], bool] = ..., backoff: Callable[[int], float] = ..., sleep: Callable[[float], None] = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def retry(
    function: Callable[P, R] | None = None,
    /,
    *,
    attempts: int = 5,
    base_seconds: float = 1.0,
    cap_seconds: float = 120.0,
    when: Callable[[BaseException], bool] | None = None,
    backoff: Callable[[int], float] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Retry the decorated function up to `attempts` calls in all, on failures `when` accepts (`transient`)."""
    if attempts < 1:
        raise ValueError(f"attempts counts every call, the first included, so at least 1; received {attempts}.")
    accept = when or transient
    delay = backoff or (lambda attempt: random.uniform(0, base_seconds * 2**attempt))

    def decorate(call: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(call) or inspect.isgeneratorfunction(call) or inspect.isasyncgenfunction(call):
            raise TypeError(f"{call.__qualname__} returns before it runs, so a failure would never reach retry; decorate the call inside it.")

        @functools.wraps(call)
        def retried(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(attempts):
                try:
                    return call(*args, **kwargs)
                except Exception as error:
                    if attempt + 1 == attempts or not accept(error):
                        raise
                    wait = min(cap_seconds, retry_after(error) or delay(attempt))
                    status = status_of(error)
                    print(f"retry: {call.__qualname__} attempt {attempt + 1}/{attempts} failed with {type(error).__name__}"
                          f"{f' {status}' if status is not None else ''}; waiting {wait:.1f} s", file=sys.stderr, flush=True)
                    sleep(wait)
            raise AssertionError("unreachable: the last attempt returns or raises")

        return retried

    return decorate if function is None else decorate(function)


def transient(error: BaseException) -> bool:
    """Whether `error` is worth another attempt: a passing HTTP status, or a failed or timed-out connection."""
    status = status_of(error)
    if status is not None:
        return status in TRANSIENT_STATUSES
    return isinstance(error, (ConnectionError, TimeoutError)) or any(kind.__name__ in TRANSPORT_ERRORS for kind in type(error).__mro__)


def status_of(error: BaseException) -> int | None:
    """The HTTP status an error carries: `response.status_code` (requests, httpx), `response.status`, or `code` (urllib)."""
    response = getattr(error, "response", None)
    for value in (getattr(response, "status_code", None), getattr(response, "status", None), getattr(error, "code", None)):
        if isinstance(value, int) and 100 <= value < 600:
            return value
    return None


def retry_after(error: BaseException) -> float | None:
    """The seconds the error's response asks to wait: its `Retry-After` header, as a number or an HTTP date, else for a
    429 the reset of the spent quota its `RateLimit` header names (`quota_reset`)."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or getattr(error, "headers", None)
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value:
        with contextlib.suppress(ValueError):
            return max(0.0, float(value))
        with contextlib.suppress(TypeError, ValueError):
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
    return quota_reset(headers.get("RateLimit")) if status_of(error) == TOO_MANY_REQUESTS else None


def quota_reset(value: str | None) -> float | None:
    """The seconds until a spent quota resets, from a `RateLimit` header (IETF draft, as the Hugging Face Hub sends it).

    Each comma-separated member names a quota and its parameters, the requests remaining `r` and the seconds to the
    reset `t`, as `"api";r=0;t=55`. The latest reset of the spent quotas, or None when none is spent.
    """
    resets: list[float] = []
    for member in (value or "").split(","):
        fields = {name.strip(): number.strip() for name, _, number in (field.partition("=") for field in member.split(";")[1:])}
        if fields.get("r") == "0" and fields.get("t", "").isdigit():
            resets.append(float(fields["t"]))
    return max(resets, default=None)


__all__ = ["TOO_MANY_REQUESTS", "TRANSIENT_STATUSES", "quota_reset", "retry", "retry_after", "status_of", "transient"]
