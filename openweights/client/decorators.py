import functools
import random
import time
from typing import Optional

import backoff
import httpx
import openai

# Optional: postgrest is used under the hood by supabase-py; handle if present
try:
    import postgrest
except Exception:  # pragma: no cover
    postgrest = None


def _is_transient_http_status(status: int) -> bool:
    # Retry on server errors & rate limiting
    return status >= 500 or status == 429


def _is_transient(exc: BaseException) -> bool:
    """
    Returns True for errors that are likely to be temporary network/service hiccups:
      - httpx timeouts / connect errors / protocol errors
      - HTTPStatusError with 5xx or 429
      - postgrest.APIError with 5xx or 429 (if postgrest available)
    """
    # httpx family (network-ish)
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    # httpx raised because .raise_for_status() was called
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return _is_transient_http_status(exc.response.status_code)
        except Exception:
            return False

    # postgrest API errors (supabase-py)
    if postgrest is not None and isinstance(exc, postgrest.APIError):
        try:
            code = getattr(exc, "code", None)
            # code may be a string; try to coerce
            code_int = int(code) if code is not None else None
            return code_int is not None and _is_transient_http_status(code_int)
        except Exception:
            return False

    # Sometimes libraries wrap the real error; walk the causal chain
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause and cause is not exc:
        return _is_transient(cause)

    return False


def openai_retry(
    *,
    # Exponential mode (default)
    factor: float = 1.5,
    max_value: int = 60,
    # Constant mode (set interval to enable)
    interval: Optional[float] = None,
    max_time: Optional[float] = None,
    # Common
    max_tries: int = 10,
):
    """
    Retry transient OpenAI API errors with backoff + jitter.

    Modes:
      • Exponential (default): pass `factor`, `max_value`, `max_tries`
      • Constant: pass `interval` (seconds) and optionally `max_time`, `max_tries`

    Examples:
        @openai_retry()  # exponential (default)
        def call(...): ...

        @openai_retry(interval=10, max_time=3600, max_tries=3600)  # constant
        def call(...): ...

    Retries on:
      - openai.RateLimitError
      - openai.APIConnectionError
      - openai.APITimeoutError
      - openai.InternalServerError
    """
    exceptions = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )

    def _decorator(fn):
        if interval is not None:
            # Constant backoff mode
            decorated = backoff.on_exception(
                wait_gen=backoff.constant,
                exception=exceptions,
                interval=interval,
                max_time=max_time,  # total wall-clock cap (optional)
                max_tries=max_tries,  # total attempts cap
                jitter=backoff.full_jitter,
                logger=None,  # stay quiet
            )(fn)
        else:
            # Exponential backoff mode
            decorated = backoff.on_exception(
                wait_gen=backoff.expo,
                exception=exceptions,
                factor=factor,  # growth factor
                max_value=max_value,  # cap per-wait
                max_tries=max_tries,  # total attempts cap
                jitter=backoff.full_jitter,
                logger=None,  # stay quiet
            )(fn)

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return decorated(*args, **kwargs)

        return inner

    return _decorator


# sentinel to indicate "raise on exhaustion"
_RAISE = object()


def supabase_retry(
    max_time: float = 60,
    max_tries: int = 8,
    *,
    base: float = 1.0,  # initial delay
    factor: float = 2.0,  # exponential growth
    max_delay: float = 60.0,  # cap for each delay step
    return_on_exhaustion=_RAISE,  # e.g., set to None to "ignore" after retries
):
    """
    Retries ONLY transient Supabase/http errors (see _is_transient) with exponential backoff + full jitter.
    If `return_on_exhaustion` is not `_RAISE`, return that value after retry budget is exhausted for a
    transient error. Non-transient errors still raise immediately.

    Args:
        max_time: maximum total wall-clock seconds spent retrying
        max_tries: maximum attempts (including the first)
        base: initial backoff delay (seconds)
        factor: exponential growth factor per attempt (>= 1)
        max_delay: max per-attempt delay (seconds)
        return_on_exhaustion: value to return after exhausting retries on a transient error.
                              Leave as `_RAISE` to re-raise instead.
    """

    def _next_delay(attempt: int) -> float:
        # attempt starts at 1 for the first retry (after one failure)
        raw = base * (factor ** (attempt - 1))
        return min(raw, max_delay) * random.random()  # full jitter

    def _decorator(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            # quick path: try once
            start = time.monotonic()
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    # Non-transient? bubble up immediately.
                    if not _is_transient(exc):
                        raise

                    attempt += 1
                    # Have we exhausted attempts?
                    if attempt >= max_tries:
                        if return_on_exhaustion is _RAISE:
                            raise
                        return return_on_exhaustion

                    # Compute delay with jitter, ensure we don't break max_time
                    delay = _next_delay(attempt)
                    if max_time is not None:
                        elapsed = time.monotonic() - start
                        remaining = max_time - elapsed
                        if remaining <= 0:
                            if return_on_exhaustion is _RAISE:
                                raise
                            return return_on_exhaustion
                        # don't sleep past the deadline
                        delay = min(delay, max(0.0, remaining))

                    time.sleep(delay)

        return inner

    return _decorator
