from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from typing import AsyncIterator
from dotenv import load_dotenv

load_dotenv()


# ── SDK import ────────────────────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError:
    sys.exit("google-genai not installed.\nRun: pip install google-genai\n")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
DEFAULT_FALLBACKS = os.getenv(
    "GEMINI_MODEL_FALLBACKS",
    "gemini-2.5-flash-lite,gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
)
DEFAULT_HEDGE      = int(os.getenv("HEDGE", "2"))
REQUEST_TIMEOUT    = float(os.getenv("REQUEST_TIMEOUT", "12.0"))
MAX_RETRIES        = int(os.getenv("MAX_RETRIES", "1"))
RETRY_BASE_DELAY   = 0.5  
MAX_RETRY_WAIT     = float(os.getenv("MAX_RETRY_WAIT", "3.0"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def _get_client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "Missing API key.\n"
            "Set GEMINI_API_KEY and re-run.\n"
            "Example: export GEMINI_API_KEY='YOUR_KEY_HERE'\n"
        )
    return genai.Client(api_key=key)


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _parse_retry_after(exc: Exception) -> float | None:
    """
    Extract the server-suggested retry delay from a 429 body.
    The Gemini API includes 'retryDelay: "17s"' in the error details.
    Returns seconds as float, or None if not found.
    """
    msg = str(exc)
    # Matches: 'retryDelay': '17s'  or  "retryDelay": "17.5s"
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", msg)
    if match:
        return float(match.group(1))
    return None


def _is_retryable(exc: Exception) -> bool:
    """True for transient errors — but caller must check retry wait time."""
    if isinstance(exc, genai_errors.APIError):
        return exc.code in (429, 503)
    msg = str(exc).lower()
    return any(t in msg for t in ("429", "503", "resource_exhausted", "unavailable"))


def _is_quota_exhausted(exc: Exception) -> bool:
    """
    True when the quota is actually exhausted (daily/total limit = 0).
    These won't recover within any reasonable retry window.
    Distinct from a momentary rate-limit that clears in <2s.
    """
    msg = str(exc)
    return "limit: 0" in msg or "GenerateRequestsPerDay" in msg


def _is_not_found(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.APIError):
        return exc.code == 404
    msg = str(exc).lower()
    return any(t in msg for t in ("404", "not_found", "not found"))


def _should_retry(exc: Exception, attempt: int) -> tuple[bool, float]:
    """
    Returns (should_retry, wait_seconds).
    Skips the model immediately if:
      - quota is fully exhausted (limit: 0)
      - server says retry in > MAX_RETRY_WAIT seconds
      - already at max retries
    """
    if attempt >= MAX_RETRIES:
        return False, 0.0
    if not _is_retryable(exc):
        return False, 0.0
    if _is_quota_exhausted(exc):
        # Daily quota gone — retrying won't help, move on immediately
        return False, 0.0

    server_wait = _parse_retry_after(exc)
    if server_wait is not None and server_wait > MAX_RETRY_WAIT:
        _log(f"  ↳ server says retry in {server_wait:.0f}s > limit {MAX_RETRY_WAIT}s — skipping model")
        return False, 0.0

    # Use server hint if available and short, otherwise exponential back-off
    wait = server_wait if (server_wait and server_wait <= MAX_RETRY_WAIT) \
           else RETRY_BASE_DELAY * (2 ** attempt)
    return True, wait


# ── Core async primitives ─────────────────────────────────────────────────────

async def _generate_one(
    client: genai.Client,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=model,
                    contents=prompt,
                ),
                timeout=timeout,
            )
            return getattr(resp, "text", "") or ""

        except asyncio.TimeoutError as exc:
            _log(f"[{model}] timeout after {timeout}s")
            raise  # propagate — don't swallow

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_not_found(exc):
                _log(f"[{model}] not found — skipping")
                raise
            do_retry, wait = _should_retry(exc, attempt)
            if do_retry:
                _log(f"[{model}] retryable, back-off {wait:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
                continue
            raise

    raise last_exc  # type: ignore[misc]


async def _stream_one(
    client: genai.Client,
    model: str,
    prompt: str,
    timeout: float,
) -> AsyncIterator[str]:
    """
    Async generator: yields text chunks from a streaming response.
    Applies timeout to the first chunk only (TTFT guard).
    Raises on all errors — never silently returns empty.
    """
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None | Exception] = asyncio.Queue()

        def _producer() -> None:
            try:
                response = client.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                )
                for chunk in response:
                    text = getattr(chunk, "text", "")
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
                loop.call_soon_threadsafe(queue.put_nowait, None)  # clean sentinel
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        asyncio.ensure_future(asyncio.to_thread(_producer))

        try:
            first_chunk = True
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=timeout if first_chunk else None,
                    )
                except asyncio.TimeoutError:
                    _log(f"[{model}] stream TTFT timeout after {timeout}s")
                    raise  # propagate — caller needs to know this failed

                if item is None:
                    return  # clean end of stream
                if isinstance(item, Exception):
                    raise item
                first_chunk = False
                yield item

            return  # unreachable but explicit

        except asyncio.TimeoutError:
            raise  # never swallow timeouts

        except StopAsyncIteration:
            return

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_not_found(exc):
                _log(f"[{model}] not found — skipping")
                raise
            do_retry, wait = _should_retry(exc, attempt)
            if do_retry:
                _log(f"[{model}] retryable, back-off {wait:.1f}s")
                await asyncio.sleep(wait)
                continue
            raise

    if last_exc:
        raise last_exc


# ── Hedged request ────────────────────────────────────────────────────────────

async def hedged_generate(
    client: genai.Client,
    models: list[str],
    prompt: str,
    hedge_width: int,
    timeout: float,
) -> str:
    if not models:
        raise ValueError("No models provided")

    hedged = models[:hedge_width]
    tail   = models[hedge_width:]

    tasks: dict[asyncio.Task, str] = {
        asyncio.create_task(_generate_one(client, m, prompt, timeout), name=m): m
        for m in hedged
    }

    while tasks:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            model_name = tasks.pop(task)
            if task.exception() is None:
                for t in tasks:
                    t.cancel()
                _log(f"[{model_name}] won the hedge race ✓")
                return task.result()
            else:
                _log(f"[{model_name}] hedge failed: {task.exception()}")

    for model in tail:
        try:
            _log(f"[{model}] trying as sequential fallback")
            return await _generate_one(client, model, prompt, timeout)
        except Exception as exc:  # noqa: BLE001
            _log(f"[{model}] failed: {exc}")

    raise RuntimeError("All models failed — no response available")


async def hedged_stream(
    client: genai.Client,
    models: list[str],
    prompt: str,
    hedge_width: int,
    timeout: float,
) -> None:
    if not models:
        raise ValueError("No models provided")

    hedged = models[:hedge_width]
    tail   = models[hedge_width:]
    start  = time.perf_counter()

    async def _race_first_chunk(model: str) -> tuple[str, str, AsyncIterator[str]]:
        gen = _stream_one(client, model, prompt, timeout)
        first = await gen.__anext__()  # raises on any error including timeout
        return model, first, gen

    tasks: dict[asyncio.Task, str] = {
        asyncio.create_task(_race_first_chunk(m), name=m): m
        for m in hedged
    }

    winner_model: str | None = None
    winner_first: str | None = None
    winner_gen: AsyncIterator[str] | None = None

    while tasks and winner_model is None:
        done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            model_name = tasks.pop(task)
            if task.exception() is None:
                winner_model, winner_first, winner_gen = task.result()
                for t in tasks:
                    t.cancel()
                break
            else:
                _log(f"[{model_name}] stream race failed: {task.exception()!r}")

    # Sequential tail fallbacks
    if winner_model is None:
        for model in tail:
            try:
                _log(f"[{model}] trying as sequential fallback")
                gen = _stream_one(client, model, prompt, timeout)
                winner_first = await gen.__anext__()
                winner_model = model
                winner_gen   = gen
                break
            except Exception as exc:  # noqa: BLE001
                _log(f"[{model}] tail stream failed: {exc!r}")

    if winner_model is None:
        raise RuntimeError(
            "All models failed.\n"
            "Likely cause: free-tier quota exhausted on all models. "
            "Check https://ai.dev/rate-limit"
        )

    ttft = time.perf_counter() - start
    _log(f"[{winner_model}] streaming | TTFT {ttft:.2f}s")

    print(winner_first, end="", flush=True)
    async for chunk in winner_gen:  # type: ignore[union-attr]
        print(chunk, end="", flush=True)
    print(flush=True)

    _log(f"[Done {time.perf_counter() - start:.2f}s]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blazing-fast Gemini client with hedged parallel requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prompt",      nargs="?", default=None)
    p.add_argument("--model",     default=DEFAULT_MODEL)
    p.add_argument("--fallbacks", default=DEFAULT_FALLBACKS,
                   help="Comma-separated fallback models")
    p.add_argument("--hedge",     type=int, default=DEFAULT_HEDGE,
                   help="Models to race in parallel (default: 2)")
    p.add_argument("--timeout",   type=float, default=REQUEST_TIMEOUT,
                   help="Seconds before abandoning a model attempt (default: 12)")
    p.add_argument("--stream",    action="store_true",
                   default=os.getenv("STREAM", "0") == "1")
    p.add_argument("--interactive", action="store_true",
                   default=os.getenv("INTERACTIVE", "0") == "1")
    return p.parse_args()


async def run_prompt(
    client: genai.Client,
    models: list[str],
    prompt: str,
    args: argparse.Namespace,
) -> None:
    if args.stream:
        await hedged_stream(client, models, prompt, args.hedge, args.timeout)
    else:
        text = await hedged_generate(client, models, prompt, args.hedge, args.timeout)
        print(text, end="\n" if text and not text.endswith("\n") else "")


async def main() -> None:
    args   = parse_args()
    client = _get_client()

    seen: set[str] = set()
    models: list[str] = []
    for m in [args.model] + _split_models(args.fallbacks):
        if m not in seen:
            seen.add(m)
            models.append(m)

    _log(f"Model order: {models}")
    _log(f"Hedge width: {args.hedge} | Timeout: {args.timeout}s | Stream: {args.stream}")

    if args.interactive:
        if args.prompt:
            await run_prompt(client, models, args.prompt, args)
        while True:
            try:
                prompt = input("\nPrompt> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not prompt:
                continue
            await run_prompt(client, models, prompt, args)
    else:
        prompt = args.prompt or "Tell me a joke about cats."
        await run_prompt(client, models, prompt, args)


if __name__ == "__main__":
    asyncio.run(main())