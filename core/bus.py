from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine, Type, TypeVar
import logging


from core.events import BaseEvent, SystemError


# Public type alias per plan.md (line 197).
Handler = Callable[[BaseEvent], Coroutine[Any, Any, None]]
E = TypeVar("E", bound=BaseEvent)

# Cap on the dedup cache to bound memory. Events are evicted FIFO
# past this point; collisions on event_id across old/new streams
# are vanishingly rare thanks to uuid4.
_DEDUP_MAX = 5000


class EventBus:
    """Async fire-and-forget pub/sub bus for KANCHA events.

    Typical lifecycle::

        bus = EventBus()
        nlu = NLUClassifier(bus); nlu.register()        # subscribes
        await text_input.run()                          # emits TextInputReceived
        ...
        await bus.close()                               # graceful shutdown

    See module docstring for the full contract.
    """

    def __init__(
        self,
        *,
        history_size: int = 1000,
        default_handler_timeout: float | None = None,
    ) -> None:
        # Typed handlers, keyed by exact event class.
        self._handlers: dict[type[BaseEvent], list[Handler]] = defaultdict(list)
        # Wildcard handlers — invoked for every event regardless of type.
        # Used for debug logging and the ResponseFormatter fan-in.
        self._global_handlers: list[Handler] = []
        # In-flight handler tasks. ``done_callback`` discards finished
        # tasks so this set never grows without bound.
        self._tasks: set[asyncio.Task[Any]] = set()
        # Bounded ring of recently emitted events for introspection.
        self._history: deque[BaseEvent] = deque(maxlen=history_size)
        # Dedup guard — suppresses duplicate event_id emissions.
        self._seen_ids: set[str] = set()
        # Insertion order for FIFO eviction when _seen_ids grows past _DEDUP_MAX.
        self._seen_order: deque[str] = deque()
        # Lifecycle flag — flipped by ``close()``.
        self._closed: bool = False
        # Optional per-handler timeout applied inside ``_run_handler``.
        self._default_handler_timeout = default_handler_timeout
        # Target loop for ``emit_threadsafe``. Captured at construction
        # (which must happen on the loop thread) so that C-thread producers
        # have a reference without needing to call get_event_loop themselves
        # — that call is deprecated in 3.12+ when made from a non-loop thread.
        try:
            self._loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    # ── Subscription ─────────────────────────────────────────────────────

    def subscribe(self, event_type: Type[E], handler: Handler) -> None:
        """Register ``handler`` to receive every event of ``event_type``.

        Raises ``TypeError`` if ``handler`` is not a coroutine function
        and ``RuntimeError`` if the bus is closed. Idempotent — adding
        the same handler twice is a no-op.
        """
        if self._closed:
            raise RuntimeError("EventBus is closed; cannot subscribe")
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(
                f"handler {handler!r} must be a coroutine function "
                f"(define it with `async def`)"
            )
        bucket = self._handlers[event_type]
        if handler not in bucket:
            bucket.append(handler)
            logger.debug(
                "subscribed %s to %s (total=%d)",
                handler.__qualname__,
                event_type.__name__,
                len(bucket),
            )

    def subscribe_all(self, handler: Handler) -> None:
        """Register ``handler`` to receive EVERY event emitted on the bus."""
        if self._closed:
            raise RuntimeError("EventBus is closed; cannot subscribe")
        if not asyncio.iscoroutinefunction(handler):
            raise TypeError(
                f"handler {handler!r} must be a coroutine function "
                f"(define it with `async def`)"
            )
        if handler not in self._global_handlers:
            self._global_handlers.append(handler)
            logger.debug("subscribed %s to ALL events", handler.__qualname__)

    def unsubscribe(self, event_type: Type[BaseEvent], handler: Handler) -> None:
        """Remove a previously subscribed handler. Silent if absent."""
        bucket = self._handlers.get(event_type)
        if bucket and handler in bucket:
            bucket.remove(handler)
            logger.debug("unsubscribed %s from %s", handler.__qualname__, event_type.__name__)
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    def handler_count(self, event_type: Type[BaseEvent] | None = None) -> int:
        """Return the number of handlers for a specific event type, or
        the grand total across all types when ``event_type`` is None."""
        if event_type is None:
            return sum(len(v) for v in self._handlers.values()) + len(self._global_handlers)
        return len(self._handlers.get(event_type, ())) + (
            len(self._global_handlers) if event_type is BaseEvent else 0
        )

    # ── Dispatch ─────────────────────────────────────────────────────────

    def emit(self, event: BaseEvent) -> None:
        """Fire-and-forget dispatch.

        Returns immediately. One asyncio task is created per registered
        handler (typed + global); each task discards itself from the
        in-flight set on completion.

        Duplicate ``event_id`` is suppressed — useful when STT retry
        logic or upstream modules accidentally re-publish the same
        event.
        """
        if self._closed:
            logger.warning("emit() on closed bus — dropping %s", type(event).__name__)
            return

        # Dedup guard.
        if event.event_id in self._seen_ids:
            logger.debug("duplicate event %s suppressed", event.event_id)
            return
        self._seen_ids.add(event.event_id)
        self._seen_order.append(event.event_id)
        if len(self._seen_order) > _DEDUP_MAX:
            evict = self._seen_order.popleft()
            self._seen_ids.discard(evict)

        # Record into history (always, even if no handlers — useful for debugging).
        self._history.append(event)

        # Collect handlers. Typed handlers receive events that match
        # their exact type; global handlers receive every event.
        handlers: list[Handler] = list(self._handlers.get(type(event), ()))
        handlers.extend(self._global_handlers)

        if not handlers:
            return

        for handler in handlers:
            task = asyncio.create_task(
                self._run_handler(handler, event),
                name=f"bus:{type(event).__name__}:{handler.__qualname__}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def emit_and_wait(self, event: BaseEvent) -> None:
        """Like ``emit`` but awaits every handler's completion.

        Used in tests and in bootstrap steps that need to know
        downstream modules have finished before proceeding.
        Handler exceptions are swallowed (``return_exceptions=True``)
        so one bad handler cannot poison the gather.
        """
        if self._closed:
            logger.warning("emit_and_wait() on closed bus — dropping %s", type(event).__name__)
            return

        # Dedup: skip replay, but still need to await any already-in-flight
        # handlers for this id? No — by definition they're gone, so just bail.
        if event.event_id in self._seen_ids:
            return
        self._seen_ids.add(event.event_id)
        self._seen_order.append(event.event_id)
        if len(self._seen_order) > _DEDUP_MAX:
            evict = self._seen_order.popleft()
            self._seen_ids.discard(evict)

        self._history.append(event)

        handlers: list[Handler] = list(self._handlers.get(type(event), ()))
        handlers.extend(self._global_handlers)
        if not handlers:
            return

        tasks = [asyncio.create_task(self._run_handler(h, event)) for h in handlers]
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            for t in tasks:
                self._tasks.discard(t)

    def emit_threadsafe(self, event: BaseEvent) -> None:
        """Dispatch ``event`` from a non-asyncio thread.

        Wake-word and audio callbacks run on C threads; calling
        ``emit()`` directly from there would raise
        ``RuntimeError: no running event loop`` (or worse, race
        the loop's internals). This method schedules the emit onto
        the loop via ``call_soon_threadsafe``.

        The bus captures the running loop the first time this is
        called and reuses it. If the loop has since closed, the
        event is dropped and a warning is logged.
        """
        # Resolve the target loop. ``asyncio.get_event_loop`` is the
        # cross-thread accessor; in 3.12+ we should call it from within
        # the event-loop thread, but ``get_running_loop`` requires the
        # current thread to be the loop thread, which is exactly what
        # the caller is NOT. So we fall back to the policy.
        if self._loop is None or not self._loop.is_running():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                logger.error(
                    "emit_threadsafe() cannot find a target loop; dropping %s",
                    type(event).__name__,
                )
                return
        loop = self._loop
        try:
            loop.call_soon_threadsafe(self.emit, event)
        except RuntimeError as exc:
            # Loop has been closed between capture and call.
            logger.warning(
                "emit_threadsafe() target loop is closed; dropping %s: %s",
                type(event).__name__,
                exc,
            )

    # ── Internal: handler execution with crash isolation ─────────────────

    async def _run_handler(self, handler: Handler, event: BaseEvent) -> None:
        """Invoke one handler with timeout + crash isolation.

        Contract:
        - NEVER re-raises. Every exception is caught, logged, and
          reported via a ``SystemError`` event so the rest of the
          system can react.
        - If the SystemError emission itself fails (e.g. its handler
          also crashes), the failure is logged and swallowed — we
          must not let error reporting cascade.
        """
        try:
            if self._default_handler_timeout is not None:
                await asyncio.wait_for(
                    handler(event), timeout=self._default_handler_timeout
                )
            else:
                await handler(event)
        except asyncio.TimeoutError:
            logger.warning(
                "handler %s timed out after %.2fs on %s",
                handler.__qualname__,
                self._default_handler_timeout or 0.0,
                type(event).__name__,
            )
            await self._report_handler_error(
                handler, event, "handler timeout"
            )
        except asyncio.CancelledError:
            # Cooperative cancellation — re-raise so the task is properly cancelled.
            raise
        except Exception as exc:  # noqa: BLE001 — by contract, we catch everything
            logger.exception(
                "handler %s crashed on %s: %s",
                handler.__qualname__,
                type(event).__name__,
                exc,
            )
            await self._report_handler_error(handler, event, repr(exc))
        except BaseException as exc:
            # SystemExit, KeyboardInterrupt etc. — log and re-raise so
            # the loop can shut down. We don't try to report these.
            logger.exception(
                "handler %s raised BaseException %s on %s",
                handler.__qualname__,
                type(exc).__name__,
                type(event).__name__,
            )
            raise

    async def _report_handler_error(
        self, handler: Handler, event: BaseEvent, message: str
    ) -> None:
        """Best-effort emission of a SystemError for a handler failure.

        The SystemError emission itself can crash; we must not let
        error reporting cascade.
        """
        try:
            err = SystemError(
                source_module=handler.__module__ or handler.__qualname__,
                error_message=f"{type(event).__name__}: {message}",
                recoverable=True,
            )
            # Synchronous, fire-and-forget — we don't want to recurse
            # through emit_and_wait's gather if the loop is shutting down.
            self.emit(err)
        except Exception:  # noqa: BLE001
            logger.exception("failed to report handler error")

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def drain(self) -> None:
        """Wait for all in-flight handler tasks to complete.

        Used during graceful shutdown so the process doesn't exit
        mid-handler. Safe to call multiple times.
        """
        if not self._tasks:
            return
        # Snapshot — the set mutates as tasks complete.
        pending = list(self._tasks)
        await asyncio.gather(*pending, return_exceptions=True)

    async def close(self, timeout: float = 5.0) -> None:
        """Close the bus: reject new subs/emits, then drain.

        After ``close()``, ``subscribe`` raises ``RuntimeError`` and
        ``emit`` becomes a logged no-op. ``drain()`` is bounded by
        ``timeout`` seconds; any tasks still running past that are
        cancelled.
        """
        if self._closed:
            return
        logger.info("EventBus closing (draining %d task(s))", len(self._tasks))
        self._closed = True
        # Invalidate cached loop so post-close emit_threadsafe drops cleanly.
        self._loop = None
        try:
            await asyncio.wait_for(self.drain(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "drain timed out after %.1fs; cancelling %d task(s)",
                timeout,
                len(self._tasks),
            )
            for t in list(self._tasks):
                t.cancel()
            # Give cancelled tasks a chance to unwind.
            await asyncio.gather(*self._tasks, return_exceptions=True)

    # ── Introspection ────────────────────────────────────────────────────

    def history(
        self, event_type: Type[BaseEvent] | None = None, limit: int = 50
    ) -> list[BaseEvent]:
        """Return the most recent ``limit`` events, newest-last.

        If ``event_type`` is given, only events of that exact type
        are returned (subclasses not included — use ``isinstance``
        on the returned events if you need subclass matching).
        """
        if event_type is None:
            return list(self._history)[-limit:]
        return [e for e in self._history if type(e) is event_type][-limit:]

    def stats(self) -> dict[str, Any]:
        """Snapshot of bus state for diagnostics / health endpoints."""
        return {
            "event_types": len(self._handlers),
            "handlers": sum(len(v) for v in self._handlers.values()) + len(self._global_handlers),
            "global_handlers": len(self._global_handlers),
            "inflight_tasks": len(self._tasks),
            "history_size": len(self._history),
            "dedup_cache_size": len(self._seen_ids),
            "closed": self._closed,
        }

    def __repr__(self) -> str:
        return (
            f"<EventBus types={len(self._handlers)} "
            f"handlers={self.handler_count()} "
            f"inflight={len(self._tasks)} "
            f"{'closed' if self._closed else 'open'}>"
        )


__all__ = ["EventBus", "Handler"]
