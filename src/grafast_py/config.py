"""Opt-in production hardening config, errors, logging, and tracing hooks.

All controls are OPT-IN with safe defaults that exactly match the engine's
pre-hardening behaviour (off / unbounded), so the drop-in
``execution_context_class=GrafastExecutionContext`` keeps working unchanged. A host
that wants limits sets a :class:`GrafastConfig` on the context class (subclass or
assign ``GrafastExecutionContext.grafast_config``); the context reads
``type(self).grafast_config`` per operation.

The library never configures logging levels itself — the host app does. Messages
follow the project convention: lower-case sentence-style text, short kv fields,
variables in the kv (never f-string-interpolated into the message).
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from graphql.error import GraphQLError

# A module logger. Prefer structlog if the host installed it (so kv pairs render
# structurally); otherwise fall back to the stdlib logger with a tiny shim that
# folds kwargs into the message as ``k=v`` pairs. No hard structlog dependency is
# added to a *library*.
try:
    import structlog

    log = structlog.get_logger("grafast_py")
except ImportError:

    class _KvLogger:
        """Minimal structlog-shaped shim over the stdlib logger.

        Renders ``log.info("msg", count=5)`` as ``msg count=5`` so the project's
        lower-case-sentence + short-kv convention reads the same with or without
        structlog installed.
        """

        def __init__(self) -> None:
            self._log = logging.getLogger("grafast_py")

        def _emit(self, level: int, event: str, **kv: Any) -> None:
            if not self._log.isEnabledFor(level):
                return
            if kv:
                rendered = " ".join(f"{k}={v!r}" for k, v in kv.items())
                self._log.log(level, "%s %s", event, rendered)
            else:
                self._log.log(level, "%s", event)

        def debug(self, event: str, **kv: Any) -> None:
            self._emit(logging.DEBUG, event, **kv)

        def info(self, event: str, **kv: Any) -> None:
            self._emit(logging.INFO, event, **kv)

        def warning(self, event: str, **kv: Any) -> None:
            self._emit(logging.WARNING, event, **kv)

        def error(self, event: str, **kv: Any) -> None:
            self._emit(logging.ERROR, event, **kv)

    log = _KvLogger()


class GrafastTimeoutError(GraphQLError):
    """Raised when an operation exceeds ``execution_timeout_s`` (async path only)."""


class GrafastDepthLimitError(GraphQLError):
    """Raised at plan time when selection-set nesting exceeds ``max_depth``."""


class GrafastCostLimitError(GraphQLError):
    """Raised at plan time when the static cost estimate exceeds ``max_cost``."""


def _noop_span(*args: Any, **kwargs: Any) -> Optional[Any]:
    """Default tracing hook: returns None (no span). Zero overhead."""
    return None


@dataclass
class GrafastConfig:
    """Opt-in hardening knobs for :class:`GrafastExecutionContext`.

    Defaults reproduce the engine's pre-hardening behaviour exactly (no limits, no
    bounded concurrency, no-op tracing), so attaching a default config is a no-op.

    Limits
    ------
    execution_timeout_s
        Wall-clock budget for the ASYNC execution path; on overrun the operation
        raises :class:`GrafastTimeoutError`. The synchronous path has no event loop to
        interrupt, so the timeout does not apply there. It bounds the CALLER but does
        not by itself guarantee in-flight DB statements are cancelled and their
        connections released — pair it with a server-side ``statement_timeout`` (via
        the pg engine's ``connect_args``) for a hard database-side bound.
        ``None`` = unbounded.
    max_depth
        Maximum object-selection nesting depth, checked at PLAN time before any
        resolver runs; over the limit raises :class:`GrafastDepthLimitError`.
        ``None`` = unbounded.
    max_cost
        Budget for a basic STATIC structural cost estimate computed at plan time:
        each field costs ``field_cost``, and a field's subtree cost is multiplied by
        ``list_factor`` for each enclosing list (the standard multiplier heuristic).
        This is intentionally basic — it has no per-argument ``first:``-aware
        weighting. Over budget raises :class:`GrafastCostLimitError`. ``None`` =
        unbounded.
    max_step_concurrency
        Secondary throttle on the bucket executor's sibling-field completion fan-out,
        via an :class:`asyncio.Semaphore`. NOTE: this is **not** the bound on
        concurrent DB round-trips — that is the pg connection pool
        (``pool_size + max_overflow``), which SQLAlchemy enforces by queuing checkouts.
        Bound DB concurrency with the pool; this knob bounds in-engine fan-out.
        ``None`` = unbounded.

    Tracing hooks (no-ops by default)
    ---------------------------------
    Each hook is called with the noted arguments and may return an object usable as
    a context manager (``__enter__``/``__exit__``) — e.g. an OpenTelemetry span — or
    ``None``. The engine enters the returned span around the work and exits it after.
    With the default no-op hooks the cost is a single ``is None`` check.

    on_operation(context, operation) -> span | None
        Around the whole operation execution.
    on_plan(context, operation) -> span | None
        Around plan construction.
    on_step_batch(step, count) -> span | None
        Around each step's per-bucket ``execute`` (the batch boundary).
    """

    execution_timeout_s: Optional[float] = None
    max_depth: Optional[int] = None
    max_cost: Optional[int] = None
    field_cost: int = 1
    list_factor: int = 10
    max_step_concurrency: Optional[int] = None

    on_operation: Callable[..., Optional[Any]] = field(default=_noop_span)
    on_plan: Callable[..., Optional[Any]] = field(default=_noop_span)
    on_step_batch: Callable[..., Optional[Any]] = field(default=_noop_span)


# the default config: every knob off, exactly matching pre-hardening behaviour.
DEFAULT_CONFIG = GrafastConfig()


__all__ = [
    "GrafastConfig",
    "DEFAULT_CONFIG",
    "GrafastTimeoutError",
    "GrafastDepthLimitError",
    "GrafastCostLimitError",
    "log",
]
