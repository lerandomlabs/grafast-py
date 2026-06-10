"""Cross-request plan cache: a bounded-LRU process cache of finalized plans.

Planning is per-request today (``plan.plan_operation`` builds the ObjectPlan tree + the
step DAG fresh every request and stashes them on the context). When two requests run the
SAME document, the plan they produce is identical EXCEPT for the per-request values that
plan-time inlining baked into the SQL — and once the host expresses those values as
value-agnostic PLACEHOLDERS, the plan is value-INDEPENDENT and reusable: the
cached SQL is shared and the per-request VALUES are supplied at SQL-RENDER time from a
per-request source map, never stored on the shared step. This module is that reuse layer.

Design (sqlalchemy-free — the core engine never imports the pg stack)
---------------------------------------------------------------------
KEY. ``(id(schema), document-text-hash, operation-name, variable-arg-fingerprint,
config-fingerprint)``:

  * ``id(schema)`` distinguishes two schemas in one process (a host serving several).
  * the document-text hash — the PRINTED-AST text of the operation PLUS every fragment
    definition it can reference (graphql-core's ``ExecutionContext`` keeps the operation node
    and the fragment map but NOT the original ``DocumentNode``, so we reconstruct the relevant
    text from those). NOT ``id(operation)``, which is not stable across re-parses, so two
    requests of the same query text hit the same entry. Printing normalises whitespace and
    aliasing-irrelevant noise to the canonical form, so semantically identical documents
    converge; folding the fragments in keeps two operations that differ only in a referenced
    fragment's body apart.
  * the operation NAME selects one operation from a multi-operation document.
  * the variable-arg fingerprint (the sorted ``(field-path, arg-name)`` pairs that resolved
    from a ``$variable``) is REDUNDANT with the document text — the same text always yields
    the same structure — but folded in for safety/clarity, so a key never collides across
    two structurally different documents that happen to print-hash alike.
  * the config fingerprint (the PLAN-AFFECTING ``GrafastConfig`` fields — ``inline_relations``,
    ``placeholders``, ``cache_plans``, ``hoist``) so two context classes serving the SAME schema under
    DIFFERENT configs, both leaving ``plan_cache=None`` (and so both sharing the process-global
    ``default_cache``), never collide on one entry. Without it a plan built under config A would
    be served to a config-B request on a hit, with that request's own ``args.is_variable``
    branching plan resolver bypassed; a cache entry must be valid for the REQUESTING config, so
    the config the plan was built under is part of its key.

VALUE. The three things ``plan_operation`` stashes on the context: the finalized
``ObjectPlan`` tree, the operation ``RootStep``, and the ``Plan`` (the step DAG). A cache
HIT returns them; the executor seeds + runs them exactly as a freshly-planned operation.

CACHEABILITY. Only a VALUE-INDEPENDENT plan is stored: ``Plan.cacheable`` is True iff no
``$variable`` value was inlined as a plan-time literal (every SQL-affecting variable value
is a same-every-request literal or a source-tagged placeholder). A plan that inlined a
variable as a literal is value-specific and is NEVER cached — reusing it would serve a
later request the earlier request's value. An operation that owns an ABSTRACT (interface /
union) field is likewise NOT cached: its per-concrete-type subtrees are planned LAZILY at
execute time, held on the completer (not in ``plan.steps``), so the per-request value map
never reaches their placeholders — caching such an operation would serve a later request the
first request's subtree value (``plan.owns_abstract_field`` flips ``cacheable`` False; see
``plan_operation``).

SHARED ENTRY ON HIT (deepcopy-free). The cached steps carry NO per-request value: a
``pg_placeholder`` WHERE bind is value-LESS, a pagination ``Placeholder`` is value-LESS, and a
variable-derived cursor is decoded per request at render — never on the shared step. So a cache
HIT stashes the SHARED cached triple DIRECTLY on the context (``context._grafast_plan IS
cached.plan`` — no copy) plus a per-request SOURCE MAP (``"var:<name>"`` ->
``variable_values[name]``, via :func:`values_by_source`). The executor threads that map through
the per-invocation ``BucketExtra.source_values`` channel into each ``wants_extra`` pg step,
which resolves its placeholder/cursor/page values from the map and injects them into the
compiled statement's ``params`` at render — never mutating the shared step. The dedup key is
value-agnostic, so the SQL is shared; only the rendered ``params`` differ per request. This is
what keeps the cache CONCURRENCY-SAFE WITHOUT a per-request copy: two concurrent requests of the
same document with DIFFERENT variables share the identical cached step objects but each carries
its OWN source map on its OWN ``BucketExtra``, so one can never observe the other's value. The
shared cached entry is never mutated, so no request-level serialization (and no deepcopy) is
needed.

EVICTION. A bounded LRU (``max_entries``, default 1000) evicts the least-recently-used
entry so an adversarial stream of unique documents cannot grow the cache without bound.

OPT-IN. The cache is consulted ONLY when ``GrafastConfig.cache_plans`` is on; the default
(off) never touches it, so the engine plans per-request and produces byte-identical output.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple

from graphql.language import (
    FragmentDefinitionNode,
    OperationDefinitionNode,
    VariableNode,
    print_ast,
)


class CachedPlan(NamedTuple):
    """The finalized plan triple a cache entry holds (what ``plan_operation`` stashes).

    ``object_plan`` is the finalized ObjectPlan tree the executor drives; ``root_step`` is
    the operation ``RootStep`` (the root bucket boundary the executor seeds); ``plan`` is the
    step DAG. The triple is SHARED across requests on a HIT (no copy): it carries no
    per-request value, so two concurrent hits reuse the identical objects, each with its OWN
    source map threaded via ``BucketExtra``.

    ``schema`` is the schema the plan was built against, kept so a HIT can verify ``entry.schema
    is request_schema``: ``id(schema)`` (a key component) is REUSED after a schema is
    garbage-collected, so a freed schema's old ``id`` could alias a new schema; the ``is``
    re-check turns such a stale-``id`` collision into a miss (the holder also pins the schema,
    which is fine — schemas are long-lived).

    ``customizer_structures`` is a STORE-time snapshot of every customizer-bearing step's
    value-agnostic predicate shape — the belt-and-suspenders for the cache-hit structural-
    divergence guard. The on-hit guard otherwise re-derives the shape from the SURVIVING
    ``plan.steps``, so a customizer-bearing step that dedup-merged / inlined / tree-shook out of
    ``plan.steps`` between store and hit would escape it and let a later differing-context request
    inherit the planning request's customizer-decided structure. Snapshotting the shape here makes
    the guard independent of which steps survived optimization. The entries are duck-typed (each
    exposes a zero-arg ``still_matches()`` re-resolving its resource customizer against the current
    request), so the sqlalchemy-free core stores + iterates them without importing the pg stack.
    Empty for an operation with no customizer-bearing step (the common case).
    """

    object_plan: Any
    root_step: Any
    plan: Any
    schema: Any = None
    customizer_structures: Tuple[Any, ...] = ()


# the plan-affecting GrafastConfig fields folded into the cache key: a plan built under one
# combination must not be served to a request under another (see config_fingerprint).
ConfigFingerprint = Tuple[bool, bool, bool, bool]

# the cache key: (schema identity, document-text hash, operation name, variable fingerprint,
# config fingerprint).
CacheKey = Tuple[
    int, int, Optional[str], Tuple[Tuple[str, ...], ...], ConfigFingerprint
]


def config_fingerprint(config: Any) -> ConfigFingerprint:
    """The PLAN-AFFECTING ``GrafastConfig`` fields, as a hashable key component.

    Only the fields that change the SHAPE of the planned DAG (and so what a cached plan is
    valid for) belong here: ``inline_relations`` (whether relations fold into LATERAL joins),
    ``placeholders`` (whether variable provenance is threaded, so a resolver placeholders vs
    inlines), ``cache_plans`` (which gates the cache itself), and ``hoist`` (whether the
    cross-parent hoist pass relocates steps, which changes each LayerPlan's run_steps/boundary
    and so the finalized DAG shape). The limit/concurrency/tracing knobs do not change the plan,
    so they are excluded — two configs differing only in those SHARE a cache entry, as they
    should. A ``None`` config is a direct unit-test key and fingerprints as all-False; note this no
    longer equals the DEFAULT config's fingerprint (``hoist`` now defaults True), but the cache path
    never passes ``None`` (it always carries the request's real config), so the distinction is
    confined to direct-key unit tests.
    """
    if config is None:
        return (False, False, False, False)
    return (
        bool(config.inline_relations),
        bool(config.placeholders),
        bool(config.cache_plans),
        bool(config.hoist),
    )


def variable_arg_fingerprint(
    operation: OperationDefinitionNode,
) -> Tuple[Tuple[str, ...], ...]:
    """The sorted set of ``(field-path, arg-name)`` pairs whose value came from a variable.

    Walks the operation's selection AST once and records, for every argument whose value
    node is a :class:`~graphql.language.VariableNode`, the response-path of its field and the
    argument name. This is the SAME provenance ``plan.variable_provenance`` reads per field,
    aggregated over the whole operation, so it captures the literal-vs-``$var`` STRUCTURE of
    the document. It is redundant with the document text (same text => same structure) yet
    folded into the key for safety, so two structurally different operations never share one
    entry on a print-hash coincidence. Pure ``graphql.language``; no execute-internals.
    """
    pairs: set[Tuple[str, ...]] = set()

    def visit(selection_set, path: Tuple[str, ...]) -> None:
        if selection_set is None:
            return
        for selection in selection_set.selections:
            arguments = getattr(selection, "arguments", None)
            name_node = getattr(selection, "name", None)
            field_name = name_node.value if name_node is not None else ""
            field_path = (*path, field_name)
            if arguments:
                for arg in arguments:
                    if isinstance(arg.value, VariableNode):
                        pairs.add((*field_path, arg.name.value))
            visit(getattr(selection, "selection_set", None), field_path)

    visit(operation.selection_set, ())
    return tuple(sorted(pairs))


def document_text(
    operation: OperationDefinitionNode,
    fragments: Optional[Mapping[str, FragmentDefinitionNode]],
) -> str:
    """The canonical printed text of an operation plus its (sorted) fragment definitions.

    graphql-core's ``ExecutionContext`` keeps the operation node and the fragment map but
    discards the original ``DocumentNode``, so we reconstruct the planning-relevant document
    text from those. Fragments are printed in name order so the text is order-stable
    regardless of dict insertion order, and folded in so two operations differing only in a
    referenced fragment's body produce different text (and so different keys).
    """
    parts = [print_ast(operation)]
    if fragments:
        parts.extend(print_ast(fragments[name]) for name in sorted(fragments))
    return "\n".join(parts)


def compute_cache_key(
    schema: Any,
    operation: OperationDefinitionNode,
    fragments: Optional[Mapping[str, FragmentDefinitionNode]] = None,
    config: Any = None,
) -> CacheKey:
    """The bounded-LRU key for one operation (with its fragments) under one schema + config.

    ``id(schema)`` keys the schema instance; ``hash(document_text(...))`` keys the operation +
    fragments by their canonical printed text (stable across re-parses, unlike ``id``); the
    operation name selects one of a multi-operation document; the variable fingerprint pins
    the literal-vs-``$var`` structure; the config fingerprint pins the PLAN-AFFECTING config so
    two configs sharing the default cache never collide (see :func:`config_fingerprint`).
    Computed ONCE per request when caching is enabled.
    """
    op_name = operation.name.value if operation.name else None
    return (
        id(schema),
        hash(document_text(operation, fragments)),
        op_name,
        variable_arg_fingerprint(operation),
        config_fingerprint(config),
    )


class PlanCache:
    """A bounded-LRU process cache of finalized plans, keyed by :func:`compute_cache_key`.

    Backed by an :class:`~collections.OrderedDict` under a lock: a GET moves the entry to the
    most-recently-used end; a PUT past ``max_entries`` evicts the least-recently-used end. The
    lock makes GET/PUT safe under the concurrent request fan-out (the entries themselves are
    immutable plan trees — the per-request VALUES live only in the per-request source map
    rendered into ``params``, never on the cached object — so two requests can share one entry
    without racing on its contents).
    """

    def __init__(self, max_entries: int = 1000) -> None:
        if max_entries < 1:
            raise ValueError(
                f"PlanCache max_entries must be >= 1, got {max_entries}"
            )
        self.max_entries = max_entries
        self._entries: "OrderedDict[CacheKey, CachedPlan]" = OrderedDict()
        self._lock = Lock()
        # observability counters (read by tests / a host metrics hook); not load-bearing.
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: CacheKey) -> Optional[CachedPlan]:
        """Return the cached plan for ``key`` (marking it most-recently-used), or ``None``."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry

    def put(self, key: CacheKey, value: CachedPlan) -> None:
        """Store ``value`` under ``key`` as most-recently-used; evict the LRU if over cap."""
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        """Drop every entry (used by tests; a host rarely needs it)."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# The process-global default cache, used when ``GrafastConfig.plan_cache`` is left None and
# ``cache_plans`` is on. A host that wants its OWN bound (or to share/inspect one across
# context classes) sets ``GrafastConfig(plan_cache=PlanCache(max_entries=...))``; otherwise
# every cache_plans-on operation in the process shares this one. Created lazily so importing
# the module allocates nothing.
_DEFAULT_CACHE: Optional[PlanCache] = None


def default_cache() -> PlanCache:
    """Return the process-global default :class:`PlanCache`, creating it on first use."""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = PlanCache()
    return _DEFAULT_CACHE


def values_by_source(
    variable_values: Optional[Mapping[str, Any]],
    operation: Optional[OperationDefinitionNode] = None,
) -> Dict[str, Any]:
    """Map each operation variable to its placeholder SOURCE tag -> value.

    A placeholder's source tag is ``"var:<variable_name>"`` (see ``FieldArgs.source`` /
    ``pg_placeholder``); a pg step resolves its placeholders off it, so translate the request's
    ``{variable_name: value}`` into ``{"var:<variable_name>": value}``. This is the per-request
    SOURCE MAP threaded via ``BucketExtra.source_values`` and rendered into the compiled
    statement's ``params`` (the deepcopy-free hit path) — never bound onto a shared step.

    When ``operation`` is given, EVERY declared variable is included — a variable OMITTED this
    request (absent from ``variable_values`` with no default graphql-core folded in) maps to
    ``None``, so a cache HIT resolves it to ``None`` rather than leaving the PRIOR request's
    value stale (the omitted-no-default correctness gap). graphql-core already folds a
    variable's DEFAULT into ``variable_values`` when omitted, so a defaulted variable carries
    its default here. Without ``operation`` (a direct map in a unit test) only the supplied
    values are mapped.
    """
    mapping: Dict[str, Any] = {
        f"var:{name}": value for name, value in (variable_values or {}).items()
    }
    if operation is not None and operation.variable_definitions:
        for definition in operation.variable_definitions:
            mapping.setdefault(f"var:{definition.variable.name.value}", None)
    return mapping


__all__ = [
    "CachedPlan",
    "CacheKey",
    "ConfigFingerprint",
    "PlanCache",
    "compute_cache_key",
    "config_fingerprint",
    "document_text",
    "variable_arg_fingerprint",
    "values_by_source",
    "default_cache",
]
