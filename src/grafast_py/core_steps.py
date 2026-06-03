"""Core plan-resolver steps for the Grafast plan-then-execute engine.

These are the source-agnostic value/container/batch steps a plan resolver wires
into a DAG. Every step subclasses :class:`grafast_py.step_model.Step` and obeys the
hard contract: ``execute(count, values)`` runs ONCE over a whole bucket and returns
a list of length ``count`` (entry ``i`` is this step's result for bucket position
``i``); ``values[d]`` is the already-computed output column of dependency ``d``.

The decisive members are :class:`LoadOneStep` / :class:`LoadManyStep`: their
``execute`` receives EVERY key in the bucket at once, coalesces duplicates, and
invokes the user batch-load callback EXACTLY ONCE, then scatters the aligned
results back per entry. That single call over N parents is the Grafast batching
payoff (the legacy resolver-adapter path loops per parent instead).

`peer_key` and `dedup_params` feed the planner's cross-step deduplication pass:
two steps with the same class, the same dependency winner ids, the same
``peer_key`` and the same ``dedup_params`` are structurally identical and collapse
to one, so a value loaded/accessed twice is loaded/accessed once.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple

from .step_model import Step

# the marker used to mean "no fallback supplied" so a real ``None`` fallback is
# distinguishable from "unset" in AccessStep.
_NO_FALLBACK = object()


def get_in(value: Any, path: Sequence[Any], fallback: Any = None) -> Any:
    """Walk ``path`` into ``value`` by mapping-key then attribute, like the default resolver.

    Each path segment is resolved against the current value as a mapping key first
    (``value[segment]``), then as an attribute (``getattr(value, segment)``); a
    missing segment or a ``None`` intermediate short-circuits to ``fallback``. This
    mirrors graphql-core's default field resolver (dict-or-attribute) and Grafast's
    ``access`` extraction.
    """
    current = value
    for segment in path:
        if current is None:
            return fallback
        if isinstance(current, dict):
            if segment in current:
                current = current[segment]
                continue
            return fallback
        try:
            current = current[segment]  # sequence index / mapping-like
            continue
        except (TypeError, KeyError, IndexError):
            pass
        try:
            current = getattr(current, segment)
        except (AttributeError, TypeError):
            return fallback
    return current


class ConstantStep(Step):
    """A 0-dependency step producing the same constant for every bucket entry.

    ``execute(count, _)`` returns ``[data] * count``. Two constant steps with equal
    data deduplicate to one.
    """

    is_sync_and_safe = True

    def __init__(self, data: Any) -> None:
        super().__init__()
        self.data = data

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return [self.data] * count

    @property
    def peer_key(self) -> str:
        return f"constant|{type(self.data).__name__}|{self.data!r}"

    def dedup_params(self) -> Tuple[Any, ...]:
        # data identity by value; unhashable data falls back to repr via peer_key
        return (repr(self.data),)


class AccessStep(Step):
    """Lazy (possibly nested) attribute/key access on an upstream step's output.

    One dependency: ``$parent`` (dep 0). ``execute`` extracts ``path`` out of each
    parent value via :func:`get_in`, returning ``fallback`` for missing/None paths.
    ``get(attr)`` / ``at(index)`` grow the chain by returning a NEW access on the
    same parent with the segment appended — matching Grafast's lazy ``.get`` chains.
    """

    is_sync_and_safe = True

    def __init__(
        self, parent: Step, path: Sequence[Any], fallback: Any = _NO_FALLBACK
    ) -> None:
        super().__init__()
        self.parent = parent
        self.path: Tuple[Any, ...] = tuple(path)
        self.fallback = None if fallback is _NO_FALLBACK else fallback
        self.add_dependency(parent)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        parents = values[0]
        path = self.path
        fallback = self.fallback
        # an upstream step that raised carries an Exception as its column entry; pass
        # it straight through rather than masking it as a missing-path fallback, so the
        # field completer still locates the original error.
        return [
            parents[i]
            if isinstance(parents[i], Exception)
            else get_in(parents[i], path, fallback)
            for i in range(count)
        ]

    def get(self, attr: Any) -> "AccessStep":
        """Project deeper: ``access($parent, [...path, attr])``."""
        return AccessStep(self.parent, (*self.path, attr), self.fallback)

    def at(self, index: int) -> "AccessStep":
        """Project a list index: ``access($parent, [...path, index])``."""
        return AccessStep(self.parent, (*self.path, index), self.fallback)

    @property
    def peer_key(self) -> str:
        has_fallback = self.fallback is not None
        return f"access|{has_fallback}|{self.path!r}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (self.path, repr(self.fallback))


class LambdaStep(Step):
    """Maps each bucket entry through a user callable ``fn``.

    One dependency: the input step (use :func:`list_step` to feed several inputs as a
    tuple). ``execute`` runs ``fn`` over the dependency's whole column in one pass.
    If ``fn`` is an async function the per-entry results are coroutines; the planner
    treats lambdas as not sync-and-safe so the executor awaits them. Dedup is by
    ``fn`` identity (a captured closure is unique, so only the same object merges).
    """

    is_sync_and_safe = False

    def __init__(self, dep: Step, fn: Callable[[Any], Any]) -> None:
        super().__init__()
        self.fn = fn
        self.add_dependency(dep)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        fn = self.fn
        col = values[0]
        # propagate a poisoned (Exception) upstream entry untouched; otherwise map fn.
        # a raise inside the user fn for an entry is carried as THAT entry's value
        # (per-entry, mirroring graphql-core's per-resolver catch): the field completer
        # turns the Exception value into a located GraphQLError at the field's path.
        return [self._apply(fn, col[i]) for i in range(count)]

    @staticmethod
    def _apply(fn: Callable[[Any], Any], value: Any) -> Any:
        if isinstance(value, Exception):
            return value
        try:
            return fn(value)
        except Exception as raw_error:  # user mapper raised → carry as a value
            return raw_error

    @property
    def peer_key(self) -> str:
        return f"lambda|{id(self.fn)}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (id(self.fn),)


class ListStep(Step):
    """Bundles N dependency columns into a per-entry tuple/list.

    ``execute`` returns, for each entry ``i``, ``[col_d[i] for each dep d]``. Used to
    feed several upstream steps into one :class:`LambdaStep` or as an explicit list
    construction.
    """

    is_sync_and_safe = True

    def __init__(self, deps: Sequence[Step]) -> None:
        super().__init__()
        for dep in deps:
            self.add_dependency(dep)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        n = len(values)
        return [[values[d][i] for d in range(n)] for i in range(count)]

    @property
    def peer_key(self) -> str:
        return f"list|{self.dependency_count}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (self.dependency_count,)


class ObjectStep(Step):
    """Builds a per-entry dict from a mapping of key -> step.

    One dependency per key, order preserved. ``execute`` returns, for each entry, a
    dict ``{key: col_key[i]}``. ``get(key)`` projects back to the step stored for
    that key (a plan-time projection, like Grafast's ``object().get``).
    """

    is_sync_and_safe = True

    def __init__(self, spec: dict[str, Step]) -> None:
        super().__init__()
        self.keys: List[str] = list(spec.keys())
        self._steps_by_key: dict[str, Step] = dict(spec)
        for key in self.keys:
            self.add_dependency(spec[key])

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = self.keys
        return [{key: values[d][i] for d, key in enumerate(keys)} for i in range(count)]

    def get(self, key: str) -> Step:
        """Return the step stored for ``key`` (plan-time projection)."""
        return self._steps_by_key[key]

    @property
    def peer_key(self) -> str:
        return f"object|{tuple(self.keys)!r}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (tuple(self.keys),)


class EachStep(Step):
    """Maps a list step item-by-item through ``mapper`` (plan-time element plan).

    One dependency: ``$list`` (dep 0), whose per-entry value is a list. ``mapper`` is
    a plan-time callable ``($item_step) -> Step`` describing how to transform ONE
    element; ``EachStep`` realizes it by exploding every parent's list into a single
    flat item bucket, running the mapper's tiny step sub-DAG ONCE over that whole
    bucket (so a ``loadMany`` inside the mapper batches across ALL items of ALL
    parents), then re-grouping per parent.

    Phase A note: the flat-bucket re-execution is driven here directly via
    ``run_steps`` over the mapper's sub-DAG, seeded by an :class:`ItemStep` source.
    """

    is_sync_and_safe = False

    def __init__(self, list_step: Step, mapper: Callable[["ItemStep"], Step]) -> None:
        super().__init__()
        self.add_dependency(list_step)
        self.mapper = mapper

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        from .dag import order_steps
        from .step_model import run_steps

        per_parent_lists = values[0]

        # explode into a flat item bucket, remembering each item's parent index
        flat_items: List[Any] = []
        owners: List[int] = []
        for i in range(count):
            items = per_parent_lists[i] or []
            for item in items:
                flat_items.append(item)
                owners.append(i)

        item_step = ItemStep(flat_items)
        mapped = self.mapper(item_step)

        ordered = order_steps([mapped])
        # the ItemStep is the source for this transient sub-DAG; ensure it is first
        if item_step not in ordered:
            ordered = [item_step, *ordered]

        def regroup(results: dict[int, List[Any]]) -> List[Any]:
            mapped_col = results[mapped.id]
            grouped: List[List[Any]] = [[] for _ in range(count)]
            for k, owner in enumerate(owners):
                grouped[owner].append(mapped_col[k])
            return grouped

        # ItemStep is a non-deterministic source for the executor; run the sub-DAG
        results = run_steps(len(flat_items), ordered, _is_coroutine)
        if _is_coroutine(results):

            async def finish():
                return regroup(await results)

            return finish()
        return regroup(results)

    @property
    def peer_key(self) -> str:
        return f"each|{id(self.mapper)}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (id(self.mapper),)


class ItemStep(Step):
    """A transient 0-dependency source seeded with a flattened item bucket.

    Used only inside :class:`EachStep` to feed exploded list elements into the
    mapper's sub-DAG; ``execute`` returns the item column verbatim.
    """

    is_sync_and_safe = True

    def __init__(self, items: List[Any]) -> None:
        super().__init__()
        self.items = items

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        return self.items


class RootStep(Step):
    """The DAG root: a 0-dependency source whose column is the current bucket.

    Built once at plan time as every root field's ``$parent``. At execute time the
    executor seeds its per-bucket column (the root value, or a child bucket's parent
    objects) before running steps; ``execute`` returns that column verbatim. The
    column is bound per bucket via :meth:`seed`, so one RootStep is reused across
    buckets without being rebuilt.
    """

    is_sync_and_safe = True

    def __init__(self) -> None:
        super().__init__()
        self._column: Optional[List[Any]] = None

    def seed(self, column: List[Any]) -> None:
        """Bind this bucket's parent column for the next ``execute``."""
        self._column = column

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        if self._column is None:
            raise AssertionError("RootStep executed before its bucket was seeded")
        return self._column

    @property
    def peer_key(self) -> str:
        # the root is a unique source; never dedup it away
        return f"root|{id(self)}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (id(self),)


class LoadStep(Step):
    """Shared core of the two batch steps: ONE ``load_fn`` call over the whole bucket.

    One dependency: the per-entry *spec*/key step (dep 0). ``execute`` gathers the
    bucket's key column, coalesces duplicate keys to a single batch entry, and calls
    ``load_fn(unique_keys)`` EXACTLY ONCE — the batching payoff. The returned list
    must be aligned 1:1 with ``unique_keys``; results are scattered back to every
    entry that requested that key. An async ``load_fn`` (returns a coroutine) makes
    ``execute`` return a coroutine resolving to the per-entry column.

    Subclasses fix only the result *shape* contract documented on them; the batch
    machinery is identical. Dedup is by ``load_fn`` identity + the spec step's winner
    id, so two fields loading the same key against the same loader merge to one call.
    """

    is_sync_and_safe = False

    def __init__(self, spec: Step, load_fn: Callable[[List[Any]], Any]) -> None:
        super().__init__()
        self.load_fn = load_fn
        self.add_dependency(spec)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = values[0]

        # coalesce duplicate keys; remember which entries asked for each unique key.
        unique_keys: List[Any] = []
        slot_for_key: dict[Any, int] = {}
        unhashable: List[Tuple[int, Any]] = []
        entry_slot: List[int] = [0] * count
        for i in range(count):
            key = keys[i]
            try:
                slot = slot_for_key.get(key, -1)
            except TypeError:  # unhashable key (e.g. dict spec) — keep distinct
                slot = len(unique_keys)
                unique_keys.append(key)
                entry_slot[i] = slot
                continue
            if slot == -1:
                slot = len(unique_keys)
                slot_for_key[key] = slot
                unique_keys.append(key)
            entry_slot[i] = slot

        # a raise from the user load callback poisons the whole bucket (it is ONE
        # batched call): carry the exception as every entry's value so the completer
        # locates it per field. Engine invariants (e.g. result-length alignment in
        # _scatter) are NOT caught here — they must surface as programming errors.
        try:
            loaded = self.load_fn(unique_keys)
        except Exception as raw_error:
            return [raw_error] * count
        if _is_coroutine(loaded):

            async def finish():
                try:
                    resolved = await loaded
                except Exception as raw_error:
                    return [raw_error] * count
                return self._scatter(resolved, entry_slot, count, len(unique_keys))

            return finish()
        return self._scatter(loaded, entry_slot, count, len(unique_keys))

    def _scatter(
        self, loaded: List[Any], entry_slot: List[int], count: int, n_keys: int
    ) -> List[Any]:
        if len(loaded) != n_keys:
            raise AssertionError(
                f"{type(self).__name__} load_fn returned {len(loaded)} results for"
                f" {n_keys} unique keys"
            )
        return [loaded[entry_slot[i]] for i in range(count)]

    @property
    def peer_key(self) -> str:
        return f"{type(self).__name__}|{id(self.load_fn)}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (id(self.load_fn),)


class LoadOneStep(LoadStep):
    """Batch-load ONE record per key. Each entry's result is a single record.

    ``load_fn(keys)`` returns a list aligned 1:1 with the unique keys, each element
    being that key's record (or None). ``get(attr)`` lazily accesses a column of the
    loaded record via an :class:`AccessStep`.
    """

    def get(self, attr: Any) -> AccessStep:
        return AccessStep(self, (attr,))


class LoadManyStep(LoadStep):
    """Batch-load a LIST of records per key. Each entry's result is a list.

    ``load_fn(keys)`` returns a list aligned 1:1 with the unique keys, each element
    being that key's list of records. ``items()`` returns an :class:`EachStep`-ready
    handle for mapping the per-entry record lists.
    """


def _is_coroutine(value: Any) -> bool:
    """Awaitable predicate used by transient sub-DAG runs (EachStep, loaders)."""
    import inspect

    return inspect.isawaitable(value)


# -------------------------------------------------------------- plan-helper API
# These are the names a plan resolver calls to build a DAG, mirroring Grafast's
# ``constant`` / ``access`` / ``lambda`` / ``list`` / ``object`` / ``each`` /
# ``loadOne`` / ``loadMany`` free functions. Side-effecting verbs are avoided here
# because each merely CONSTRUCTS a step (no execution happens at plan time).


def constant(data: Any) -> ConstantStep:
    return ConstantStep(data)


def access(parent: Step, path: Sequence[Any], fallback: Any = _NO_FALLBACK) -> AccessStep:
    return AccessStep(parent, path, fallback)


def get(parent: Step, key: Any) -> AccessStep:
    """The default-plan projection: ``access($parent, [key])``."""
    return AccessStep(parent, (key,))


def lambda_step(dep: Step, fn: Callable[[Any], Any]) -> LambdaStep:
    return LambdaStep(dep, fn)


def list_step(deps: Sequence[Step]) -> ListStep:
    return ListStep(deps)


def object_step(spec: dict[str, Step]) -> ObjectStep:
    return ObjectStep(spec)


def each(list_step_value: Step, mapper: Callable[[ItemStep], Step]) -> EachStep:
    return EachStep(list_step_value, mapper)


def load_one(spec: Step, load_fn: Callable[[List[Any]], Any]) -> LoadOneStep:
    return LoadOneStep(spec, load_fn)


def load_many(spec: Step, load_fn: Callable[[List[Any]], Any]) -> LoadManyStep:
    return LoadManyStep(spec, load_fn)


__all__ = [
    "ConstantStep",
    "AccessStep",
    "LambdaStep",
    "ListStep",
    "ObjectStep",
    "EachStep",
    "ItemStep",
    "RootStep",
    "LoadStep",
    "LoadOneStep",
    "LoadManyStep",
    "get_in",
    "constant",
    "access",
    "get",
    "lambda_step",
    "list_step",
    "object_step",
    "each",
    "load_one",
    "load_many",
]
