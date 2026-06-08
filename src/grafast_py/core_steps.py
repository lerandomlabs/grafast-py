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

import base64
import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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
    # NOT hoistable: ``fn`` is arbitrary host code whose purity the engine cannot verify. Hoisting
    # a request-constant lambda would fire ``fn`` once and fan one value to every child — wrong for
    # an impure ``fn`` (counter / uuid / now). Dedup already assumes ``fn`` is deterministic (it
    # merges by ``fn`` identity), but hoisting affects even a SINGLE-use lambda under a list, so it
    # stays in the child layer (fired per entry) regardless of the hoist default.
    hoistable = False

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

    The flat-bucket re-execution is driven here directly via ``run_steps`` over the
    mapper's sub-DAG, seeded by an :class:`ItemStep` source.
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
    executor binds this bucket's parent column (the root value, or a child bucket's
    parent objects) by seeding ``step.id`` in :func:`run_steps` and excluding the step
    from the ordered list, so ``execute`` is never invoked. One RootStep is reused
    across buckets without being rebuilt or mutated.
    """

    is_sync_and_safe = True

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        raise AssertionError(
            "RootStep is seeded at the bucket boundary and excluded from execution"
        )

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


# ------------------------------------------------------------- Relay global ids
# A Relay global object identifier opaquely names ONE node across the whole schema:
# its type plus the local id(s) that locate the row within that type. We encode it
# the way Grafast / graphql-relay do — a base64 of a compact JSON ``[typename, *id]``
# array — so it stays printable, order-stable, and self-describing (the typename is
# carried inside the id so ``node(id)`` knows which loader to dispatch to). Pure
# stdlib (base64 + json): no value here needs a SQL type, so this is core-agnostic and
# keeps :mod:`core_steps` free of any sqlalchemy import (the hard core invariant).


def encode_global_id(typename: str, *id_parts: Any) -> str:
    """Encode a Relay global id from a typename and its local id part(s).

    The payload is ``[typename, id1, id2, ...]`` JSON-serialised then URL-safe-base64-wrapped
    (the ``-``/``_`` alphabet, so the id is URL-printable without escaping).
    Multiple parts support composite primary keys (the per-column values, in order);
    the single-part case (the common one) is just ``[typename, id]``. The parts must be
    JSON-native (int/str/bool/None) — the same shape a row's key columns carry — so the
    id round-trips losslessly through :func:`decode_global_id`.
    """
    if not id_parts:
        raise ValueError(
            f"cannot encode a global id for {typename!r} with no id parts"
        )
    payload = [typename, *id_parts]
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_global_id(global_id: str) -> Tuple[str, Tuple[Any, ...]]:
    """Decode a Relay global id to ``(typename, id_parts)``; raise LOUDLY on garbage.

    Mirrors :func:`encode_global_id`. A non-string id (``None`` / int / bytes), a malformed
    base64/JSON body, a non-array payload, or a payload missing the typename + at least one
    id part is REJECTED with a ``ValueError`` — there is no silent decode-to-empty, so a
    forged or stale id is surfaced rather than dispatched to the wrong loader. The non-string
    guard is a ``ValueError`` (not the ``AttributeError`` a bare ``.encode()`` would raise) so
    a single non-string entry is carried as THAT entry's per-entry error in
    :meth:`NodeStep.execute` rather than poisoning the whole bucket. Returns the typename and
    the remaining parts as a tuple (a 1-tuple for the common single-column id).
    """
    if not isinstance(global_id, str):
        raise ValueError(
            f"malformed global id {global_id!r}: expected a string, got "
            f"{type(global_id).__name__}"
        )
    try:
        # binascii.Error (bad base64) is a ValueError subclass, so ValueError covers
        # both the decode and the json.loads failure modes here. altchars=-_ matches the
        # URL-safe alphabet encode_global_id emits; validate=True rejects stray characters.
        raw = base64.b64decode(global_id.encode(), altchars=b"-_", validate=True)
        payload = json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError) as err:
        raise ValueError(f"malformed global id {global_id!r}: {err}") from err

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(
            f"malformed global id {global_id!r}: expected [typename, id, ...]"
        )
    typename, *id_parts = payload
    if not isinstance(typename, str):
        raise ValueError(
            f"malformed global id {global_id!r}: typename is not a string"
        )
    return typename, tuple(id_parts)


class NodeStep(Step):
    """Resolve a Relay ``node(id)`` field by decoding the id and dispatching by type.

    One dependency: the per-entry global-id step (dep 0, a string column). At execute
    time every entry's id is decoded to ``(typename, id_parts)`` and routed to the
    loader registered for that typename in ``loaders``; each loader is a batch callback
    with the SAME ``load_fn(keys) -> records`` contract as :class:`LoadOneStep`, so all
    ids for a given type are loaded in ONE call (the Grafast batching payoff carries
    through the node interface). Per type the keys are the decoded id parts — the bare
    value for a single-column id, the tuple for a composite one.

    Routing per type rather than one global loader is what lets ``node`` span many
    resources: a query selecting nodes of several types issues one batched load PER
    type, never one-load-per-id. A typename with no registered loader, or an id whose
    decode fails, is carried as that entry's value (an Exception) so the field completer
    locates it at the node field's path — it never poisons sibling entries of other
    types. An async loader (one that returns a coroutine) makes ``execute`` return a
    coroutine resolving to the per-entry column.
    """

    is_sync_and_safe = False

    def __init__(
        self, id_step: Step, loaders: Dict[str, Callable[[List[Any]], Any]]
    ) -> None:
        super().__init__()
        # snapshot the registry so a later mutation of the caller's dict cannot change
        # which loaders a built plan dispatches to (the plan is immutable once built).
        self.loaders: Dict[str, Callable[[List[Any]], Any]] = dict(loaders)
        self.add_dependency(id_step)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        ids = values[0]

        # decode every entry up front, partitioning the bucket by typename so each
        # type's keys batch into ONE loader call. A poisoned upstream entry or a decode
        # failure is recorded per-entry and excluded from any batch.
        per_entry_error: List[Optional[Exception]] = [None] * count
        keys_by_type: Dict[str, List[Any]] = {}
        entry_type: List[Optional[str]] = [None] * count
        for i in range(count):
            raw = ids[i]
            if isinstance(raw, Exception):
                per_entry_error[i] = raw
                continue
            try:
                typename, id_parts = decode_global_id(raw)
            except ValueError as err:
                per_entry_error[i] = err
                continue
            loader = self.loaders.get(typename)
            if loader is None:
                per_entry_error[i] = KeyError(
                    f"node(id): no loader registered for type {typename!r}"
                )
                continue
            # a single-column id loads under its bare value; a composite id under the
            # tuple — matching the key shape a per-type loader expects.
            key = id_parts[0] if len(id_parts) == 1 else id_parts
            entry_type[i] = typename
            keys_by_type.setdefault(typename, []).append(key)

        # run each type's loader ONCE over its keys; keep results addressable by
        # (typename, key) so they scatter back to every entry that asked for them.
        loaded_by_type: Dict[str, Any] = {}
        pending_types: List[str] = []
        for typename, keys in keys_by_type.items():
            loader = self.loaders[typename]
            try:
                result = loader(keys)
            except Exception as raw_error:  # a type's batch raised → poison ITS entries
                loaded_by_type[typename] = raw_error
                continue
            loaded_by_type[typename] = result
            if _is_coroutine(result):
                pending_types.append(typename)

        if pending_types:

            async def finish():
                for typename in pending_types:
                    try:
                        loaded_by_type[typename] = await loaded_by_type[typename]
                    except Exception as raw_error:
                        loaded_by_type[typename] = raw_error
                return self._scatter_nodes(
                    count, per_entry_error, entry_type,
                    keys_by_type, loaded_by_type,
                )

            return finish()

        return self._scatter_nodes(
            count, per_entry_error, entry_type,
            keys_by_type, loaded_by_type,
        )

    def _scatter_nodes(
        self,
        count: int,
        per_entry_error: List[Optional[Exception]],
        entry_type: List[Optional[str]],
        keys_by_type: Dict[str, List[Any]],
        loaded_by_type: Dict[str, Any],
    ) -> List[Any]:
        """Align each type's batch result back to the entries it was loaded for.

        A type's result is asserted 1:1 with the keys passed to its loader (the
        :class:`LoadStep` alignment contract). Keys are appended in entry order, so the
        Nth entry of a given type takes the Nth result for that type; the running
        ``next_pos`` cursor walks entries in order to reproduce that alignment. An entry
        whose decode/dispatch failed earlier carries its recorded Exception, and an
        entry whose type's batch raised carries that batch error.
        """
        for typename, keys in keys_by_type.items():
            loaded = loaded_by_type[typename]
            if isinstance(loaded, Exception):
                continue  # whole-type failure: handled per entry below
            if len(loaded) != len(keys):
                raise AssertionError(
                    f"node(id) loader for {typename!r} returned {len(loaded)} results"
                    f" for {len(keys)} keys"
                )

        next_pos: Dict[str, int] = {t: 0 for t in keys_by_type}
        out: List[Any] = []
        for i in range(count):
            error = per_entry_error[i]
            if error is not None:
                out.append(error)
                continue
            typename = entry_type[i]
            loaded = loaded_by_type[typename]
            if isinstance(loaded, Exception):
                out.append(loaded)
                continue
            pos = next_pos[typename]
            next_pos[typename] = pos + 1
            out.append(loaded[pos])
        return out

    @property
    def peer_key(self) -> str:
        # the registered loaders fix which rows each id resolves to, so two node steps
        # merge only when they dispatch over the SAME loader set (by identity, like
        # LoadStep keys on load_fn identity). Type order is irrelevant — sort it.
        loader_ids = tuple(
            (typename, id(fn)) for typename, fn in sorted(self.loaders.items())
        )
        return f"node|{loader_ids!r}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return tuple(
            (typename, id(fn)) for typename, fn in sorted(self.loaders.items())
        )


# ---------------------------------------------------------- list-transform steps
# These reshape the per-entry LIST produced by a list-producing step (a loadMany, a
# list_step, an each, or any step whose column entries are lists) WITHOUT a round trip
# to the source: filter/first/last/reverse run in one pass over the already-materialised
# per-entry lists. They mirror Grafast's ``listTransform`` family (``filter`` / ``first``
# / ``last`` / ``reverse``) and stay core-agnostic — the predicate is a plain Python
# callable, never compiled to SQL. A non-list entry (None or an upstream Exception) is
# passed through untouched so a missing list or a poisoned upstream is not masked.


class FilterStep(Step):
    """Keep only the per-entry list items for which ``predicate(item)`` is truthy.

    One dependency: the list-producing step (dep 0). ``execute`` filters each entry's
    list in one pass; a ``None`` entry stays ``None`` and an upstream Exception is
    passed through. A predicate that RAISES for an item is caught PER ENTRY and carried
    as that entry's value (the completer locates it at the field's path), so one bad
    entry never poisons the whole bucket — mirroring :class:`LambdaStep`. Dedup is by
    predicate identity (a captured closure is unique), like :class:`LambdaStep`.
    """

    is_sync_and_safe = True

    def __init__(self, list_step: Step, predicate: Callable[[Any], bool]) -> None:
        super().__init__()
        self.predicate = predicate
        self.add_dependency(list_step)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        predicate = self.predicate
        col = values[0]
        out: List[Any] = []
        for i in range(count):
            entry = col[i]
            if entry is None or isinstance(entry, Exception):
                out.append(entry)
                continue
            try:
                out.append([item for item in entry if predicate(item)])
            except Exception as raw_error:  # predicate raised → carry as THIS entry's value
                out.append(raw_error)
        return out

    @property
    def peer_key(self) -> str:
        return f"filter|{id(self.predicate)}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (id(self.predicate),)


class FirstStep(Step):
    """Take the FIRST item of each entry's list (or ``None`` when empty/absent).

    One dependency: the list-producing step (dep 0). The ``connection.edges.node`` and
    ``edge.node`` plumbing aside, this is the plain ``first`` transform: it reduces a
    list column to a scalar column. Two ``first`` steps over the same list dedup to one.
    """

    is_sync_and_safe = True

    def __init__(self, list_step: Step) -> None:
        super().__init__()
        self.add_dependency(list_step)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        col = values[0]
        out: List[Any] = []
        for i in range(count):
            entry = col[i]
            if isinstance(entry, Exception):
                out.append(entry)
                continue
            out.append(entry[0] if entry else None)
        return out

    @property
    def peer_key(self) -> str:
        return "first"


class LastStep(Step):
    """Take the LAST item of each entry's list (or ``None`` when empty/absent).

    The mirror of :class:`FirstStep`. One dependency: the list-producing step (dep 0).
    """

    is_sync_and_safe = True

    def __init__(self, list_step: Step) -> None:
        super().__init__()
        self.add_dependency(list_step)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        col = values[0]
        out: List[Any] = []
        for i in range(count):
            entry = col[i]
            if isinstance(entry, Exception):
                out.append(entry)
                continue
            out.append(entry[-1] if entry else None)
        return out

    @property
    def peer_key(self) -> str:
        return "last"


class ReverseStep(Step):
    """Reverse each entry's list in place-of-copy (a ``None`` entry stays ``None``).

    One dependency: the list-producing step (dep 0). Returns a NEW reversed list per
    entry (the source column is never mutated). Two ``reverse`` steps over the same list
    dedup to one.
    """

    is_sync_and_safe = True

    def __init__(self, list_step: Step) -> None:
        super().__init__()
        self.add_dependency(list_step)

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        col = values[0]
        out: List[Any] = []
        for i in range(count):
            entry = col[i]
            if entry is None or isinstance(entry, Exception):
                out.append(entry)
                continue
            out.append(list(reversed(entry)))
        return out

    @property
    def peer_key(self) -> str:
        return "reverse"


def _is_coroutine(value: Any) -> bool:
    """Awaitable predicate used by transient sub-DAG runs (EachStep, loaders)."""
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


def node(
    id_step: Step, loaders: Dict[str, Callable[[List[Any]], Any]]
) -> NodeStep:
    """Resolve a Relay ``node(id)`` field via a per-typename loader registry.

    ``id_step`` produces the per-entry global id (string); ``loaders`` maps a typename
    to its batch load callback (the :class:`LoadOneStep` ``load_fn(keys) -> records``
    contract). See :class:`NodeStep`.
    """
    return NodeStep(id_step, loaders)


def filter_step(list_step: Step, predicate: Callable[[Any], bool]) -> FilterStep:
    return FilterStep(list_step, predicate)


def first_step(list_step: Step) -> FirstStep:
    return FirstStep(list_step)


def last_step(list_step: Step) -> LastStep:
    return LastStep(list_step)


def reverse_step(list_step: Step) -> ReverseStep:
    return ReverseStep(list_step)


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
    "NodeStep",
    "FilterStep",
    "FirstStep",
    "LastStep",
    "ReverseStep",
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
    "encode_global_id",
    "decode_global_id",
    "node",
    "filter_step",
    "first_step",
    "last_step",
    "reverse_step",
]
