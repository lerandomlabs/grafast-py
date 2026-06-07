"""The @defer / @stream incremental-delivery driver (graphql-core 3.3 only).

P7's OWNED payload-assembly engine. It emits the graphql-core 3.3 incremental wire
protocol (the ``pending`` / ``incremental`` / ``completed`` shape) by constructing ONLY
the user-facing result classes (``InitialIncrementalExecutionResult`` and friends) via
their public ``__init__`` — it does NOT couple to 3.3's internal
``incremental_publisher`` / ``incremental_graph`` record graph. Instead it REPLICATES that
machinery observably with its own record graph + publisher:

* a deferred fragment is one or more EXECUTION GROUPS — grouped-field-sets executed at an
  absolute path, keyed by a defer-usage-set. Two overlapping @defer fragments that select the
  same field share ONE group, so the field is delivered once (the dedup); a group whose data
  sits deeper than its fragment's path is emitted with a ``subPath`` (the best-id / max-length
  pick, mirroring ``incremental_publisher._get_best_id_and_sub_path``).
* a nested @defer's pending is released only when its parent fragment completes
  (``incremental_graph.complete_deferred_fragment`` → ``_promote_non_empty_to_root``).
* a @stream'd list streams its items as ``IncrementalStreamResult`` batches: a sync list (or
  list of awaitables) and an async iterator alike are drained item-by-item, accumulating
  synchronously-available items into ONE batch flushed at the next async boundary, with the
  iterator-exhausted sentinel flushing the tail then emitting the terminal ``completed`` —
  a transcription of ``incremental_graph._on_stream_items``.

The driver runs each grouped-field-set through the engine's own ``execute_object_plan`` (P2.5
detachment), so a ``loadMany`` inside a deferred fragment over N parents still fires once.

The whole module is import-safe on 3.2 (the result classes are imported lazily inside the
functions), but it is only ever reached on 3.3 (the entry gates on
``_compat.supports_incremental()``).
"""

from asyncio import (
    FIRST_COMPLETED,
    Future,
    ensure_future,
    get_running_loop,
    isfuture,
    sleep,
    wait,
)
from collections import deque
from typing import Any, Dict, List, Optional

from graphql.error import GraphQLError, located_error
from graphql.pyutils import Undefined

from . import _compat
from .bubble import Bubble
from .config import log


# ---------------------------------------------------------------------------
# record graph nodes (driver-owned; NOT upstream's internal records)
# ---------------------------------------------------------------------------


class DeferRecord:
    """A deferred fragment record: one @defer'd group instance at a path.

    Mirrors upstream ``DeferredFragmentRecord``: ``path`` is the absolute path the @defer
    hangs off, ``label`` its @defer label, ``parent`` the enclosing deferred fragment record
    (None at the operation/abstract root). ``children`` are nested defers/streams revealed when
    this fragment completes (the promote-on-complete). ``pending_groups`` are the execution
    groups still running for this fragment; ``successful_groups`` the completed ones. ``id`` is
    assigned at pending-emission time.
    """

    __slots__ = (
        "path",
        "label",
        "parent",
        "children",
        "pending_groups",
        "successful_groups",
        "id",
    )

    def __init__(self, path, label, parent):
        self.path = path
        self.label = label
        self.parent = parent
        self.children: Dict[Any, None] = {}
        self.pending_groups: Dict[Any, None] = {}
        self.successful_groups: Dict[Any, None] = {}
        self.id: Optional[str] = None


class ExecGroup:
    """A pending execution group: one grouped-field-set to run at a path for some fragments.

    Mirrors upstream ``PendingExecutionGroup`` + its boxed result. ``defer_records`` are the
    deferred fragment records this group belongs to (a group shared by two overlapping
    fragments lists both). ``runner`` is a 0-arg callable producing the group's data SYNC or as
    a coroutine. ``result`` is the completed group (a :class:`GroupResult`) once run. ``spec``
    carries ``(context, parent, path, parent_type, field_map, defer_usage_set, registry)`` so
    same-field_map groups promoted together can be BATCHED into one ``execute_object_plan`` over
    all their parents — the grafast guarantee that a deferred relation's loadMany fires once.
    """

    __slots__ = ("defer_records", "runner", "result", "spec")

    def __init__(self, defer_records, runner):
        self.defer_records = defer_records
        self.runner = runner
        self.result: Optional[Any] = None
        self.spec: Optional[tuple] = None


class GroupResult:
    """A completed execution group's result: ``path`` (absolute), data, errors, children.

    ``failed`` marks an unrecoverable null-bubble at the group's top boundary (the fragment is
    completed-with-errors and emits no incremental entry). ``children`` are the nested
    execution-group records + stream records revealed while running this group.
    """

    __slots__ = ("group", "path", "data", "errors", "failed", "children")

    def __init__(self, group, path, data, errors, failed, children):
        self.group = group
        self.path = path
        self.data = data
        self.errors = errors
        self.failed = failed
        self.children = children


class StreamRec:
    """A @stream'd list field's record: a live item producer rooted at the list field path.

    ``producer`` is a :class:`StreamItemProducer` yielding one item-result at a time (sync
    list, list of awaitables, or async iterator, uniformly). ``early_return`` is the async
    iterator's ``aclose`` (or None) so a closed subsequent generator can finish it.
    """

    __slots__ = ("path", "label", "producer", "early_return", "id")

    def __init__(self, path, label, producer, early_return):
        self.path = path
        self.label = label
        self.producer = producer
        self.early_return = early_return
        self.id: Optional[str] = None


# ---------------------------------------------------------------------------
# the incremental graph (a transcription of upstream IncrementalGraph)
# ---------------------------------------------------------------------------


class IncGraph:
    """Server-side record graph + completed-result queue for one incremental response.

    A faithful transcription of graphql-core 3.3's ``IncrementalGraph`` (kept observable, no
    import of the internal class): ``_root_nodes`` is the still-pending set (``has_next``);
    ``_completed_queue`` holds synchronously-ready completed records; ``_next_queue`` holds
    futures resolved when the next async record completes. Stream item-pumps run as tasks.
    """

    def __init__(self, context):
        self._context = context
        self._root_nodes: Dict[Any, None] = {}
        self._completed_queue: List[Any] = []
        self._next_queue: List[Future] = []
        self._tasks: set = set()
        # execution groups collected during a promotion wave, flushed batched by spec so a
        # deferred relation's loadMany over N parents fires once (the grafast guarantee).
        self._pending_runs: List[ExecGroup] = []

    # -- root-node promotion -------------------------------------------------

    def get_new_root_nodes(self, incremental_records):
        """Promote new top-level records (initial result children) to root nodes."""
        initial_children: Dict[Any, None] = {}
        self._add_incremental_records(incremental_records, None, initial_children)
        new_root_nodes = self._promote_non_empty_to_root(initial_children)
        self._flush_runs()
        return new_root_nodes

    def add_completed_group(self, result: GroupResult):
        """Record a successful execution group's children + bookkeeping (upstream parity).

        Keys ``successful_groups`` by the GroupResult (carrying both data + its ExecGroup), so
        the publisher reads the data and the best-id pick reads the group's defer records.
        """
        group = result.group
        for record in group.defer_records:
            record.pending_groups.pop(group, None)
            record.successful_groups[result] = None
        if result.children:
            self._add_incremental_records(result.children, group.defer_records)
            # a child group that completes an already-root record is collected for run here; flush
            # it (batched by spec) so it actually executes and enqueues.
            self._flush_runs()

    def complete_deferred_fragment(self, record: DeferRecord):
        """Release a deferred fragment once its execution groups are all done.

        Returns ``(new_root_nodes, successful_results)`` or None if not yet ready (still has
        pending groups, or already removed). Promotes the fragment's children (nested defers /
        streams) to root nodes — the promote-on-complete the wire ordering depends on.
        """
        if record not in self._root_nodes or record.pending_groups:
            return None
        successful_results = list(record.successful_groups)
        self._remove_root_node(record)
        for result in successful_results:
            for other in result.group.defer_records:
                other.successful_groups.pop(result, None)
        new_root_nodes = self._promote_non_empty_to_root(record.children)
        self._flush_runs()
        return new_root_nodes, successful_results

    def remove_deferred_fragment(self, record: DeferRecord) -> bool:
        """Drop a deferred fragment (a group failed); True if it was still a root node."""
        if record not in self._root_nodes:
            return False
        self._remove_root_node(record)
        return True

    def remove_stream(self, stream: StreamRec) -> None:
        self._remove_root_node(stream)

    def _remove_root_node(self, node) -> None:
        del self._root_nodes[node]

    def _add_incremental_records(
        self, incremental_records, parents=None, initial_children=None
    ):
        """Wire new groups/streams into the graph (upstream ``_add_incremental_data_records``)."""
        for record in incremental_records:
            if isinstance(record, ExecGroup):
                for defer_record in record.defer_records:
                    self._add_deferred_fragment_node(defer_record, initial_children)
                    defer_record.pending_groups[record] = None
                if self._completes_root_node(record):
                    self._on_execution_group(record)
            elif parents is None:
                initial_children[record] = None
            else:
                for parent in parents:
                    self._add_deferred_fragment_node(parent, initial_children)
                    parent.children[record] = None

    def _promote_non_empty_to_root(self, maybe_empty):
        """Promote non-empty deferred fragments / streams to root nodes (upstream parity)."""
        new_root_nodes: List[Any] = []
        unprocessed = deque(maybe_empty)
        while unprocessed:
            node = unprocessed.popleft()
            if isinstance(node, DeferRecord):
                if node.pending_groups:
                    for group in node.pending_groups:
                        if not self._completes_root_node(group):
                            self._on_execution_group(group)
                    self._root_nodes[node] = None
                    new_root_nodes.append(node)
                    continue
                for child in node.children:
                    if child not in maybe_empty:
                        maybe_empty[child] = None
                        unprocessed.append(child)
            else:
                self._root_nodes[node] = None
                new_root_nodes.append(node)
                self._add_task(self._on_stream_items(node))
        return new_root_nodes

    def _completes_root_node(self, group: ExecGroup) -> bool:
        return any(record in self._root_nodes for record in group.defer_records)

    def _add_deferred_fragment_node(self, record: DeferRecord, initial_children):
        if record in self._root_nodes:
            return
        parent = record.parent
        if parent is None:
            initial_children[record] = None
            return
        parent.children[record] = None
        self._add_deferred_fragment_node(parent, initial_children)

    # -- execution-group + stream pumping -----------------------------------

    def _on_execution_group(self, group: ExecGroup) -> None:
        """Collect a group to run; eagerly-run groups (early execution) enqueue immediately.

        Collected groups are flushed batched (by field_map + parent_type) at the end of the
        promotion wave so a deferred relation over N parents runs its loadMany once.
        """
        if group.result is not None:
            self._enqueue_run(group.result)
            return
        self._pending_runs.append(group)

    def _flush_runs(self) -> None:
        """Run the collected groups, BATCHING same-spec groups into one execute_object_plan.

        Groups sharing a field_map + parent_type (a deferred relation captured once per parent)
        run over all their parents in a single bucket, so the relation's loadMany fires once. Each
        group's per-parent slice becomes its own GroupResult, enqueued like a singleton run.
        """
        if not self._pending_runs:
            return
        runs = self._pending_runs
        self._pending_runs = []
        batches: Dict[Any, List[ExecGroup]] = {}
        order: List[Any] = []
        for group in runs:
            ctx, parent, path, parent_type, field_map, _dus, registry = group.spec
            key = (id(field_map), parent_type.name, id(registry))
            if key not in batches:
                batches[key] = []
                order.append(key)
            batches[key].append(group)
        for key in order:
            members = batches[key]
            if len(members) == 1:
                self._enqueue_run(members[0].runner())
            else:
                produced = run_execution_group_batch(members)
                if self._context.is_awaitable(produced):

                    async def await_and_enqueue_batch(p=produced):
                        for result in await p:
                            self._enqueue(result)

                    self._add_task(await_and_enqueue_batch())
                else:
                    for result in produced:
                        self._enqueue(result)

    def _enqueue_run(self, produced) -> None:
        """Enqueue a single group's result sync, or as a task when async."""
        if isfuture(produced) or self._context.is_awaitable(produced):

            async def await_and_enqueue():
                self._enqueue(await produced)

            self._add_task(await_and_enqueue())
        else:
            self._enqueue(produced)

    async def _on_stream_items(self, stream: StreamRec) -> None:
        """Drain a stream's items, batching exactly like upstream ``_on_stream_items``.

        Synchronously-available item-results accumulate into ONE ``items`` batch; when the next
        item requires an await the accumulated batch is flushed first, then we ``sleep(0)`` and
        await. The Undefined sentinel (iterator exhausted) flushes the tail then emits the
        terminal completed (a :class:`StreamBatch` with ``done=True``). An item that is a stop
        carries its own located errors (a stream-level error → completed-with-errors).
        """
        enqueue = self._enqueue
        items: List[Any] = []
        errors: List[Any] = []
        children: List[Any] = []
        queue = stream.producer.queue
        while True:
            try:
                record = queue.pop(0)
            except IndexError:
                break
            # a queue entry is a ready box or a thunk producing one when popped (the non-early
            # path boxes — ensure_futures — each item only when reached). ``.value`` is a ready
            # StreamItemResult or a scheduled Future (the look-ahead that batches fast items).
            result = record.value if isinstance(record, _Box) else record().value
            if isfuture(result) and not result.done():
                # a genuinely-pending item is the async boundary: flush the accumulated batch
                # (sync-ready items that resolved in the same tick) BEFORE awaiting. A future
                # that is ALREADY done (the look-ahead scheduled it and it resolved while we
                # awaited an earlier item) is consumed without a flush, so same-tick items batch
                # into one payload — matching upstream's observed producer-runs-ahead behaviour.
                if items:
                    enqueue(StreamBatch(stream, list(items), errors or None, children))
                    items = []
                    errors = []
                    children = []
                await sleep(0)
                result = await result
            elif isfuture(result):
                result = result.result()
            if result.item is Undefined:
                if items:
                    enqueue(StreamBatch(stream, list(items), errors or None, children))
                enqueue(StreamBatch(stream, None, result.errors or None, None, done=True))
                return
            items.append(result.item)
            if result.errors:
                errors.extend(result.errors)
            if result.children:
                children.extend(result.children)

    # -- queue draining ------------------------------------------------------

    def current_completed_batch(self):
        queue = self._completed_queue
        while queue:
            yield queue.pop(0)
        if not self._root_nodes:
            self.abort()

    def next_completed_batch(self) -> Future:
        loop = get_running_loop()
        future: Future = loop.create_future()
        self._next_queue.append(future)
        return future

    def abort(self) -> None:
        for resolve in self._next_queue:
            if not resolve.done():
                resolve.set_result(None)

    def has_next(self) -> bool:
        return bool(self._root_nodes)

    def stop_incremental_data(self) -> None:
        for future in self._next_queue:
            future.cancel()

    def _yield_current(self, first):
        yield first
        yield from self.current_completed_batch()

    def _enqueue(self, completed) -> None:
        # append first, then resolve any waiting consumer future with a drain-generator. Appending
        # before resolving lets the producer's synchronous burst (multiple flushes between awaits)
        # accumulate in the queue; the consumer, when it wakes and iterates the generator, drains
        # the whole burst into ONE payload — matching upstream's producer-runs-ahead batching.
        self._completed_queue.append(completed)
        if self._next_queue:
            future = self._next_queue.pop(0)
            if not future.done():
                future.set_result(self._drain())

    def _drain(self):
        yield from self.current_completed_batch()

    def _add_task(self, awaitable) -> None:
        task = ensure_future(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class StreamBatch:
    """One drained stream batch: an items payload, or the terminal completed (``done``)."""

    __slots__ = ("stream", "items", "errors", "children", "done")

    def __init__(self, stream, items, errors, children, done=False):
        self.stream = stream
        self.items = items
        self.errors = errors
        self.children = children
        self.done = done


# ---------------------------------------------------------------------------
# the publisher (a transcription of upstream IncrementalPublisher)
# ---------------------------------------------------------------------------


class Publisher:
    """Assemble the initial payload + drive subsequent payloads (upstream parity).

    Owns monotonic id allocation (``_next_id``) and the ``_subscribe`` pump that drains the
    graph's completed batches into ``SubsequentIncrementalExecutionResult`` payloads, with
    ``hasNext`` recomputed after each batch and best-id / subPath emission for execution groups.
    """

    def __init__(self, context, graph: IncGraph):
        self._context = context
        self._graph = graph
        self._next_id = 0

    def next_id(self) -> str:
        id_ = self._next_id
        self._next_id += 1
        return str(id_)

    def to_pending(self, new_root_nodes) -> list:
        from graphql.execution import PendingResult

        pending = []
        for node in new_root_nodes:
            node.id = self.next_id()
            path = node.path.as_list() if node.path else []
            pending.append(PendingResult(node.id, path, node.label))
        return pending

    def build_response(self, data, errors, incremental_records):
        from graphql.execution import (
            ExperimentalIncrementalExecutionResults,
            InitialIncrementalExecutionResult,
        )

        new_root_nodes = self._graph.get_new_root_nodes(incremental_records)
        pending = self.to_pending(new_root_nodes)
        initial = InitialIncrementalExecutionResult(
            data, errors or None, pending=pending, has_next=True
        )
        log.debug("incremental initial payload", pending=len(pending))
        return ExperimentalIncrementalExecutionResults(initial, self._subscribe())

    async def _subscribe(self):
        from graphql.execution import SubsequentIncrementalExecutionResult

        graph = self._graph
        try:
            while True:
                batch = graph.current_completed_batch()
                while batch is not None:
                    pending: list = []
                    incremental: list = []
                    completed: list = []
                    for completed_record in batch:
                        await self._handle_completed(
                            completed_record, pending, incremental, completed
                        )
                    if incremental or completed:
                        has_next = graph.has_next()
                        yield SubsequentIncrementalExecutionResult(
                            has_next=has_next,
                            pending=pending or None,
                            incremental=incremental or None,
                            completed=completed or None,
                        )
                        if not has_next:
                            return
                    batch = await graph.next_completed_batch()
        finally:
            await self._stop_async_iterators()

    async def _stop_async_iterators(self):
        self._graph.stop_incremental_data()

    async def _handle_completed(self, completed_record, pending, incremental, completed):
        if isinstance(completed_record, StreamBatch):
            await self._handle_stream_batch(
                completed_record, pending, incremental, completed
            )
        else:
            self._handle_execution_group(
                completed_record, pending, incremental, completed
            )

    def _handle_execution_group(self, result: GroupResult, pending, incremental, completed):
        from graphql.execution import CompletedResult, IncrementalDeferResult

        group = result.group
        graph = self._graph
        if result.failed:
            for record in group.defer_records:
                if not graph.remove_deferred_fragment(record):
                    continue
                completed.append(CompletedResult(record.id, result.errors))
            return

        # filter the group's revealed children (nested defers / streams) whose enclosing object
        # was nulled in the group's data — a stream/defer under a bubbled object in a DEFERRED
        # payload is dropped (its iterator closed), mirroring the initial-result filter.
        if result.children:
            result.children = filter_group_children(
                self._context, result.children, result
            )
        graph.add_completed_group(result)
        for record in group.defer_records:
            completion = graph.complete_deferred_fragment(record)
            if completion is None:
                continue
            new_root_nodes, successful_results = completion
            pending.extend(self.to_pending(new_root_nodes))
            for gres in successful_results:
                best_id, sub_path = self._best_id_and_sub_path(record.id, record, gres)
                incremental.append(
                    IncrementalDeferResult(
                        data=gres.data,
                        id=best_id,
                        sub_path=sub_path,
                        errors=gres.errors or None,
                    )
                )
            completed.append(CompletedResult(record.id))

    def _best_id_and_sub_path(self, initial_id, initial_record, gres: GroupResult):
        """Pick the deepest-path fragment's id and compute the subPath (upstream parity)."""
        path = initial_record.path
        max_length = len(path.as_list()) if path else 0
        best_id = initial_id
        for record in gres.group.defer_records:
            if record is initial_record:
                continue
            if record.id is None:
                continue
            fragment_path = record.path
            length = len(fragment_path.as_list()) if fragment_path else 0
            if length > max_length:
                max_length = length
                best_id = record.id
        sub_path = gres.path[max_length:] or None
        return best_id, sub_path

    async def _handle_stream_batch(self, batch: StreamBatch, pending, incremental, completed):
        """Handle a drained stream batch (upstream ``_handle_completed_stream_items``).

        A terminal batch (``done``) with errors → completed-with-errors; a terminal with no
        errors → completed; a normal items batch → an IncrementalStreamResult (errors riding it)
        plus any revealed child records' pending, then an ``await sleep(0)`` (upstream's
        unconditional yield after a non-terminal items result) — the yield that lets the producer
        flush its full burst before the consumer re-waits, so same-tick items batch into one
        payload.
        """
        from graphql.execution import CompletedResult, IncrementalStreamResult

        stream = batch.stream
        graph = self._graph
        if batch.done:
            completed.append(CompletedResult(stream.id, batch.errors))
            graph.remove_stream(stream)
            # a stream that completed-with-errors (a non-null item bubbled / the iterator raised)
            # closes its underlying async iterator (upstream's early_return on the cancellable
            # stream record). A normal exhaustion already drained the iterator.
            if batch.errors is not None and stream.early_return is not None:
                from contextlib import suppress

                with suppress(Exception):
                    coro = stream.early_return()
                    if self._context.is_awaitable(coro):
                        await coro
            return
        incremental.append(
            IncrementalStreamResult(
                items=batch.items, id=stream.id, errors=batch.errors or None
            )
        )
        if batch.children:
            new_root_nodes = graph.get_new_root_nodes(batch.children)
            pending.extend(self.to_pending(new_root_nodes))
        await sleep(0)


# ---------------------------------------------------------------------------
# stream item production
# ---------------------------------------------------------------------------


class StreamItemResult:
    """One produced stream item: a completed value, its errors, and revealed children.

    ``item`` is ``Undefined`` for the iterator-exhausted sentinel (with ``errors`` carrying a
    stream-level error when the iterator raised). ``completed`` is the inline-head completion of
    a single item (used by the head-pull path); for the streamed-tail path ``item`` IS the
    completed value.
    """

    __slots__ = ("item", "errors", "children", "completed")

    def __init__(self, item=Undefined, errors=None, children=None, completed=None):
        self.item = item
        self.errors = errors
        self.children = children
        self.completed = completed


def build_stream_record(path, label, producer, early_return):
    """Build a :class:`StreamRec` for a @stream'd list field (called by the executor seam)."""
    return StreamRec(path, label, producer, early_return)


def complete_stream_item(context, field_plan, item_completer, item, item_path, info, item_bridge=None):
    """Complete ONE streamed item through the item completer → a :class:`StreamItemResult`.

    Replicates upstream ``complete_stream_item`` + ``build_stream_item_result``: a nullable-item
    bubble becomes ``item=None`` + the located error on the batch; a NON-null item bubble nulls
    the whole stream (a terminal sentinel carrying the error). Deeper @defer groups revealed
    inside the item are captured as the item's ``children`` (per-item defer records). SYNC →
    a StreamItemResult; async → a coroutine resolving to one.

    `item_bridge` is the per-parent P4 hoist seed (a 1-element ``value_owner``): completing this
    single item descends into the child layer owned by that parent bucket row, so it seeds the
    columns hoisted OUT of that layer. None (the default / hoist-off) leaves completion byte-identical.
    """
    from .completion import NonNullCompleter, complete_values

    item_is_non_null = isinstance(item_completer, NonNullCompleter)

    # a fresh per-item context: own errors + a defer/stream sink so the item's own deeper
    # @defer / @stream are captured as this item's children, not the request's initial data.
    item_context = _StreamItemContext(context)
    children: List[Any] = []
    item_context._grafast_defer_sink = make_defer_sink(item_context, children)
    item_context._grafast_stream_sink = lambda rec: children.append(rec)

    completed = complete_values(
        item_context,
        item_completer,
        [item],
        [item_path],
        [info],
        field_plan.field_nodes,
        field_plan.field_label,
        item_bridge,
    )

    def build(values):
        result = values[0]
        if isinstance(result, Bubble):
            if item_is_non_null:
                return StreamItemResult(errors=[result.error])
            return StreamItemResult(
                item=None, errors=[result.error], children=children or None
            )
        return StreamItemResult(
            item=result, errors=item_context.errors or None, children=children or None
        )

    if context.is_awaitable(completed):

        async def finish():
            return build(await completed)

        return finish()
    return build(completed)


class _StreamItemContext:
    """A per-streamed-item context wrapper: fresh errors + own defer/stream sinks.

    Like :class:`JobContext`, mirrors ``build_per_event_execution_context`` — shares schema /
    fragments / variables / resolvers / config but owns a FRESH ``errors`` list so the item's
    errors ride its own stream batch, and its own ``_grafast_*`` scratch so nested defers
    captured during the item complete attach to the item (not the request initial data).
    """

    def __init__(self, base):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "errors", [])
        object.__setattr__(self, "_grafast_on_step_batch", getattr(base, "_grafast_on_step_batch", None))
        object.__setattr__(self, "_grafast_concurrency", getattr(base, "_grafast_concurrency", None))

    def __getattr__(self, name):
        return getattr(self._base, name)

    def build_resolve_info(self, field_def, field_nodes, parent_type, path):
        return self._base.build_resolve_info(field_def, field_nodes, parent_type, path)

    def collect_subfields(self, object_type, field_nodes):
        return _compat.collect_subfields(self, object_type, field_nodes)

    def ensure_valid_runtime_type(self, runtime_type_name, return_type, field_nodes, info, result):
        return _compat.ensure_valid_runtime_type(
            self, runtime_type_name, return_type, field_nodes, info, result
        )


class _Box:
    """A boxed value/future that self-updates when its scheduled awaitable resolves.

    A transcription of upstream ``BoxedAwaitableOrValue``: an awaitable is ``ensure_future``d
    EAGERLY (so a fast item starts computing before the driver pulls it — the look-ahead that
    lets fast items batch into one payload), and ``.value`` returns the ready result once done.
    """

    __slots__ = ("_value",)

    def __init__(self, value):
        if _is_coro_or_future(value):
            future = ensure_future(value) if not isfuture(value) else value
            future.add_done_callback(self._update)
            self._value = future
        else:
            self._value = value

    @property
    def value(self):
        value = self._value
        if isfuture(value) and value.done():
            self._value = value = value.result()
        return value

    def _update(self, future):
        from asyncio import CancelledError
        from contextlib import suppress

        with suppress(CancelledError):
            self._value = future.result()


def _is_coro_or_future(value):
    from asyncio import iscoroutine

    return isfuture(value) or iscoroutine(value)


class SyncStreamProducer:
    """Streams a fully-materialised tail (sync list or list of awaitables), item-by-item.

    Transcribes upstream ``build_sync_stream_item_queue``: the queue starts with ONE entry — the
    first-executor (a ready box when early, else a thunk boxing on pop). The first executor boxes
    item 0, then LOOPS the remaining tail appending one entry per item (a thunk when not early,
    so each item is ensure_future'd only when the driver reaches it), then a terminal Undefined.
    """

    def __init__(self, context, field_plan, item_completer, path, info, tail, initial_count, item_bridge=None):
        self._context = context
        self._field_plan = field_plan
        self._item_completer = item_completer
        self._path = path
        self._info = info
        self._tail = tail
        self._initial_index = initial_count
        self._item_bridge = item_bridge
        early = getattr(context, "_grafast_enable_early_execution", False)
        self.queue: List[Any] = []
        if early:

            async def await_first():
                return self._first_executor()

            self.queue.append(_Box(await_first()))
        else:
            self.queue.append(lambda: _Box(self._first_executor()))

    def _item_executor(self, item, index):
        item_path = self._path.add_key(index, None)
        return complete_stream_item(
            self._context, self._field_plan, self._item_completer, item, item_path, self._info,
            self._item_bridge,
        )

    def _first_executor(self):
        early = getattr(self._context, "_grafast_enable_early_execution", False)
        append = self.queue.append
        if not self._tail:
            append(_Box(StreamItemResult()))
            return StreamItemResult()
        first_box = _Box(self._item_executor(self._tail[0], self._initial_index))
        index = self._initial_index + 1
        for item in self._tail[1:]:
            captured = item
            captured_index = index

            def executor(it=captured, ix=captured_index):
                return self._item_executor(it, ix)

            append(_Box(executor()) if early else (lambda ex=executor: _Box(ex())))
            index += 1
        append(_Box(StreamItemResult()))
        return first_box.value


class AsyncStreamProducer:
    """Streams a live async iterator item-by-item (each pull awaits ``anext``).

    Transcribes upstream ``build_async_stream_item_queue`` / ``get_next_async_stream_item_result``:
    head items (``items[:initialCount]``) are pulled inline by the executor BEFORE the producer is
    handed to the driver (:meth:`next_async_item`); the queue then holds ONE entry — the next
    pull — and each pull appends the following pull's entry (the look-ahead). A thunk when not
    early so the next ``anext`` only fires when the driver reaches it.
    """

    def __init__(self, context, field_plan, item_completer, path, info, iterator, initial_count, item_bridge=None):
        self._context = context
        self._field_plan = field_plan
        self._item_completer = item_completer
        self._path = path
        self._info = info
        self._iterator = iterator
        self.initial_count = initial_count
        self._item_bridge = item_bridge
        self._index = 0
        early = getattr(context, "_grafast_enable_early_execution", False)
        self._early = early
        self.queue: List[Any] = []

    async def next_async_item(self):
        """Pull + complete one HEAD item inline (returns a StreamItemResult, advancing _index)."""
        from graphql.error import located_error

        index = self._index
        self._index += 1
        try:
            item = await self._iterator.__anext__()
        except StopAsyncIteration:
            return StreamItemResult()
        except (GraphQLError, TypeError, ValueError, RuntimeError) as exc:
            return StreamItemResult(
                errors=[located_error(exc, self._field_plan.field_nodes, self._path.as_list())]
            )
        item_path = self._path.add_key(index, None)
        completed = complete_stream_item(
            self._context, self._field_plan, self._item_completer, item, item_path, self._info,
            self._item_bridge,
        )
        result = await completed if self._context.is_awaitable(completed) else completed
        return StreamItemResult(item=result.item, errors=result.errors, children=result.children)

    def start(self):
        """Seed the queue's first pull entry (called once after the head pull, before draining)."""
        self._append_next()

    def _append_next(self):
        executor = self._next_executor
        self.queue.append(
            _Box(executor()) if self._early else (lambda: _Box(executor()))
        )

    async def _next_executor(self):
        from graphql.error import located_error

        index = self._index
        self._index += 1
        try:
            item = await self._iterator.__anext__()
        except StopAsyncIteration:
            return StreamItemResult()
        except (GraphQLError, TypeError, ValueError, RuntimeError) as exc:
            return StreamItemResult(
                errors=[located_error(exc, self._field_plan.field_nodes, self._path.as_list())]
            )
        item_path = self._path.add_key(index, None)
        completed = complete_stream_item(
            self._context, self._field_plan, self._item_completer, item, item_path, self._info,
            self._item_bridge,
        )
        # look-ahead: schedule the NEXT pull's entry before completing this one.
        self._append_next()
        result = await completed if self._context.is_awaitable(completed) else completed
        return StreamItemResult(item=result.item, errors=result.errors, children=result.children)


def make_defer_sink(context, sink_list):
    """Build a defer sink that records execution groups for captured @defer levels.

    The sink receives ``(defer_plan, parent, path, parent_type)`` (one call per live parent at a
    @defer'd object level) and appends the level's execution-group records to ``sink_list``,
    minting the deferred fragment records via the context's defer_map.
    """

    def sink(defer_plan, parent, path, parent_type):
        records = capture_defer_groups(context, defer_plan, parent, path, parent_type)
        sink_list.extend(records)

    return sink


# ---------------------------------------------------------------------------
# @defer execution groups (the multi-fragment subPath / dedup core)
# ---------------------------------------------------------------------------


def capture_defer_groups(context, defer_plan, parent, path, parent_type):
    """Mint this object level's deferred fragment records + execution groups (P7).

    Transcribes upstream ``add_new_deferred_fragments`` + ``collect_execution_groups``: mints a
    deferred fragment record per new defer usage at ``path`` (parent records resolved from the
    ENCLOSING defer_map on the context, so a nested @defer's record points at the fragment whose
    completion releases it), then builds an :class:`ExecGroup` per new grouped-field-set keyed by
    its defer-usage-set. Returns the new execution-group records (the initial-result children /
    the running group's revealed children).
    """
    # the defer records are PATH-keyed (a @defer used on a list produces one record per item
    # path); the records were already minted (records for this level's new defer usages at this
    # path) by the executor's pre-mint pass BEFORE the field walk, so an INITIAL object field's
    # deeper groups (captured mid-walk) resolve a parent record minted at this level. Look each
    # group's defer record up by (usage, path) from the path-keyed registry on the context.
    registry = get_defer_registry(context)
    path_key = _path_key(path)
    early = getattr(context, "_grafast_enable_early_execution", False)
    records = []
    for defer_usage_set, field_map in defer_plan.new_groups:
        # a group's defer usage was minted where the @defer was DECLARED (a shallower-or-equal
        # path), not necessarily this group's path; resolve each to the record at the longest
        # ancestor path prefix (e.g. a @defer at ["hero"] whose only NEW group is the deeper
        # ["hero","nestedObject","deeperObject"] bar field — the parent-payload dedup).
        defer_records = [_lookup_record(registry, du, path_key) for du in defer_usage_set]
        group = ExecGroup(defer_records, None)
        group.spec = (context, parent, path, parent_type, field_map, defer_usage_set, registry)
        group.runner = make_group_runner(
            context, group, parent, path, parent_type, field_map, defer_usage_set, registry
        )
        if early:
            # enable_early_execution: run the group's fields NOW (during the walk) so a fast
            # deferred resolver fires before a slow initial one (upstream's collect_execution_
            # groups boxes the result eagerly). The boxed result is consumed at promotion.
            group.result = _run_early(context, group)
        records.append(group)
    return records


def _run_early(context, group):
    """Run an execution group eagerly (early execution), boxing the result for promotion."""
    produced = group.runner()
    if context.is_awaitable(produced):
        return ensure_future(produced)
    return produced


def get_defer_registry(context):
    """The path-keyed deferred-fragment-record registry for the current walk (lazy)."""
    registry = getattr(context, "_grafast_defer_map", None)
    if registry is None:
        registry = {}
        object.__setattr__(context, "_grafast_defer_map", registry)
    return registry


def _path_key(path):
    return tuple(path.as_list()) if path else ()


def add_new_deferred_fragments(registry, new_defer_usages, path):
    """Mint a DeferRecord per new DeferUsage at ``path`` into the PATH-keyed registry.

    A child defer usage's parent record is the record for ``usage.parent_defer_usage`` at the
    nearest ancestor path already in the registry (a @defer declared at a shallower level). A
    root-level usage has parent None. Keys by ``(usage id, path tuple)`` — a list @defer mints a
    distinct record per item path. Mirrors upstream ``add_new_deferred_fragments``.
    """
    path_key = _path_key(path)
    for usage in new_defer_usages:
        key = (id(usage), path_key)
        if key in registry:
            continue
        parent_usage = usage.parent_defer_usage
        parent = (
            None if parent_usage is None
            else _find_parent_record(registry, parent_usage, path_key)
        )
        registry[key] = DeferRecord(path, usage.label, parent)
    return registry


def _find_parent_record(registry, parent_usage, path_key):
    """Find the parent defer record: ``parent_usage`` at the longest ancestor path prefix."""
    return _lookup_record(registry, parent_usage, path_key)


def _lookup_record(registry, usage, path_key):
    """The DeferRecord for ``usage`` at the longest registered path prefix of ``path_key``.

    A @defer's record is minted at its DECLARATION path; a group it owns may sit at a deeper
    path. Resolving to the longest ancestor-prefix path picks the right record (and, for a list
    @defer, the record at the matching item-path prefix)."""
    best = None
    best_len = -1
    uid = id(usage)
    for (rid, pkey), record in registry.items():
        if rid != uid:
            continue
        if pkey == path_key[: len(pkey)] and len(pkey) > best_len:
            best = record
            best_len = len(pkey)
    return best


def make_group_runner(
    context, group, parent, path, parent_type, field_map, defer_usage_set, registry
):
    """Build the 0-arg runner that executes one grouped-field-set → a :class:`GroupResult`.

    Runs ``field_map`` against ``parent`` at ``path`` through the engine's own
    ``execute_object_plan`` (P2.5 detachment, so a loadMany over N parents fires once), with a
    FRESH error collector + a defer/stream sink whose minted records carry THIS group's
    registry as their enclosing scope (so a nested @defer's parent is one of this group's
    fragments). A null-bubble at the group's top boundary marks the result ``failed``.
    """

    def runner():
        return run_execution_group(
            context, group, parent, path, parent_type, field_map, defer_usage_set, registry
        )

    return runner


def run_execution_group(
    context, group, parent, path, parent_type, field_map, defer_usage_set, registry
):
    """Execute one grouped-field-set at a path; return a GroupResult (sync or coroutine)."""
    from .execute import execute_object_plan

    object_plan = build_group_plan(context, parent_type, field_map)

    job_context = JobContext(context)
    # the group run inherits the capturing scope's path-keyed records (so a deeper defer's parent
    # resolves) but mints its OWN new records into a private copy of the registry, so concurrent
    # group runs don't collide (records at deeper paths are minted by the run's pre-mint pass).
    object.__setattr__(job_context, "_grafast_defer_map", dict(registry))
    children: List[Any] = []
    job_context._grafast_defer_sink = make_defer_sink(job_context, children)
    job_context._grafast_stream_sink = lambda rec: children.append(rec)

    abs_path = path.as_list() if path else []
    result = execute_object_plan(job_context, object_plan, [parent], [path])

    def build(out):
        data = out[0]
        if isinstance(data, Bubble):
            return GroupResult(group, abs_path, None, [data.error], True, None)
        return GroupResult(
            group, abs_path, data, job_context.errors or None, False, children or None
        )

    if job_context.is_awaitable(result):

        async def finish():
            return build(await result)

        return finish()
    return build(result)


def run_execution_group_batch(groups):
    """Run same-spec execution groups over ALL their parents in ONE bucket; per-group results.

    The grafast batching guarantee for @defer: a deferred relation captured once per parent (same
    field_map + parent_type) runs its loadMany ONCE over all parents. One ``execute_object_plan``
    over ``[parent for each group]`` produces a per-parent output; each group's slice + its
    path-scoped errors + its parent-routed children become its own :class:`GroupResult`. Returns a
    list of GroupResults (parallel to ``groups``) or a coroutine resolving to it.
    """
    from .execute import execute_object_plan

    context, _p0, _path0, parent_type, field_map, _dus, registry = groups[0].spec
    object_plan = build_group_plan(context, parent_type, field_map)

    job_context = JobContext(context)
    object.__setattr__(job_context, "_grafast_defer_map", dict(registry))
    # per-parent captured children: route each child record to the owning group by parent identity.
    captured: List[Any] = []
    job_context._grafast_defer_sink = lambda dp, parent, path, pt: captured.append(
        ("defer", parent, dp, path, pt)
    )
    job_context._grafast_stream_sink = lambda rec: captured.append(("stream", rec))

    parents = [g.spec[1] for g in groups]
    paths = [g.spec[2] for g in groups]
    result = execute_object_plan(job_context, object_plan, parents, paths)

    def build(outs):
        # children captured by the shared sink: defer children are re-materialised per owning
        # parent (so the records carry the right paths); stream children route by path prefix.
        defer_children_by_parent: Dict[int, list] = {}
        for entry in captured:
            if entry[0] == "defer":
                _kind, parent, dp, path, pt = entry
                recs = capture_defer_groups(job_context, dp, parent, path, pt)
                defer_children_by_parent.setdefault(id(parent), []).extend(recs)
        results = []
        for group, out, path in zip(groups, outs, paths):
            abs_path = path.as_list() if path else []
            if isinstance(out, Bubble):
                results.append(GroupResult(group, abs_path, None, [out.error], True, None))
                continue
            my_errors = [e for e in job_context.errors if _error_under(e, abs_path)]
            children = list(defer_children_by_parent.get(id(group.spec[1]), []))
            for entry in captured:
                if entry[0] == "stream" and _stream_under(entry[1], abs_path):
                    children.append(entry[1])
            results.append(
                GroupResult(group, abs_path, out, my_errors or None, False, children or None)
            )
        return results

    if job_context.is_awaitable(result):

        async def finish():
            return build(await result)

        return finish()
    return build(result)


def _error_under(error, path_prefix):
    epath = getattr(error, "path", None) or []
    return list(epath[: len(path_prefix)]) == list(path_prefix)


def _stream_under(stream, path_prefix):
    spath = stream.path.as_list() if stream.path else []
    return spath[: len(path_prefix)] == path_prefix


def build_group_plan(context, parent_type, field_map):
    """Plan a grouped-field-set as a self-contained ObjectPlan (own RootStep), upstream parity.

    Mirrors the abstract concrete-type subtree path: a fresh ``RootStep`` seeds the parent,
    the group's fields (re-collected WITH their defer_usage so deeper defers partition right)
    hang off it, and the DAG is finalized. Replanned per run — cheap, and required so a nested
    defer sees the right ``parent_defer_usages``.
    """
    from .core_steps import RootStep
    from .dag import Plan
    from .plan import LayerReason, finalize_plan, plan_object

    plan = Plan()
    config = context.grafast_config
    plan.inline_relations = config.inline_relations
    plan.placeholders = config.placeholders
    plan.cache_plans = config.cache_plans
    plan.hoist = config.hoist
    plan.incremental = True
    root_step = RootStep()
    plan.add_step(root_step)
    initial_nodes = {
        response_name: [detail.node for detail in details]
        for response_name, details in field_map.items()
    }
    object_plan = plan_object(
        context,
        parent_type,
        initial_nodes,
        parent_step=root_step,
        plan=plan,
        reason=LayerReason.ROOT,
        details_map=field_map,
    )
    return finalize_plan(plan, object_plan)


class JobContext:
    """A shallow per-execution-group wrapper over the request context.

    Mirrors upstream's ``build_per_event_execution_context``: shares the request's schema /
    fragments / variables / resolvers / config, but owns a FRESH ``errors`` list so a group's
    errors scope to its own incremental entry and never corrupt the already-emitted initial
    data. Attribute reads fall through to the wrapped context; the pipeline's ``_grafast_*``
    scratch attributes are set on this wrapper (so the parent's subfield cache etc. are not
    shared mutably across groups).
    """

    def __init__(self, base):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "errors", [])
        object.__setattr__(self, "_grafast_on_step_batch", getattr(base, "_grafast_on_step_batch", None))
        object.__setattr__(self, "_grafast_concurrency", getattr(base, "_grafast_concurrency", None))

    def __getattr__(self, name):
        return getattr(self._base, name)

    def build_resolve_info(self, field_def, field_nodes, parent_type, path):
        return self._base.build_resolve_info(field_def, field_nodes, parent_type, path)

    def collect_subfields(self, object_type, field_nodes):
        return _compat.collect_subfields(self, object_type, field_nodes)

    def ensure_valid_runtime_type(self, runtime_type_name, return_type, field_nodes, info, result):
        return _compat.ensure_valid_runtime_type(
            self, runtime_type_name, return_type, field_nodes, info, result
        )


# ---------------------------------------------------------------------------
# entry: assemble the initial payload + run the graph
# ---------------------------------------------------------------------------


def run_incremental(context, run_initial, incremental_records):
    """Build an incremental response from the initial data + captured top records.

    ``incremental_records`` is the flat list of root-level execution-group records and stream
    records the initial walk captured (via the defer/stream sinks). Records whose path was nulled
    by an initial error (a strict-prefix error path) are FILTERED — a stream/defer hanging off a
    bubbled parent is dropped (its iterator closed). If filtering leaves no records, the result
    is a plain ExecutionResult (no incremental work survived). The publisher then promotes the
    surviving records to root nodes (minting ids), emits the initial pending, and drives the
    subsequent payloads.
    """
    surviving = filter_nulled_records(context, incremental_records, run_initial.data)
    if not surviving:
        return _compat.make_result(run_initial.data, list(run_initial.errors or []))
    graph = IncGraph(context)
    publisher = Publisher(context, graph)
    return publisher.build_response(
        run_initial.data, run_initial.errors or None, surviving
    )


def filter_nulled_records(context, records, data):
    """Drop records whose enclosing object was nulled in ``data`` (upstream record-graph filter).

    A stream / execution-group hangs off a path; if any ANCESTOR of that path resolved to null in
    the delivered data (a non-null violation bubbled the enclosing object to None), the record's
    data would write into a nulled parent, so it is filtered out — its iterator closed. Returns
    the surviving records. (Upstream filters via the record graph; we check the data tree, which
    is observably equivalent for the initial result.)
    """
    surviving = []
    for record in records:
        if _path_nulled(data, _record_path(record)):
            _close_filtered(context, record)
            continue
        surviving.append(record)
    return surviving


def filter_group_children(context, children, result: GroupResult):
    """Drop a completed group's revealed children whose enclosing object was nulled in its data.

    The group's ``data`` is rooted at ``result.path``; a child record's absolute path, projected
    relative to that, must index a non-null object — else (an inner non-null bubble) the child is
    filtered (its iterator closed). Mirrors the initial-result :func:`filter_nulled_records`.
    """
    base_len = len(result.path)
    surviving = []
    for child in children:
        rel = _record_path(child)[base_len:]
        if _path_nulled(result.data, rel):
            _close_filtered(context, child)
            continue
        surviving.append(child)
    return surviving


def _record_path(record):
    """A record's path (as a list): a stream's field path, or an exec group's deepest fragment."""
    if isinstance(record, StreamRec):
        return record.path.as_list() if record.path else []
    deepest = []
    for defer_record in record.defer_records:
        p = defer_record.path.as_list() if defer_record.path else []
        if len(p) > len(deepest):
            deepest = p
    return deepest


def _path_nulled(data, path):
    """Whether any STRICT ancestor of ``path`` is None in ``data`` (the enclosing object bubbled)."""
    node = data
    for key in path[:-1]:
        if node is None:
            return True
        try:
            node = node[key]
        except (KeyError, IndexError, TypeError):
            return False
    return node is None and bool(path)


def _close_filtered(context, record):
    """Close a filtered stream's iterator (schedule its early-return), ignoring errors."""
    if isinstance(record, StreamRec) and record.early_return is not None:
        from contextlib import suppress

        with suppress(Exception):
            coro = record.early_return()
            if context.is_awaitable(coro):
                ensure_future(coro)
