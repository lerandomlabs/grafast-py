"""Metamorphic + type-identity tests for @defer / @stream incremental delivery.

These run ONLY on graphql-core 3.3 (the @defer/@stream directives do not exist on 3.2);
the whole module is skipped on the 3.2 line via ``_compat.supports_incremental()``, so the
3.2 baseline is unaffected.

Metamorphic relations (independent of byte-level wire shape, so they hold even where the
exact payload grouping is only partially implemented):

* defer == inline-merge: a query with ``@defer`` yields the SAME merged data as the query
  with the ``@defer`` stripped.
* stream == inline-list: a ``@stream(initialCount:k)`` list yields ``initial[:k]`` plus all
  streamed items == the full list resolved inline.
* defer-batching: a ``@defer``'d loader relation over N parents fires the SAME number of
  load batches as the non-deferred query (the host-parent_step-reuse guarantee).

Type-identity: ``@defer(if:false)`` / no-directive returns a plain ``ExecutionResult``;
``@defer`` returns an ``ExperimentalIncrementalExecutionResults``.
"""

import asyncio

import pytest
from graphql.language import parse
from graphql.type import (
    GraphQLField,
    GraphQLID,
    GraphQLList,
    GraphQLObjectType,
    GraphQLSchema,
    GraphQLString,
)

from grafast_py import _compat
from grafast_py.config import GrafastConfig
from grafast_py.entry import experimental_execute_incrementally

pytestmark = pytest.mark.skipif(
    not _compat.supports_incremental(),
    reason="@defer/@stream incremental delivery requires graphql-core 3.3",
)


friend_type = GraphQLObjectType(
    "Friend",
    {"id": GraphQLField(GraphQLID), "name": GraphQLField(GraphQLString)},
)
hero_type = GraphQLObjectType(
    "Hero",
    {
        "id": GraphQLField(GraphQLID),
        "name": GraphQLField(GraphQLString),
        "friends": GraphQLField(GraphQLList(friend_type)),
    },
)
schema = GraphQLSchema(GraphQLObjectType("Query", {"hero": GraphQLField(hero_type)}))

HERO = {
    "id": 1,
    "name": "Luke",
    "friends": [
        {"id": 2, "name": "Han"},
        {"id": 3, "name": "Leia"},
        {"id": 4, "name": "C-3PO"},
    ],
}


async def run_collect(document, root=None):
    """Execute and return the result: a single formatted dict, or a list of formatted payloads."""
    from graphql.execution import (
        ExecutionResult,
        ExperimentalIncrementalExecutionResults,
    )

    result = experimental_execute_incrementally(schema, document, root or {"hero": HERO})
    if asyncio.iscoroutine(result):
        result = await result
    if isinstance(result, ExperimentalIncrementalExecutionResults):
        payloads = [result.initial_result.formatted]
        async for patch in result.subsequent_results:
            payloads.append(patch.formatted)
        return payloads
    assert isinstance(result, ExecutionResult)
    return result.formatted


def merge_incremental(payloads):
    """Merge an incremental payload list into one ``data`` dict (defer == inline-merge).

    Folds each ``incremental[].data`` into the initial data at ``path + subPath`` (for stream,
    extends the list at ``path``). Path-walking handles the byte-shape-independent merge so the
    metamorphic equality holds regardless of which payload a given fragment landed in.
    """
    # pre-scan EVERY payload's pending so an id's declared path is known regardless of which
    # payload its data arrives in (pending for an id may precede its data by several payloads).
    pending_paths: dict = {}
    for payload in payloads:
        for pending in payload.get("pending", []) or []:
            pending_paths[pending["id"]] = pending["path"]

    data = payloads[0].get("data")
    for payload in payloads:
        for entry in payload.get("incremental", []) or []:
            path = pending_paths.get(entry["id"], []) + (entry.get("subPath") or [])
            target = _resolve_path(data, path)
            if "items" in entry:
                target.extend(entry["items"])
            else:
                _deep_update(target, entry["data"])
    return data


def _resolve_path(data, path):
    node = data
    for key in path:
        node = node[key]
    return node


def _deep_update(target, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


# ----------------------------------------------------------- defer == inline-merge
def test_defer_equals_inline_merge():
    deferred = parse(
        "{ hero { id ...N @defer } } fragment N on Hero { name }"
    )
    inline = parse("{ hero { id name } }")

    deferred_payloads = asyncio.run(run_collect(deferred))
    inline_result = asyncio.run(run_collect(inline))

    assert isinstance(deferred_payloads, list)
    assert merge_incremental(deferred_payloads) == inline_result["data"]


def test_defer_on_list_items_equals_inline_merge():
    deferred = parse(
        "{ hero { friends { id ...F @defer } } } fragment F on Friend { name }"
    )
    inline = parse("{ hero { friends { id name } } }")

    deferred_payloads = asyncio.run(run_collect(deferred))
    inline_result = asyncio.run(run_collect(inline))

    assert merge_incremental(deferred_payloads) == inline_result["data"]


def test_multi_fragment_defer_dedup_equals_inline_merge():
    """Two overlapping root-level @defers merge to the inline data AND never re-deliver a leaf.

    The multi-fragment subPath / dedup invariant: a field selected by two overlapping deferred
    fragments (and by the initial selection) is delivered ONCE. Asserts (a) the merged incremental
    data equals the non-deferred query's data, and (b) no incremental entry re-delivers a leaf path
    already present in the initial payload (the parent-payload dedup).
    """
    deferred = parse(
        """
        {
          hero {
            id
            ... @defer { id name }
            ... @defer { name }
          }
        }
        """
    )
    inline = parse("{ hero { id name } }")

    deferred_payloads = asyncio.run(run_collect(deferred))
    inline_result = asyncio.run(run_collect(inline))

    assert isinstance(deferred_payloads, list)

    # snapshot the INITIAL leaves + pending paths BEFORE merge_incremental mutates the data dict.
    import copy

    initial_data = copy.deepcopy(deferred_payloads[0].get("data") or {})
    pending_paths = {}
    for payload in deferred_payloads:
        for pending in payload.get("pending", []) or []:
            pending_paths[pending["id"]] = pending["path"]

    # dedup invariant: no incremental entry re-delivers a leaf already in the initial payload.
    for payload in deferred_payloads:
        for entry in payload.get("incremental", []) or []:
            if "items" in entry:
                continue
            base = pending_paths.get(entry["id"], []) + (entry.get("subPath") or [])
            for leaf_path in _leaf_paths(entry["data"], base):
                assert not _has_leaf(initial_data, leaf_path), (
                    "deferred entry re-delivered an initial leaf",
                    leaf_path,
                )

    assert merge_incremental(deferred_payloads) == inline_result["data"]


def _leaf_paths(value, prefix):
    """Yield the absolute path of every scalar leaf in ``value`` (prefixed by ``prefix``)."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaf_paths(child, [*prefix, key])
    else:
        yield prefix


def _has_leaf(data, path):
    """Whether ``data`` already holds a (non-dict) leaf at the absolute ``path``."""
    node = data
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return not isinstance(node, dict)


# ----------------------------------------------------------- stream == inline-list
def test_stream_equals_inline_list():
    stream_schema = GraphQLSchema(
        GraphQLObjectType(
            "Query", {"items": GraphQLField(GraphQLList(GraphQLString))}
        )
    )
    full = ["apple", "banana", "coconut", "date"]

    async def go(query):
        from graphql.execution import ExperimentalIncrementalExecutionResults

        result = experimental_execute_incrementally(
            stream_schema, parse(query), {"items": full}
        )
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, ExperimentalIncrementalExecutionResults):
            payloads = [result.initial_result.formatted]
            async for patch in result.subsequent_results:
                payloads.append(patch.formatted)
            return payloads
        return result.formatted

    streamed = asyncio.run(go("{ items @stream(initialCount: 2) }"))
    inline = asyncio.run(go("{ items }"))

    # initial[:k] + all streamed items == the full inline list.
    initial_items = streamed[0]["data"]["items"]
    streamed_items = []
    for payload in streamed[1:]:
        for entry in payload.get("incremental", []) or []:
            streamed_items.extend(entry["items"])
    assert initial_items + streamed_items == inline["data"]["items"]
    assert inline["data"]["items"] == full


def test_async_stream_equals_inline_list():
    """A @stream'd ASYNC GENERATOR yields the same full list as the inline query, over >1 payload.

    The async-iterator stream invariant: the items arrive over awaits (per-await batching), so the
    streamed delivery uses MORE THAN ONE incremental payload, and initial[:k] + all streamed items
    equals the full inline list (proving the async-iterator drain is data-faithful, not a single
    eager blob).
    """
    full = ["apple", "banana", "coconut", "date"]
    stream_schema = GraphQLSchema(
        GraphQLObjectType(
            "Query", {"items": GraphQLField(GraphQLList(GraphQLString))}
        )
    )

    async def gen(_info):
        for item in full:
            await asyncio.sleep(0)
            yield item

    async def go(query, root):
        from graphql.execution import ExperimentalIncrementalExecutionResults

        result = experimental_execute_incrementally(stream_schema, parse(query), root)
        if asyncio.iscoroutine(result):
            result = await result
        if isinstance(result, ExperimentalIncrementalExecutionResults):
            payloads = [result.initial_result.formatted]
            async for patch in result.subsequent_results:
                payloads.append(patch.formatted)
            return payloads
        return result.formatted

    streamed = asyncio.run(go("{ items @stream(initialCount: 1) }", {"items": gen}))
    inline = asyncio.run(go("{ items }", {"items": list(full)}))

    initial_items = streamed[0]["data"]["items"]
    streamed_items = []
    stream_payloads = 0
    for payload in streamed[1:]:
        entries = payload.get("incremental", []) or []
        if entries:
            stream_payloads += 1
        for entry in entries:
            streamed_items.extend(entry["items"])
    assert initial_items + streamed_items == inline["data"]["items"] == full
    # the async generator awaits between items, so the stream uses more than one payload (per-await
    # batching), not a single eager blob.
    assert stream_payloads > 1


def test_stream_threads_hoisted_columns_into_streamed_items():
    """@stream + hoist=True: streamed child objects still seed their hoisted-out columns.

    Regression for the @stream + hoisting interaction: a @stream'd list of objects whose child plan has a
    request-constant step HOISTED OUT of its layer must thread the HoistBridge into BOTH the inline
    head AND the per-item tail producer — otherwise the streamed Person items run their child layer
    without the hoisted ``tag`` column. ``Person.tag = load_one(constant, …)`` is request-constant
    (hoisted to root); ``Person.id`` reads the per-row boundary (stays). Asserts every head +
    streamed item carries ``tag``, and hoist ON == hoist OFF (the control that always worked).
    """
    from grafast_py import (
        GrafastExecutionContext,
        access,
        constant,
        load_one,
        make_grafast_schema,
    )
    from grafast_py._compat import collect_root_fields
    from grafast_py.completion import find_object_completer
    from grafast_py.plan import plan_operation

    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    sdl = "type Query { people: [Person!]! } type Person { id: Int! tag: Int! }"
    query = "{ people @stream(initialCount: 1) { id tag } }"

    def build_schema():
        def plan_people(parent, args, info):
            return load_one(constant("all"), lambda keys: [rows for _ in keys])

        def plan_tag(parent, args, info):
            return load_one(constant("k"), lambda keys: [100 for _ in keys])

        return make_grafast_schema(
            sdl,
            {
                "Query": {"people": plan_people},
                "Person": {"id": lambda p, a, i: access(p, ("id",)), "tag": plan_tag},
            },
        )

    # sanity: the scenario actually hoists Person.tag OUT of the child layer (so the stream path
    # MUST thread the bridge to recover it) — otherwise this test wouldn't exercise C6 at all.
    class _Ctx(GrafastExecutionContext):
        grafast_config = GrafastConfig(hoist=True)

    document = parse(query)
    operation = document.definitions[0]
    plan_ctx = _Ctx.build(build_schema(), document)
    root_fields = collect_root_fields(plan_ctx, plan_ctx.schema.query_type, operation)
    plan = plan_operation(plan_ctx, operation, plan_ctx.schema.query_type, root_fields)
    child = find_object_completer(plan.fields[0].completer).child_plan
    tag_fp = next(fp for fp in child.fields if fp.response_name == "tag")
    assert tag_fp.step.id in child.layer.hoisted_out_ids

    async def collect(hoist):
        from graphql.execution import ExperimentalIncrementalExecutionResults

        result = experimental_execute_incrementally(
            build_schema(), parse(query), None, config=GrafastConfig(hoist=hoist)
        )
        if asyncio.iscoroutine(result):
            result = await result
        assert isinstance(result, ExperimentalIncrementalExecutionResults)
        items = list(result.initial_result.formatted["data"]["people"])
        async for patch in result.subsequent_results:
            for entry in patch.formatted.get("incremental", []) or []:
                items.extend(entry["items"])
        return items

    on = asyncio.run(collect(hoist=True))
    off = asyncio.run(collect(hoist=False))

    expected = [{"id": 1, "tag": 100}, {"id": 2, "tag": 100}, {"id": 3, "tag": 100}]
    # the bug dropped the bridge → streamed items lost `tag`. Now head + tail recover it: ON == OFF == full.
    assert on == expected
    assert off == expected


# ----------------------------------------------------------- defer-batching proof
def test_defer_relation_fires_same_batch_count_as_non_deferred():
    """A @defer'd loader relation over N parents fires the SAME number of batches as inline.

    The host-parent_step-reuse guarantee: a deferred fragment runs its loader once over all the
    parents of its group, so a defer does not multiply the batch count vs the non-deferred query.
    """
    from grafast_py import access, load_many, make_grafast_schema

    sdl = """
    type Query { heroes: [Hero] }
    type Hero { id: Int! pets: [Pet] }
    type Pet { name: String! }
    """
    heroes = [{"id": 1}, {"id": 2}, {"id": 3}]
    pets_by_hero = {1: [{"name": "a"}], 2: [{"name": "b"}], 3: [{"name": "c"}]}

    def plan_heroes(parent, args, info):
        from grafast_py import constant

        return load_many(constant(0), lambda keys: [heroes])

    def plan_id(parent, args, info):
        return access(parent, ["id"])

    def plan_pets(parent, args, info):
        # one batched load over ALL hero ids in the bucket — fires once per bucket run.
        from grafast_py import lambda_step

        hero_id = access(parent, ["id"])
        return load_many(hero_id, lambda ids: [pets_by_hero[i] for i in ids])

    def plan_name(parent, args, info):
        return access(parent, ["name"])

    batched_schema = make_grafast_schema(
        sdl,
        {
            "Query": {"heroes": plan_heroes},
            "Hero": {"id": plan_id, "pets": plan_pets},
            "Pet": {"name": plan_name},
        },
    )

    def count_pet_batches(query):
        batches = {"n": 0}

        from contextlib import contextmanager

        @contextmanager
        def span(step, count):
            from grafast_py import LoadManyStep

            # count the pets-relation load (a LoadManyStep keyed by hero id). The heroes
            # root load is also a LoadManyStep but keyed off a constant; both are LoadManyStep,
            # so count ALL LoadManyStep batches and compare like-for-like across the two queries.
            if isinstance(step, LoadManyStep):
                batches["n"] += 1
            yield

        config = GrafastConfig(on_step_batch=lambda s, c: span(s, c))
        result = experimental_execute_incrementally(
            batched_schema, parse(query), {}, config=config
        )

        async def drain():
            from graphql.execution import ExperimentalIncrementalExecutionResults

            r = await result if asyncio.iscoroutine(result) else result
            if isinstance(r, ExperimentalIncrementalExecutionResults):
                async for _ in r.subsequent_results:
                    pass

        asyncio.run(drain())
        return batches["n"]

    non_deferred = count_pet_batches(
        "{ heroes { id pets { name } } }"
    )
    deferred = count_pet_batches(
        "{ heroes { id ...P @defer } } fragment P on Hero { pets { name } }"
    )
    assert deferred == non_deferred, (deferred, non_deferred)


# ----------------------------------------------------------- type identity
def test_if_false_and_no_directive_return_plain_execution_result():
    from graphql.execution import (
        ExecutionResult,
        ExperimentalIncrementalExecutionResults,
    )

    if_false = experimental_execute_incrementally(
        schema,
        parse("{ hero { id ...N @defer(if: false) } } fragment N on Hero { name }"),
        {"hero": HERO},
    )
    assert type(if_false) is ExecutionResult

    no_directive = experimental_execute_incrementally(
        schema, parse("{ hero { id name } }"), {"hero": HERO}
    )
    assert type(no_directive) is ExecutionResult

    deferred = experimental_execute_incrementally(
        schema,
        parse("{ hero { id ...N @defer } } fragment N on Hero { name }"),
        {"hero": HERO},
    )
    assert type(deferred) is ExperimentalIncrementalExecutionResults
