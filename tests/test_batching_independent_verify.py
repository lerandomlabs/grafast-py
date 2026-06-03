"""Independent Phase-A batching verification (final-verification pass).

This file is written by the verification stage to MEASURE — not assume — that a
``loadMany`` over N parents invokes its batch-load callback EXACTLY ONCE. The
counter lives inside the user batch callback, so the assertion is a direct
observation of how many times the engine called the loader. We also:

  * scale N up (N=50) and confirm the call count stays 1 (a per-parent fallback
    would make it 50);
  * capture the keys the loader received and confirm they are the WHOLE bucket in
    one call (proving fan-in, not N separate single-key calls);
  * record an explicit N+1 baseline (a naive per-parent resolver over the same
    data fires N times) so the contrast is on the record.
"""

from typing import Any, List

from graphql import execute, graphql_sync, parse

from grafast_py import (
    GrafastExecutionContext,
    access,
    load_many,
    make_grafast_schema,
)
from graphql import GraphQLObjectType, GraphQLSchema, GraphQLField, GraphQLList, GraphQLNonNull, GraphQLInt, GraphQLString

SDL = """
type Query { users: [User!]! }
type User { id: Int!  posts: [Post!]! }
type Post { id: Int!  title: String! }
"""


def build(n: int, counter: dict):
    users = [{"id": i, "name": f"u{i}"} for i in range(1, n + 1)]
    posts = {i: [{"id": i * 100 + j, "title": f"p{i}-{j}"} for j in range(2)] for i in range(1, n + 1)}

    def load_posts(keys: List[Any]) -> List[Any]:
        counter["n"] += 1
        counter["keys"] = list(keys)
        return [posts.get(k, []) for k in keys]

    def plan_users(p, a, i):
        from grafast_py import constant, load_one

        return load_one(constant("all"), lambda k: [users])

    def plan_posts(p, a, i):
        return load_many(access(p, ["id"]), load_posts)

    schema = make_grafast_schema(
        SDL,
        {
            "Query": {"users": plan_users},
            "User": {"id": lambda p, a, i: access(p, ["id"]), "posts": plan_posts},
            "Post": {
                "id": lambda p, a, i: access(p, ["id"]),
                "title": lambda p, a, i: access(p, ["title"]),
            },
        },
    )
    return schema, users, posts


def test_loadmany_over_50_parents_is_one_call():
    counter = {"n": 0, "keys": None}
    schema, users, posts = build(50, counter)

    result = execute(
        schema,
        parse("{ users { id posts { id title } } }"),
        execution_context_class=GrafastExecutionContext,
    )
    assert not result.errors, result.errors

    # THE GATE, independently observed: ONE batch call for all 50 parents.
    assert counter["n"] == 1, f"expected 1 batch call, got {counter['n']}"
    # the loader saw the entire bucket in that single call:
    assert sorted(counter["keys"]) == list(range(1, 51))

    by_id = {u["id"]: u for u in result.data["users"]}
    assert len(by_id) == 50
    assert by_id[7]["posts"] == [
        {"id": 700, "title": "p7-0"},
        {"id": 701, "title": "p7-1"},
    ]


def test_n_plus_1_baseline_with_naive_resolvers_fires_n_times():
    """Control: a plain per-parent resolver over the same shape fires N times.

    This is the N+1 the plan path eliminates. Built with stock graphql-core types
    and run via graphql_sync (NOT our engine) to record the unbatched count.
    """
    n = 50
    users = [{"id": i} for i in range(1, n + 1)]
    posts = {i: [{"id": i * 100, "title": "p"}] for i in range(1, n + 1)}
    calls = {"n": 0}

    def resolve_posts(user, info):
        calls["n"] += 1  # once PER parent — the N+1
        return posts[user["id"]]

    post_t = GraphQLObjectType(
        "Post",
        {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt)),
            "title": GraphQLField(GraphQLNonNull(GraphQLString)),
        },
    )
    user_t = GraphQLObjectType(
        "User",
        {
            "id": GraphQLField(GraphQLNonNull(GraphQLInt)),
            "posts": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(post_t))),
                resolve=resolve_posts,
            ),
        },
    )
    query_t = GraphQLObjectType(
        "Query",
        {
            "users": GraphQLField(
                GraphQLNonNull(GraphQLList(GraphQLNonNull(user_t))),
                resolve=lambda root, info: users,
            )
        },
    )
    schema = GraphQLSchema(query=query_t)

    res = graphql_sync(schema, "{ users { id posts { id title } } }")
    assert res.errors is None
    # the naive path: one resolver call per parent.
    assert calls["n"] == n
