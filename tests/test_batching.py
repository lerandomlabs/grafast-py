"""End-to-end batching proof: a LIVE query exercises the plan-resolver step DAG.

The unit tests in `test_core_steps.py` prove each step batches in isolation; this
file proves the WHOLE engine batches when an actual GraphQL query runs through
`GrafastExecutionContext`. A field carries a plan resolver
(`extensions["grafast"]["plan"]`) that returns a `loadMany` / `loadOne` step; the
executor must run that step ONCE over the whole bucket, so the user batch-load
callback fires exactly once per layer — verified with a hard call counter — not once
per parent (the N+1 it replaces).

The schema/data are static plain dicts (no database — the batch model is source-agnostic);
the loaders are plain Python callbacks with counters. We assert BOTH the data and
the batch-call counts.
"""

from typing import Any, List

import pytest
from graphql import execute, parse

from grafast_py import (
    GrafastExecutionContext,
    access,
    each,
    load_many,
    load_one,
    make_grafast_schema,
)

# ----------------------------------------------------------------- static data
USERS = {
    1: {"id": 1, "name": "Luke", "friend_ids": [2, 3]},
    2: {"id": 2, "name": "Han", "friend_ids": [1, 3]},
    3: {"id": 3, "name": "Leia", "friend_ids": [1, 2]},
    4: {"id": 4, "name": "Chewie", "friend_ids": [2]},
}

POSTS = {
    1: [{"id": 11, "title": "a"}, {"id": 12, "title": "b"}],
    2: [{"id": 21, "title": "c"}],
    3: [],
    4: [{"id": 41, "title": "d"}],
}

SDL = """
type Query {
  users: [User!]!
  user(id: Int!): User
}

type User {
  id: Int!
  name: String!
  friends: [User!]!
  posts: [Post!]!
}

type Post {
  id: Int!
  title: String!
}
"""


def make_schema(load_users, load_posts):
    """Build the schema wiring plan resolvers onto the batched relations.

    `users` returns the whole user list (a root constant-ish step via a loader keyed
    on a sentinel). `friends` and `posts` are the batched relations: each parent
    contributes its key(s) and the loader is invoked once for the whole bucket.
    """

    def plan_users(parent, args, info):
        # one batch call returning all users; key is a constant so dedup/scatter is
        # trivial — the point here is the nested relations below.
        from grafast_py import constant

        return load_one(constant("__all__"), lambda keys: [list(USERS.values())])

    def plan_user(parent, args, info):
        from grafast_py import constant

        user_id = args["id"]
        return load_one(constant(user_id), load_users)

    def plan_id(parent, args, info):
        return access(parent, ["id"])

    def plan_name(parent, args, info):
        return access(parent, ["name"])

    def plan_friends(parent, args, info):
        # each parent's friend_ids -> a flat batched loadMany over the union of ids
        friend_ids = access(parent, ["friend_ids"])
        return each(friend_ids, lambda item: load_one(item, load_users))

    def plan_posts(parent, args, info):
        # one loadMany over every parent user's id; returns that user's post list
        return load_many(access(parent, ["id"]), load_posts)

    return make_grafast_schema(
        SDL,
        {
            "Query": {"users": plan_users, "user": plan_user},
            "User": {
                "id": plan_id,
                "name": plan_name,
                "friends": plan_friends,
                "posts": plan_posts,
            },
            "Post": {"id": plan_id, "title": access_title()},
        },
    )


def access_title():
    def plan_title(parent, args, info):
        return access(parent, ["title"])

    return plan_title


def run(schema, query, variables=None):
    result = execute(
        schema,
        parse(query),
        execution_context_class=GrafastExecutionContext,
        variable_values=variables,
    )
    assert not (getattr(result, "errors", None)), result.errors
    return result.data


# --------------------------------------------------------------------- N parents
def test_load_many_over_n_parents_calls_batch_fn_exactly_once():
    """posts is a loadMany over every user; ONE call, not N (the N+1 killer)."""
    user_calls = {"n": 0}
    post_calls = {"n": 0, "keys": None}

    def load_users(keys: List[Any]) -> List[Any]:
        user_calls["n"] += 1
        return [USERS.get(k) for k in keys]

    def load_posts(keys: List[Any]) -> List[Any]:
        post_calls["n"] += 1
        post_calls["keys"] = list(keys)
        return [POSTS.get(k, []) for k in keys]

    schema = make_schema(load_users, load_posts)

    data = run(
        schema,
        """
        { users { id name posts { id title } } }
        """,
    )

    # THE GATE: posts (a loadMany) fired ONCE for all 4 users, not 4 times.
    assert post_calls["n"] == 1
    assert sorted(post_calls["keys"]) == [1, 2, 3, 4]

    by_id = {u["id"]: u for u in data["users"]}
    assert by_id[1]["posts"] == [
        {"id": 11, "title": "a"},
        {"id": 12, "title": "b"},
    ]
    assert by_id[2]["posts"] == [{"id": 21, "title": "c"}]
    assert by_id[3]["posts"] == []
    assert by_id[4]["posts"] == [{"id": 41, "title": "d"}]


def test_nested_friends_loadone_batches_across_all_parents():
    """friends -> loadOne per friend id; ONE batch call over the union of all ids."""
    user_calls = {"n": 0, "keys": None}

    def load_users(keys: List[Any]) -> List[Any]:
        user_calls["n"] += 1
        user_calls["keys"] = list(keys)
        return [USERS.get(k) for k in keys]

    def load_posts(keys):
        return [POSTS.get(k, []) for k in keys]

    schema = make_schema(load_users, load_posts)

    data = run(
        schema,
        """
        { users { name friends { name } } }
        """,
    )

    # `users` itself uses a separate loader instance, so `load_users` here only
    # serves the `friends` relation: the `each(loadOne)` over every friend of every
    # user runs ONCE across the whole flattened friend bucket.
    assert user_calls["n"] == 1
    # union of all friend ids across all four users (deduped): {1,2,3}
    assert sorted(set(user_calls["keys"])) == [1, 2, 3]

    by_name = {u["name"]: u for u in data["users"]}
    assert sorted(f["name"] for f in by_name["Luke"]["friends"]) == ["Han", "Leia"]
    assert sorted(f["name"] for f in by_name["Chewie"]["friends"]) == ["Han"]


def test_single_user_loadone_one_call():
    """A single root loadOne fires its batch callback exactly once."""
    calls = {"n": 0}

    def load_users(keys):
        calls["n"] += 1
        return [USERS.get(k) for k in keys]

    schema = make_schema(load_users, lambda keys: [[] for _ in keys])
    data = run(schema, "{ user(id: 2) { id name } }")

    assert calls["n"] == 1
    assert data == {"user": {"id": 2, "name": "Han"}}


def test_two_relations_each_batch_once_in_one_query():
    """friends (loadOne) and posts (loadMany) each fire once — independent layers."""
    user_calls = {"n": 0}
    post_calls = {"n": 0}

    def load_users(keys):
        user_calls["n"] += 1
        return [USERS.get(k) for k in keys]

    def load_posts(keys):
        post_calls["n"] += 1
        return [POSTS.get(k, []) for k in keys]

    schema = make_schema(load_users, load_posts)

    data = run(
        schema,
        """
        { users { name friends { name } posts { title } } }
        """,
    )

    assert user_calls["n"] == 1  # friends loadOne batched once
    assert post_calls["n"] == 1  # posts loadMany batched once
    assert len(data["users"]) == 4


# --------------------------------------------------------------------- dedup
def test_cross_step_dedup_collapses_identical_loaders_in_a_live_query():
    """Two fields loading the SAME key with the SAME loader merge to one batch call.

    `a` and `b` both `loadOne(access(parent, ['best_friend_id']), load_users)`. The
    planner's cross-step dedup recognises them as structurally identical (same class,
    same dependency access step, same loader) and collapses them to one step, so the
    batch callback fires ONCE for the whole bucket, not once per field.
    """
    from grafast_py import constant

    data = {
        1: {"id": 1, "name": "Luke", "best_friend_id": 2},
        2: {"id": 2, "name": "Han", "best_friend_id": 1},
    }
    calls = {"n": 0}

    def load_users(keys):
        calls["n"] += 1
        return [data.get(k) for k in keys]

    sdl = """
    type Query { users: [User!]! }
    type User { id: Int! a: User b: User }
    """

    def plan_users(p, a, i):
        return load_one(constant("all"), lambda k: [list(data.values())])

    def plan_id(p, a, i):
        return access(p, ["id"])

    def plan_a(p, a, i):
        return load_one(access(p, ["best_friend_id"]), load_users)

    def plan_b(p, a, i):
        return load_one(access(p, ["best_friend_id"]), load_users)

    schema = make_grafast_schema(
        sdl,
        {
            "Query": {"users": plan_users},
            "User": {"id": plan_id, "a": plan_a, "b": plan_b},
        },
    )
    result = run(schema, "{ users { a { id } b { id } } }")

    # one batch call despite two fields requesting the same key+loader
    assert calls["n"] == 1
    assert result == {
        "users": [
            {"a": {"id": 2}, "b": {"id": 2}},
            {"a": {"id": 1}, "b": {"id": 1}},
        ]
    }


# ------------------------------------------------------------------------- async
@pytest.mark.asyncio
async def test_async_loadmany_batches_once():
    """An async batch loader is still invoked exactly once over the bucket."""
    import asyncio

    post_calls = {"n": 0}

    def load_users(keys):
        return [USERS.get(k) for k in keys]

    async def load_posts(keys):
        post_calls["n"] += 1
        await asyncio.sleep(0)
        return [POSTS.get(k, []) for k in keys]

    schema = make_schema(load_users, load_posts)

    result = execute(
        schema,
        parse("{ users { id posts { title } } }"),
        execution_context_class=GrafastExecutionContext,
    )
    if hasattr(result, "__await__"):
        result = await result
    assert not result.errors, result.errors

    assert post_calls["n"] == 1
    by_id = {u["id"]: u for u in result.data["users"]}
    assert by_id[1]["posts"] == [{"title": "a"}, {"title": "b"}]
