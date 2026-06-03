"""In-memory plan-resolver example (no database required).

Run it:

    uv run python examples/plan_blog.py

Shows the Grafast value-prop with plain Python data: a `loadMany` plan resolver
batches the `posts` relation across ALL authors into a SINGLE call, instead of the
classic per-author N+1. A call counter inside the loader proves it.
"""

from graphql import execute, parse

from grafast_py import (
    GrafastExecutionContext,
    access,
    load_many,
    make_grafast_schema,
)

AUTHORS = [
    {"id": 1, "name": "Ada"},
    {"id": 2, "name": "Alan"},
    {"id": 3, "name": "Grace"},
]
POSTS_BY_AUTHOR = {
    1: [{"id": 11, "title": "On Analytical Engines"}],
    2: [{"id": 21, "title": "Computing Machinery"}, {"id": 22, "title": "Turing Test"}],
    3: [{"id": 31, "title": "Compilers"}],
}

SDL = """
type Query {
  authors: [Author!]!
}
type Author {
  id: Int!
  name: String!
  posts: [Post!]!
}
type Post {
  id: Int!
  title: String!
}
"""

# a counter INSIDE the user callback: a direct observation that the loader fired once.
LOAD_CALLS = {"posts": 0}


def build_schema():
    def plan_authors(parent, args, info):
        from grafast_py import constant, load_one

        # one batch returning the whole author list (root list).
        return load_one(constant(0), lambda keys: [AUTHORS])

    def plan_posts(parent, args, info):
        # ONE loadMany over every author's id; the loader receives all ids at once.
        def load_posts(author_ids):
            LOAD_CALLS["posts"] += 1
            print(f"  load_posts called once with author_ids={list(author_ids)}")
            return [POSTS_BY_AUTHOR.get(aid, []) for aid in author_ids]

        return load_many(access(parent, ["id"]), load_posts)

    def field(name):
        return lambda parent, args, info: access(parent, [name])

    return make_grafast_schema(
        SDL,
        {
            "Query": {"authors": plan_authors},
            "Author": {"id": field("id"), "name": field("name"), "posts": plan_posts},
            "Post": {"id": field("id"), "title": field("title")},
        },
    )


def main():
    schema = build_schema()
    query = "{ authors { id name posts { id title } } }"
    result = execute(
        schema, parse(query), execution_context_class=GrafastExecutionContext
    )
    assert not result.errors, result.errors
    print("data:", result.data)
    print(f"posts loader call count: {LOAD_CALLS['posts']} (expected 1, not 3)")
    assert LOAD_CALLS["posts"] == 1, "the loadMany should fire exactly once for the bucket"
    print("OK: the N+1 collapsed to a single batched call.")


if __name__ == "__main__":
    main()
