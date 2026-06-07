"""Unit tests for the inline substrate: InlineSpec + NestedExtractStep.

These pin the two data structures the LATERAL fold survives into execution as, in
ISOLATION — no optimize wiring, no DB. A :class:`NestedExtractStep` is fed FAKE parent
rows carrying a nested ``json`` column (exactly what asyncpg materialises from a
``LEFT JOIN LATERAL (... json_agg/to_jsonb ...) AS <alias>``), and we assert it scatters
the child rows the standalone batched path WOULD have scattered:

- hasMany -> the decoded list, defaulting ``[]`` when the column is null / absent;
- hasOne  -> the single decoded dict, or ``None`` when the column is null / absent;
- codecs decode through the SAME ``resource.decode_rows`` as the batched path, so a
  codec'd column round-trips identically.

The byte-identical equivalence-vs-batched proof is DB-backed and lives elsewhere; here
we only prove the extract step reproduces the scatter shape and the decode, since that
is the contract the fold rests on.
"""

import pytest

from grafast_py.pg.inline import (
    KIND_HAS_MANY,
    KIND_HAS_ONE,
    InlineSpec,
    NestedExtractStep,
    inline_spec_from_relation,
)
from grafast_py.pg.resource import PgCodec, PgColumn, PgRegistry, PgResource
from grafast_py.step_model import Step, run_steps


def never_awaitable(_value):
    return False


class SourceStep(Step):
    """A 0-dependency source seeded with a fixed column of parent row dicts."""

    def __init__(self, column):
        super().__init__()
        self.column = column

    def execute(self, count, values):
        return self.column


def make_resource(columns=("id", "title"), name="post"):
    """A bare child resource (the relation TARGET whose rows the nested column carries)."""
    return PgResource(name, "grafast_demo", name, list(columns), registry=PgRegistry())


def run_extract(parent_rows, alias, resource, kind):
    """Wire a SourceStep(parent_rows) -> NestedExtractStep and run one bucket; return column."""
    source = SourceStep(parent_rows)
    extract = NestedExtractStep(source, alias, resource, kind)
    source.id = 0
    extract.id = 1
    results = run_steps(len(parent_rows), [source, extract], never_awaitable)
    return results[extract.id]


# ----------------------------------------------------------------- InlineSpec shape


def test_inline_spec_records_the_fold():
    """A spec carries the child resource, kind, FK columns and the nested column name."""
    resource = make_resource()
    spec = InlineSpec(
        resource=resource,
        kind=KIND_HAS_MANY,
        nested_alias="__posts",
        local_columns=("id",),
        remote_columns=("author_id",),
    )
    assert spec.resource is resource
    assert spec.kind == KIND_HAS_MANY
    assert spec.is_has_many is True
    assert spec.nested_alias == "__posts"
    assert spec.local_columns == ("id",)
    assert spec.remote_columns == ("author_id",)


def test_inline_spec_hasone_is_not_has_many():
    spec = InlineSpec(
        resource=make_resource(("id", "name"), name="author"),
        kind=KIND_HAS_ONE,
        nested_alias="__author",
        local_columns=("author_id",),
        remote_columns=("id",),
    )
    assert spec.is_has_many is False


def test_inline_spec_is_frozen_and_hashable():
    """Frozen so it slots into the parent step's dedup key as a stable component."""
    spec = InlineSpec(
        resource=make_resource(),
        kind=KIND_HAS_MANY,
        nested_alias="__posts",
        local_columns=("id",),
        remote_columns=("author_id",),
    )
    assert hash(spec) == hash(spec)
    with pytest.raises((AttributeError, TypeError)):
        spec.nested_alias = "other"  # frozen dataclass rejects mutation


def test_inline_spec_equality_ignores_resource_identity_but_keys_on_columns():
    """Two specs with the same kind/alias/FK columns compare equal (dedup-mergeable).

    ``resource`` rides ``compare=False`` (a resource has no value equality), so equality is
    driven by the dedup-relevant fields — the alias, kind and FK column tuples that change
    the emitted LATERAL. Different FK columns therefore do NOT compare equal.
    """
    base = dict(kind=KIND_HAS_MANY, nested_alias="__posts")
    a = InlineSpec(resource=make_resource(), local_columns=("id",),
                   remote_columns=("author_id",), **base)
    b = InlineSpec(resource=make_resource(), local_columns=("id",),
                   remote_columns=("author_id",), **base)
    c = InlineSpec(resource=make_resource(), local_columns=("id",),
                   remote_columns=("editor_id",), **base)
    assert a == b
    assert a != c


def test_inline_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        InlineSpec(
            resource=make_resource(),
            kind="has_few",
            nested_alias="__posts",
            local_columns=("id",),
            remote_columns=("author_id",),
        )


def test_inline_spec_rejects_mismatched_column_tuples():
    with pytest.raises(ValueError, match="local/remote"):
        InlineSpec(
            resource=make_resource(),
            kind=KIND_HAS_MANY,
            nested_alias="__posts",
            local_columns=("id",),
            remote_columns=("a", "b"),
        )


def test_inline_spec_rejects_empty_columns():
    with pytest.raises(ValueError, match="local/remote"):
        InlineSpec(
            resource=make_resource(),
            kind=KIND_HAS_MANY,
            nested_alias="__posts",
            local_columns=(),
            remote_columns=(),
        )


def test_inline_spec_from_relation_lifts_relation_fields():
    """The convenience builder lifts a PgRelation's target/kind/FK tuples into the spec."""
    registry = PgRegistry()
    author = PgResource("author", "grafast_demo", "author", ["id", "name"],
                        registry=registry)
    post = PgResource("post", "grafast_demo", "post", ["id", "author_id", "title"],
                      registry=registry)
    relation = author.has_many("posts", post, local_column="id",
                               remote_column="author_id")
    spec = inline_spec_from_relation(relation, "__posts")
    assert spec.resource is post
    assert spec.kind == KIND_HAS_MANY
    assert spec.local_columns == ("id",)
    assert spec.remote_columns == ("author_id",)
    assert spec.nested_alias == "__posts"


# -------------------------------------------------------- NestedExtractStep: hasMany


def test_extract_has_many_scatters_the_nested_list():
    """Each parent's nested list is scattered as that entry's child rows, in order."""
    resource = make_resource()
    parent_rows = [
        {"id": 1, "__posts": [{"id": 10, "title": "a"}, {"id": 11, "title": "b"}]},
        {"id": 2, "__posts": [{"id": 20, "title": "c"}]},
    ]
    out = run_extract(parent_rows, "__posts", resource, KIND_HAS_MANY)
    assert out == [
        [{"id": 10, "title": "a"}, {"id": 11, "title": "b"}],
        [{"id": 20, "title": "c"}],
    ]


def test_extract_has_many_empty_children_is_empty_list():
    """A parent with no children yields [] — exactly the batched path's empty scatter.

    Covers BOTH an explicit empty list (coalesce(json_agg, '[]') emitted []) and a null /
    absent column (a row the LATERAL produced no join for), since both mean no children.
    """
    resource = make_resource()
    parent_rows = [
        {"id": 1, "__posts": []},
        {"id": 2, "__posts": None},
        {"id": 3},  # column absent entirely
    ]
    out = run_extract(parent_rows, "__posts", resource, KIND_HAS_MANY)
    assert out == [[], [], []]


def test_extract_has_many_is_an_independent_list_copy():
    """decode_rows returns fresh dicts, so mutating the output cannot corrupt a parent row."""
    resource = make_resource()
    parent_rows = [{"id": 1, "__posts": [{"id": 10, "title": "a"}]}]
    out = run_extract(parent_rows, "__posts", resource, KIND_HAS_MANY)
    out[0][0]["title"] = "mutated"
    assert parent_rows[0]["__posts"][0]["title"] == "a"


# -------------------------------------------------------- NestedExtractStep: hasOne


def test_extract_has_one_scatters_the_single_dict():
    resource = make_resource(("id", "name"), name="author")
    parent_rows = [
        {"author_id": 7, "__author": {"id": 7, "name": "ada"}},
        {"author_id": 8, "__author": {"id": 8, "name": "lin"}},
    ]
    out = run_extract(parent_rows, "__author", resource, KIND_HAS_ONE)
    assert out == [{"id": 7, "name": "ada"}, {"id": 8, "name": "lin"}]


def test_extract_has_one_missing_is_none():
    """A hasOne whose FK points nowhere yields None — the batched path's None scatter."""
    resource = make_resource(("id", "name"), name="author")
    parent_rows = [
        {"author_id": None, "__author": None},
        {"author_id": 9},  # column absent entirely
    ]
    out = run_extract(parent_rows, "__author", resource, KIND_HAS_ONE)
    assert out == [None, None]


# ------------------------------------------------------------------ codec decode


def test_extract_decodes_codec_columns_like_the_batched_path():
    """A codec'd nested column decodes through resource.decode_rows identically.

    The nested json carries the RAW stored value (``"abc"``); the resource's ``to_py``
    (uppercasing here) must run on extract so the presented value matches what the
    standalone batched select would have decoded — the codec round-trip the fold preserves.
    """
    resource = PgResource(
        "label",
        "grafast_demo",
        "label",
        ["id", PgColumn("code", codec=PgCodec(to_py=str.upper))],
        registry=PgRegistry(),
    )
    parent_rows = [
        {"id": 1, "__labels": [{"id": 5, "code": "abc"}, {"id": 6, "code": "xy"}]},
    ]
    out = run_extract(parent_rows, "__labels", resource, KIND_HAS_MANY)
    assert out == [[{"id": 5, "code": "ABC"}, {"id": 6, "code": "XY"}]]
    # the standalone path's decode_rows produces the identical decoded rows.
    assert out[0] == resource.decode_rows(parent_rows[0]["__labels"])


def test_extract_has_one_decodes_codec_column():
    resource = PgResource(
        "label",
        "grafast_demo",
        "label",
        ["id", PgColumn("code", codec=PgCodec(to_py=str.upper))],
        registry=PgRegistry(),
    )
    parent_rows = [{"id": 1, "__label": {"id": 5, "code": "abc"}}]
    out = run_extract(parent_rows, "__label", resource, KIND_HAS_ONE)
    assert out == [{"id": 5, "code": "ABC"}]


# -------------------------------------------------------------- step contract


def test_extract_step_is_sync_and_one_dependency():
    """No DB round-trip: it reads off the parent row, so it is sync with a single dep."""
    resource = make_resource()
    source = SourceStep([{"id": 1, "__posts": []}])
    extract = NestedExtractStep(source, "__posts", resource, KIND_HAS_MANY)
    assert extract.is_sync_and_safe is True
    assert extract.dependency_count == 1
    assert extract.dependencies[0] is source


def test_extract_step_rejects_unknown_kind():
    resource = make_resource()
    source = SourceStep([])
    with pytest.raises(ValueError, match="kind"):
        NestedExtractStep(source, "__posts", resource, "has_few")


def test_extract_step_dedup_key_discriminates_alias_kind_resource():
    """Two extracts merge only on the same nested column, kind and child resource."""
    resource = make_resource()
    source = SourceStep([])
    a = NestedExtractStep(source, "__posts", resource, KIND_HAS_MANY)
    b = NestedExtractStep(source, "__posts", resource, KIND_HAS_MANY)
    c = NestedExtractStep(source, "__comments", resource, KIND_HAS_MANY)
    d = NestedExtractStep(source, "__posts", resource, KIND_HAS_ONE)
    assert a.peer_key == b.peer_key
    assert a.dedup_params() == b.dedup_params()
    assert a.peer_key != c.peer_key
    assert a.peer_key != d.peer_key
