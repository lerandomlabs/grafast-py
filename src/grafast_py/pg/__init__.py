"""Postgres / SQLAlchemy data source for grafast-py.

Mirrors Grafast's ``@dataplan/pg``: :class:`PgResource` (table/columns/pk/relations),
batched :class:`PgSelectStep` / :class:`PgSelectSingleStep` (one ``WHERE col = ANY($1)``
statement per bucket), and a batched Relay :class:`PgConnectionStep`. Built on the core
step model, so a depth-D nested query issues ~D batched SQL statements total.
"""

from .codecs import (
    apply_to_py,
    array_codec,
    codec_for,
    composite_codec,
    enum_codec,
    range_codec,
    scalar_codecs,
)
from .conditions import And, Compare, Condition, Not, Or, compile_condition
from .connection import (
    AGGREGATE_FUNCTIONS,
    PgAggregate,
    PgConnectionStep,
    connection,
    connection_aggregates,
    connection_needs_total,
)
from .cursor import (
    decode_keyset_cursor,
    encode_keyset_cursor,
    keyset_where,
    order_digest,
)
from .customize import (
    PgCustomizable,
    PgSelectQueryBuilder,
    check_predicate,
    predicate_key,
    resolve_customizer_predicates,
)
from .engine import configure_engine, count_sql, dispose_engine, get_engine
from .executor import (
    PgExecutor,
    PgRequestContext,
    RawExecutor,
    SQLAlchemyExecutor,
    current_pg_request,
    pg_request_context,
    pg_request_context_async,
)
from .from_sqlalchemy import (
    columns_from_table,
    resource_from_model,
    resources_from_models,
)
# LATERAL relation inlining substrate: the fold spec + nested-row extract step the
# parent's optimize/build_query reproduce a child relation through.
from .inline import (
    InlineSpec,
    NestedExtractStep,
    inline_spec_from_relation,
)
from .mutations import (
    PgDeleteSingleStep,
    PgInsertSingleStep,
    PgUpdateSingleStep,
    pg_delete_single,
    pg_insert_single,
    pg_update_single,
)
from .ordering import OrderTerm, normalize_order, order_clauses
# value-agnostic placeholders: a source-tagged WHERE bindparam (`pg_placeholder`) and a
# `Placeholder` sentinel for a variable-derived pagination value (first/offset/after/
# before), both dedup-keyed by their stable source tag, not their runtime value.
from .placeholders import (
    Placeholder,
    pg_placeholder,
    placeholder_source,
    placeholder_source_tag,
    resolve_placeholder,
)
from .resource import (
    PgCodec,
    PgColumn,
    PgRegistry,
    PgRelation,
    PgResource,
    as_column,
    match_columns_tuple,
    relation_columns_pair,
    relation_key_step,
)
from .steps import (
    PgSelectAllStep,
    PgSelectSingleStep,
    PgSelectStep,
    pg_select,
    pg_select_single,
)
# cross-table union: pgUnionAll over N member tables (UNION ALL + keyset + separate count)
from .union import (
    PgUnionAllStep,
    PgUnionMember,
    pg_union_all,
    union_all_connection,
)

__all__ = [
    "PgResource",
    "PgRelation",
    "PgRegistry",
    "PgColumn",
    "PgCodec",
    "as_column",
    # composite keys: single match_column generalised to a match_columns tuple
    "match_columns_tuple",
    "relation_columns_pair",
    "relation_key_step",
    "PgSelectStep",
    "PgSelectSingleStep",
    "PgSelectAllStep",
    # LATERAL relation inlining substrate
    "InlineSpec",
    "NestedExtractStep",
    "inline_spec_from_relation",
    "PgConnectionStep",
    "connection",
    "connection_needs_total",
    # connection domain aggregates: a separate batched sum/avg/min/max/count statement
    "PgAggregate",
    "AGGREGATE_FUNCTIONS",
    "connection_aggregates",
    # keyset cursors: seek-cursor encode/decode + the keyset WHERE comparator
    "encode_keyset_cursor",
    "decode_keyset_cursor",
    "keyset_where",
    "order_digest",
    "pg_select",
    "pg_select_single",
    # cross-table union: pgUnionAll over N member tables
    "PgUnionAllStep",
    "PgUnionMember",
    "pg_union_all",
    "union_all_connection",
    # single-row CRUD mutation steps: the serial-seam write counterpart
    "PgInsertSingleStep",
    "PgUpdateSingleStep",
    "PgDeleteSingleStep",
    "pg_insert_single",
    "pg_update_single",
    "pg_delete_single",
    "get_engine",
    "configure_engine",
    "dispose_engine",
    "count_sql",
    "PgExecutor",
    "PgRequestContext",
    "SQLAlchemyExecutor",
    "RawExecutor",
    "pg_request_context",
    # shared_txn: opt-in single-connection / single REPEATABLE READ transaction per request
    "pg_request_context_async",
    "current_pg_request",
    "resource_from_model",
    "resources_from_models",
    "columns_from_table",
    "OrderTerm",
    "normalize_order",
    "order_clauses",
    # query customization: host-facing WHERE + builder seam
    "PgSelectQueryBuilder",
    "PgCustomizable",
    "check_predicate",
    "predicate_key",
    "resolve_customizer_predicates",
    # value-agnostic placeholders for a variable-derived value: WHERE bindparam +
    # the pagination-value (first/offset/after/before) sentinel
    "pg_placeholder",
    "placeholder_source",
    "Placeholder",
    "resolve_placeholder",
    "placeholder_source_tag",
    # codec registry: PgCodec lookup by pg type name + recursive array/range/composite/enum
    "codec_for",
    "array_codec",
    "range_codec",
    "composite_codec",
    "enum_codec",
    "apply_to_py",
    "scalar_codecs",
    # structured filter AST: a Condition tree compiling to a Core boolean WHERE predicate
    "Condition",
    "Compare",
    "And",
    "Or",
    "Not",
    "compile_condition",
]
