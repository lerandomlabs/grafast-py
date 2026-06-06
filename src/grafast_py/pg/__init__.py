"""Postgres / SQLAlchemy data source for grafast-py.

Mirrors Grafast's ``@dataplan/pg``: :class:`PgResource` (table/columns/pk/relations),
batched :class:`PgSelectStep` / :class:`PgSelectSingleStep` (one ``WHERE col = ANY($1)``
statement per bucket), and a batched Relay :class:`PgConnectionStep`. Built on the core
step model, so a depth-D nested query issues ~D batched SQL statements total.
"""

from .connection import PgConnectionStep, connection, connection_needs_total
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
)
from .from_sqlalchemy import (
    columns_from_table,
    resource_from_model,
    resources_from_models,
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
from .resource import PgCodec, PgColumn, PgRegistry, PgRelation, PgResource, as_column
from .steps import (
    PgSelectAllStep,
    PgSelectSingleStep,
    PgSelectStep,
    pg_select,
    pg_select_single,
)

__all__ = [
    "PgResource",
    "PgRelation",
    "PgRegistry",
    "PgColumn",
    "PgCodec",
    "as_column",
    "PgSelectStep",
    "PgSelectSingleStep",
    "PgSelectAllStep",
    "PgConnectionStep",
    "connection",
    "connection_needs_total",
    # keyset cursors: seek-cursor encode/decode + the keyset WHERE comparator
    "encode_keyset_cursor",
    "decode_keyset_cursor",
    "keyset_where",
    "order_digest",
    "pg_select",
    "pg_select_single",
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
]
