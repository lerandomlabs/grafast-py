"""Postgres / SQLAlchemy data source for grafast-py (Phase B).

Mirrors Grafast's ``@dataplan/pg``: :class:`PgResource` (table/columns/pk/relations),
batched :class:`PgSelectStep` / :class:`PgSelectSingleStep` (one ``WHERE col = ANY($1)``
statement per bucket), and a batched Relay :class:`PgConnectionStep`. Built on the
Phase A step model, so a depth-D nested query issues ~D batched SQL statements total.
"""

from .connection import PgConnectionStep, connection, decode_cursor, encode_cursor
from .engine import configure_engine, count_sql, dispose_engine, get_engine
from .from_sqlalchemy import resource_from_model, resources_from_models
from .resource import PgRegistry, PgRelation, PgResource
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
    "PgSelectStep",
    "PgSelectSingleStep",
    "PgSelectAllStep",
    "PgConnectionStep",
    "connection",
    "encode_cursor",
    "decode_cursor",
    "pg_select",
    "pg_select_single",
    "get_engine",
    "configure_engine",
    "dispose_engine",
    "count_sql",
    "resource_from_model",
    "resources_from_models",
]
