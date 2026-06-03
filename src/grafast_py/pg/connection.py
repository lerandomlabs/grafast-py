"""Relay connection step, batched across parents via window functions.

:class:`PgConnectionStep` wraps a hasMany lookup and produces, per bucket entry, a
Relay connection dict ``{"edges": [...], "nodes": [...], "totalCount": n,
"pageInfo": {...}}``. The decisive property — paging EVERY parent's connection in ONE
SQL statement — is achieved with window functions partitioned by the match column:

    SELECT <cols>,
           row_number() OVER (PARTITION BY <match> ORDER BY <order>) AS __rn,
           count(*)     OVER (PARTITION BY <match>)                  AS __total
    FROM <schema>.<table>
    WHERE <match> = ANY($1)

The ``__rn`` / ``__total`` window columns let us slice each parent's page
(``after_offset < rn <= after_offset + first``) and compute ``totalCount`` /
``hasNextPage`` in Python without a second round-trip. So a connection layer is still
ONE statement across all parents — the same O(depth) guarantee as a plain
``pg_select``.

Cursors are opaque, offset-based (``base64("pgcursor:" + rn)``), which satisfies the
Relay contract for the demo. Connection sub-fields (``totalCount``, ``pageInfo``,
``edges { node }``) are plain :class:`AccessStep` projections into the per-entry dict;
``edges[].node`` is the row dict, so leaf access and nested relations under a node
batch exactly like a plain row.
"""

import base64
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import any_, bindparam, column, func, select, table

from ..step_model import Step
from .engine import get_engine
from .resource import PgResource

_CURSOR_PREFIX = "pgcursor:"


def encode_cursor(row_number: int) -> str:
    """Encode an offset-based Relay cursor."""
    return base64.b64encode(f"{_CURSOR_PREFIX}{row_number}".encode()).decode()


def decode_cursor(cursor: Optional[str]) -> int:
    """Decode an offset cursor to its row number, or 0 when absent/invalid."""
    if not cursor:
        return 0
    try:
        raw = base64.b64decode(cursor.encode()).decode()
    except (ValueError, UnicodeDecodeError):
        return 0
    if not raw.startswith(_CURSOR_PREFIX):
        return 0
    try:
        return int(raw[len(_CURSOR_PREFIX) :])
    except ValueError:
        return 0


class PgConnectionStep(Step):
    """Batched Relay connection over a hasMany lookup keyed on ``match_column``."""

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        key_step: Step,
        match_column: str,
        order_by: Sequence[str],
        first: Optional[int] = None,
        after: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.resource = resource
        self.match_column = match_column
        self.order_by: Tuple[str, ...] = tuple(order_by or (resource.primary_key,))
        self.first = first
        self.after_offset = decode_cursor(after)
        self.add_dependency(key_step)

    def build_query(self):
        """Build the window-partitioned ``= ANY($1)`` SELECT via SQLAlchemy Core."""
        cols = [column(c) for c in self.resource.columns]
        match = column(self.match_column)
        order_cols = [column(c) for c in self.order_by]
        rn = (
            func.row_number()
            .over(partition_by=match, order_by=order_cols)
            .label("__rn")
        )
        total = func.count().over(partition_by=match).label("__total")
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        return (
            select(*cols, rn, total)
            .select_from(tbl)
            .where(match == any_(bindparam("keys", expanding=False)))
        )

    async def run_query(self, unique_keys: List[Any]) -> List[Dict[str, Any]]:
        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(self.build_query(), {"keys": unique_keys})
            return [dict(row) for row in result.mappings().all()]

    def build_connection(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Slice ``rows`` (one parent's window-numbered rows) into a connection dict."""
        start = self.after_offset
        total = rows[0]["__total"] if rows else 0
        page: List[Dict[str, Any]] = []
        for row in rows:
            rn = row["__rn"]
            if rn <= start:
                continue
            if self.first is not None and len(page) >= self.first:
                break
            node = {c: row[c] for c in self.resource.columns}
            page.append({"node": node, "cursor": encode_cursor(rn), "__rn": rn})

        edges = [{"node": e["node"], "cursor": e["cursor"]} for e in page]
        nodes = [e["node"] for e in page]
        end_offset = page[-1]["__rn"] if page else start
        has_next = end_offset < total
        page_info = {
            "hasNextPage": has_next,
            "hasPreviousPage": start > 0,
            "startCursor": page[0]["cursor"] if page else None,
            "endCursor": page[-1]["cursor"] if page else None,
        }
        return {
            "edges": edges,
            "nodes": nodes,
            "totalCount": total,
            "pageInfo": page_info,
        }

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = values[0]
        unique_keys = [k for k in dict.fromkeys(keys) if k is not None]
        empty = self.build_connection([])
        if not unique_keys:
            return [dict(empty) for _ in range(count)]

        async def run():
            rows = await self.run_query(unique_keys)
            by_key: Dict[Any, List[Dict[str, Any]]] = {}
            for row in rows:
                by_key.setdefault(row[self.match_column], []).append(row)
            return [
                self.build_connection(by_key.get(keys[i], [])) for i in range(count)
            ]

        return run()

    @property
    def peer_key(self) -> str:
        return (
            f"pg_connection|{self.resource.qualified_table}|{self.match_column}"
            f"|{self.order_by!r}|{self.first}|{self.after_offset}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (
            self.resource.qualified_table,
            self.match_column,
            self.order_by,
            self.first,
            self.after_offset,
        )

    def get(self, attr: Any) -> Step:
        """Project a connection sub-field (``totalCount`` / ``pageInfo`` / ...)."""
        from ..core_steps import access

        return access(self, (attr,))


def connection(
    resource: PgResource,
    key_step: Step,
    match_column: str,
    order_by: Sequence[str],
    first: Optional[int] = None,
    after: Optional[str] = None,
) -> PgConnectionStep:
    """Plan-helper: a batched Relay connection over a hasMany lookup."""
    return PgConnectionStep(
        resource, key_step, match_column, order_by, first=first, after=after
    )


__all__ = ["PgConnectionStep", "connection", "encode_cursor", "decode_cursor"]
