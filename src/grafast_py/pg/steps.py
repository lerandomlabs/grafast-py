"""Batch pg steps: one parameterised SQL per bucket, scattered back to parents.

:class:`PgSelectStep` (hasMany / collections) and :class:`PgSelectSingleStep`
(hasOne / single row) are ordinary :class:`grafast_py.step_model.Step` subclasses, so
they ride the SAME bucket executor, dedup and ``EachStep`` flattening as the core
steps — no second executor. The Grafast batching property is realised in
``execute``: it receives EVERY lookup key across the bucket at once (its dep-0 key
column), folds them into a single ``WHERE match_column = ANY($1)`` statement run ONCE
on the async engine, then scatters the rows to the parents whose key matched.

Hence a depth-D nested query issues ~D SQL statements total (one per resource-layer):
``authors`` (1) -> for ALL authors' posts ``WHERE author_id = ANY(...)`` (1) -> for
ALL those posts' comments ``WHERE post_id = ANY(...)`` (1). The flattening of list
items into one bucket (Phase A ``EachStep`` / list completion) is what lets the inner
layer see every key at once.

Rows come back as plain dicts keyed by column name so the existing ``AccessStep``
projects leaf columns with no special-casing.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import any_, bindparam, column, select, table

from ..config import log
from ..step_model import Step
from .engine import get_engine
from .resource import PgResource


def _coalesce_keys(keys: List[Any], count: int) -> Tuple[List[Any], List[Optional[int]]]:
    """Collapse a bucket's key column to unique non-null keys + per-entry slot.

    Returns ``(unique_keys, entry_slot)`` where ``entry_slot[i]`` is the index into
    ``unique_keys`` for entry ``i`` (or ``None`` when that entry's key is ``None`` and
    therefore looks up nothing — missing key -> null/empty).
    """
    unique_keys: List[Any] = []
    slot_for_key: Dict[Any, int] = {}
    entry_slot: List[Optional[int]] = [None] * count
    for i in range(count):
        key = keys[i]
        if key is None:
            continue
        slot = slot_for_key.get(key)
        if slot is None:
            slot = len(unique_keys)
            slot_for_key[key] = slot
            unique_keys.append(key)
        entry_slot[i] = slot
    return unique_keys, entry_slot


class PgSelectStep(Step):
    """Batched ``SELECT ... WHERE match_column = ANY($1)`` returning a LIST per entry.

    One dependency: the per-entry key step (dep 0), e.g. an author id for an
    ``Author.posts`` hasMany. ``execute`` gathers every key in the bucket, runs ONE
    statement, groups rows by their ``match_column`` value, and scatters each entry's
    list of matching rows (missing key -> empty list).
    """

    is_sync_and_safe = False

    def __init__(
        self,
        resource: PgResource,
        key_step: Step,
        match_column: str,
        order_by: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.resource = resource
        self.match_column = match_column
        self.order_by: Tuple[str, ...] = tuple(order_by or ())
        self.add_dependency(key_step)

    def build_query(self):
        """Build the parameterised ``= ANY($1)`` SELECT via SQLAlchemy Core."""
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        match = column(self.match_column)
        stmt = select(tbl).where(match == any_(bindparam("keys", expanding=False)))
        if self.order_by:
            stmt = stmt.order_by(*[column(c) for c in self.order_by])
        return stmt

    async def run_query(self, unique_keys: List[Any]) -> List[Dict[str, Any]]:
        """Run the batched statement once and return rows as column-keyed dicts."""
        engine = get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(self.build_query(), {"keys": unique_keys})
            rows = [dict(row) for row in result.mappings().all()]
        log.debug(
            "pg batch select",
            resource=self.resource.name,
            keys=len(unique_keys),
            rows=len(rows),
        )
        return rows

    def group_rows(self, rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
        """Group rows by their ``match_column`` value (preserving query order)."""
        grouped: Dict[Any, List[Dict[str, Any]]] = {}
        col = self.match_column
        for row in rows:
            grouped.setdefault(row[col], []).append(row)
        return grouped

    def scatter(
        self,
        grouped: Dict[Any, List[Dict[str, Any]]],
        keys: List[Any],
        count: int,
    ) -> List[Any]:
        """Per-entry list of matching rows (missing key -> empty list)."""
        return [grouped.get(keys[i], []) for i in range(count)]

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = values[0]
        unique_keys, _slot = _coalesce_keys(keys, count)
        if not unique_keys:
            return [[] for _ in range(count)]

        async def run():
            rows = await self.run_query(unique_keys)
            return self.scatter(self.group_rows(rows), keys, count)

        return run()

    @property
    def peer_key(self) -> str:
        return (
            f"pg_select|{self.resource.qualified_table}|{self.match_column}"
            f"|{self.order_by!r}"
        )

    def dedup_params(self) -> Tuple[Any, ...]:
        return (self.resource.qualified_table, self.match_column, self.order_by)


class PgSelectAllStep(Step):
    """Batched ``SELECT ... [ORDER BY]`` (no WHERE) returning ALL rows per entry.

    The root-collection step (``Query.authors`` / ``Query.posts``): one dependency,
    the bucket's parent step (the operation root), used only to size the bucket — the
    same row list is returned for every entry. A root list has a single bucket entry
    (the root value), so this is ONE statement; the relation layers chain off the
    returned row steps.
    """

    is_sync_and_safe = False

    def __init__(
        self, resource: PgResource, order_by: Optional[Sequence[str]] = None
    ) -> None:
        super().__init__()
        self.resource = resource
        self.order_by: Tuple[str, ...] = tuple(order_by or (resource.primary_key,))

    def for_parent(self, parent_step: Step) -> "PgSelectAllStep":
        """Wire the bucket-sizing parent dependency and return self."""
        self.add_dependency(parent_step)
        return self

    def build_query(self):
        tbl = table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )
        stmt = select(tbl)
        if self.order_by:
            stmt = stmt.order_by(*[column(c) for c in self.order_by])
        return stmt

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        async def run():
            engine = get_engine()
            async with engine.connect() as conn:
                result = await conn.execute(self.build_query())
                rows = [dict(row) for row in result.mappings().all()]
            return [list(rows) for _ in range(count)]

        return run()

    @property
    def peer_key(self) -> str:
        return f"pg_select_all|{self.resource.qualified_table}|{self.order_by!r}"

    def dedup_params(self) -> Tuple[Any, ...]:
        return (self.resource.qualified_table, self.order_by)


class PgSelectSingleStep(PgSelectStep):
    """Batched ``SELECT ... WHERE match_column = ANY($1)`` returning ONE row per entry.

    Same single-statement batching as :class:`PgSelectStep`; each entry's result is
    its single matching row (or ``None`` for a missing key) rather than a list. Used
    for hasOne relations (``Post.author``) and ``resource.get(id)``.
    """

    def scatter(
        self,
        grouped: Dict[Any, List[Dict[str, Any]]],
        keys: List[Any],
        count: int,
    ) -> List[Any]:
        out: List[Any] = []
        for i in range(count):
            rows = grouped.get(keys[i])
            out.append(rows[0] if rows else None)
        return out

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        keys = values[0]
        unique_keys, _slot = _coalesce_keys(keys, count)
        if not unique_keys:
            return [None] * count

        async def run():
            rows = await self.run_query(unique_keys)
            return self.scatter(self.group_rows(rows), keys, count)

        return run()

    @property
    def peer_key(self) -> str:
        return (
            f"pg_select_single|{self.resource.qualified_table}|{self.match_column}"
            f"|{self.order_by!r}"
        )

    def get(self, attr: Any) -> Step:
        """Lazily access a column of the loaded row (an :class:`AccessStep`)."""
        from ..core_steps import access

        return access(self, (attr,))


# -------------------------------------------------------------- plan-helper API
# Free-function constructors mirroring Grafast's pgSelect / pgSelectSingle, so a plan
# resolver reads naturally. Each merely CONSTRUCTS a step (no SQL at plan time).


def pg_select(
    resource: PgResource,
    key_step: Step,
    match_column: str,
    order_by: Optional[Sequence[str]] = None,
) -> PgSelectStep:
    return PgSelectStep(resource, key_step, match_column, order_by=order_by)


def pg_select_single(
    resource: PgResource, key_step: Step, match_column: Optional[str] = None
) -> PgSelectSingleStep:
    return PgSelectSingleStep(
        resource, key_step, match_column or resource.primary_key
    )


__all__ = [
    "PgSelectStep",
    "PgSelectSingleStep",
    "PgSelectAllStep",
    "pg_select",
    "pg_select_single",
]
