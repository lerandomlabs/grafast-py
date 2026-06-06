"""Single-row CRUD mutation steps: pgInsertSingle / pgUpdateSingle / pgDeleteSingle.

The WRITE counterpart to the batched read steps in :mod:`grafast_py.pg.steps`, riding the
EXISTING serial mutation seam (mutation root fields run SERIALLY, so each write is observed
by the next) rather than the O(depth) batched read path — they never touch the read path,
so its batching profile is unchanged. Each builds a Core ``insert`` / ``update`` /
``delete`` that binds EVERY value as a PARAM (never string-interpolated, so SQL
metacharacters in an input are stored verbatim, never executed) and ``.returning`` the
resource columns, codec-decoded via ``resource.decode_row`` so returned fields decode like
a read. The statement runs with ``commit=True`` (the no-commit read path rolls back on
close, so the write would not persist); a PK-keyed update/delete matching no row returns
``None``.

A ``values`` entry is a plan-time literal or a step DEPENDENCY (its computed column supplies
the value at execute time); the attribute's codec ``to_pg`` encodes it before binding.
Omitted columns fall back to the DB default; a NOT NULL no-default column with no value
fails LOUD with a clear message rather than a raw DB error.

Deferred (NOT in this v1 scope): bulk / multi-row mutations, arbitrary-unique-key ``getBy``
(update/delete are PK-keyed only), and SAVEPOINT nesting for nested mutations.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from sqlalchemy import bindparam, column, delete, insert, table, update

from ..config import log
from ..step_model import Step
from .executor import current_pg_request
from .resource import PgResource

# A value source for an insert/update column: either a plan-time literal known from the
# mutation input args, or a step whose per-entry output column supplies it at execute time.
ValueSource = Union[Any, Step]


class PgWriteSingleStep(Step):
    """Shared base for the three single-row write steps.

    Holds the value sources (``values``) for the columns being written and tracks which
    are step DEPENDENCIES (resolved from the bucket column at execute time) vs plan-time
    literals (bound directly). Subclasses build their specific Core statement in
    :meth:`build_statement` and run it per bucket entry via :meth:`execute`.
    """

    is_sync_and_safe = False
    # a mutation is a distinct side effect: two same-class writes over the same resource
    # in one operation must BOTH run, so they are never collapsed by the dedup pass.
    dedupable = False

    def __init__(self, resource: PgResource, values: Mapping[str, ValueSource]) -> None:
        super().__init__()
        self.resource = resource
        # column name -> dependency index, for values sourced from a step. Literals are
        # kept inline in ``literal_values``; both are merged per entry at execute time.
        self.dep_index_for_column: Dict[str, int] = {}
        self.literal_values: Dict[str, Any] = {}
        for name, source in values.items():
            if isinstance(source, Step):
                self.dep_index_for_column[name] = self.add_dependency(source)
            else:
                self.literal_values[name] = source

    def core_table(self):
        """The SQLAlchemy Core ``table`` over the resource's stored columns."""
        return table(
            self.resource.table,
            *[column(c) for c in self.resource.columns],
            schema=self.resource.schema,
        )

    def returning_columns(self, tbl) -> List[Any]:
        """The resource's stored columns to RETURNING (the projected result row)."""
        return [tbl.c[name] for name in self.resource.columns]

    def values_for_entry(self, values: List[List[Any]], i: int) -> Dict[str, Any]:
        """Merge the literal + dependency-sourced column values for bucket entry ``i``.

        Each value is encoded with the attribute's codec ``to_pg`` before binding (a
        pass-through no-op when the codec or value is ``None``). A NOT NULL column with no
        DB default and no supplied value fails LOUD here, before any statement is built.
        """
        merged: Dict[str, Any] = dict(self.literal_values)
        for name, dep_index in self.dep_index_for_column.items():
            merged[name] = values[dep_index][i]
        encoded: Dict[str, Any] = {}
        for name, value in merged.items():
            encoded[name] = self.encode_value(name, value)
        self.assert_required_columns_present(encoded)
        return encoded

    def encode_value(self, name: str, value: Any) -> Any:
        """Apply the attribute's codec ``to_pg`` (pass-through when None/absent)."""
        attribute = self.resource.attributes.get(name)
        if attribute is not None and attribute.codec is not None:
            to_pg = attribute.codec.to_pg
            if to_pg is not None and value is not None:
                return to_pg(value)
        return value

    def assert_required_columns_present(self, provided: Mapping[str, Any]) -> None:
        """Fail loud if a NOT NULL no-default column is omitted (no silent DB error).

        A column that is NOT NULL, has no DB default, and is not the primary key (which a
        DB sequence/explicit value supplies) must be provided by the mutation; otherwise
        the INSERT would raise a raw not-null violation deep in the driver. We surface a
        clear, column-named error at plan/execute time instead.
        """
        for name, attribute in self.resource.attributes.items():
            if attribute.is_computed:
                continue
            if name == self.resource.primary_key:
                continue
            if attribute.not_null and not attribute.has_default and name not in provided:
                raise ValueError(
                    f"{self.resource.name}.{name} is NOT NULL with no default; "
                    f"a value must be supplied to insert a {self.resource.name} row"
                )

    def decode_returned(self, rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Codec-decode the single RETURNING row (or ``None`` when none matched)."""
        if not rows:
            return None
        return self.resource.decode_row(rows[0])

    def execute(self, count: int, values: List[List[Any]]) -> List[Any]:
        async def run():
            request = current_pg_request()
            out: List[Any] = []
            for i in range(count):
                stmt, params = self.build_statement(values, i)
                rows = await request.executor.run(
                    stmt, params, settings=request.settings, commit=True
                )
                out.append(self.decode_returned(rows))
            log.debug("pg mutate", resource=self.resource.name, op=self.op, rows=count)
            return out

        return run()

    # subclasses provide these
    op: str = "write"

    def build_statement(
        self, values: List[List[Any]], i: int
    ) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError


class PgInsertSingleStep(PgWriteSingleStep):
    """INSERT one row, RETURNING the resource columns as a row dict.

    ``values`` maps column -> value source (literal or step). Columns not provided are
    omitted so DB defaults apply; a NOT NULL no-default column with no value fails loud.
    """

    op = "insert"

    def build_statement(
        self, values: List[List[Any]], i: int
    ) -> Tuple[Any, Dict[str, Any]]:
        tbl = self.core_table()
        entry = self.values_for_entry(values, i)
        # bind EVERY value as a param (col__v) — never string-interpolated, so an input
        # carrying SQL metacharacters is stored verbatim and never executed.
        bound = {name: bindparam(f"{name}__v") for name in entry}
        params = {f"{name}__v": value for name, value in entry.items()}
        stmt = insert(tbl).values(**bound).returning(*self.returning_columns(tbl))
        return stmt, params

    @property
    def peer_key(self) -> str:
        return f"pg_insert_single|{self.resource.qualified_table}"


class PgUpdateSingleStep(PgWriteSingleStep):
    """UPDATE the PK-keyed row, RETURNING the updated row dict (``None`` if no match).

    The primary key is bound as a param in the WHERE; ``values`` are the columns to set.
    A non-matching PK yields an empty RETURNING -> ``None``.
    """

    op = "update"

    def __init__(
        self,
        resource: PgResource,
        pk_value: ValueSource,
        values: Mapping[str, ValueSource],
    ) -> None:
        super().__init__(resource, values)
        self.pk_dep_index: Optional[int] = None
        if isinstance(pk_value, Step):
            self.pk_dep_index = self.add_dependency(pk_value)
            self.pk_literal: Any = None
        else:
            self.pk_literal = pk_value

    def pk_for_entry(self, values: List[List[Any]], i: int) -> Any:
        """The primary-key value for bucket entry ``i`` (literal or dependency column)."""
        if self.pk_dep_index is not None:
            return values[self.pk_dep_index][i]
        return self.pk_literal

    def build_statement(
        self, values: List[List[Any]], i: int
    ) -> Tuple[Any, Dict[str, Any]]:
        tbl = self.core_table()
        entry = self.values_for_entry(values, i)
        if not entry:
            # update(tbl).values() would render `SET ` (empty) and raise a raw DB syntax
            # error; an update with no columns is a no-op the caller must not send.
            raise ValueError(
                f"{self.resource.name}: pg_update_single requires at least one column "
                "to set"
            )
        bound = {name: bindparam(f"{name}__v") for name in entry}
        params = {f"{name}__v": value for name, value in entry.items()}
        # the PK is bound as a param (:pk), never interpolated.
        params["pk"] = self.pk_for_entry(values, i)
        stmt = (
            update(tbl)
            .where(tbl.c[self.resource.primary_key] == bindparam("pk"))
            .values(**bound)
            .returning(*self.returning_columns(tbl))
        )
        return stmt, params

    def assert_required_columns_present(self, provided: Mapping[str, Any]) -> None:
        """An UPDATE sets only the supplied columns; untouched ones keep their value."""
        # No required-column check for an update: omitting a column leaves it unchanged.

    @property
    def peer_key(self) -> str:
        return f"pg_update_single|{self.resource.qualified_table}"


class PgDeleteSingleStep(PgWriteSingleStep):
    """DELETE the PK-keyed row, RETURNING the deleted row dict (``None`` if no match).

    The primary key is bound as a param in the WHERE. A non-matching PK yields an empty
    RETURNING -> ``None``.
    """

    op = "delete"

    def __init__(self, resource: PgResource, pk_value: ValueSource) -> None:
        super().__init__(resource, {})
        self.pk_dep_index: Optional[int] = None
        if isinstance(pk_value, Step):
            self.pk_dep_index = self.add_dependency(pk_value)
            self.pk_literal: Any = None
        else:
            self.pk_literal = pk_value

    def pk_for_entry(self, values: List[List[Any]], i: int) -> Any:
        """The primary-key value for bucket entry ``i`` (literal or dependency column)."""
        if self.pk_dep_index is not None:
            return values[self.pk_dep_index][i]
        return self.pk_literal

    def build_statement(
        self, values: List[List[Any]], i: int
    ) -> Tuple[Any, Dict[str, Any]]:
        tbl = self.core_table()
        params = {"pk": self.pk_for_entry(values, i)}
        stmt = (
            delete(tbl)
            .where(tbl.c[self.resource.primary_key] == bindparam("pk"))
            .returning(*self.returning_columns(tbl))
        )
        return stmt, params

    @property
    def peer_key(self) -> str:
        return f"pg_delete_single|{self.resource.qualified_table}"


# -------------------------------------------------------------- plan-helper API
# Free-function constructors mirroring the read helpers (pg_select / pg_select_single),
# so a mutation plan resolver reads naturally. Each merely CONSTRUCTS a step (no SQL at
# plan time); the engine runs it on the serial mutation seam.


def pg_insert_single(
    resource: PgResource, values: Mapping[str, ValueSource]
) -> PgInsertSingleStep:
    return PgInsertSingleStep(resource, values)


def pg_update_single(
    resource: PgResource,
    pk_value: ValueSource,
    values: Mapping[str, ValueSource],
) -> PgUpdateSingleStep:
    return PgUpdateSingleStep(resource, pk_value, values)


def pg_delete_single(resource: PgResource, pk_value: ValueSource) -> PgDeleteSingleStep:
    return PgDeleteSingleStep(resource, pk_value)


__all__ = [
    "PgInsertSingleStep",
    "PgUpdateSingleStep",
    "PgDeleteSingleStep",
    "pg_insert_single",
    "pg_update_single",
    "pg_delete_single",
]
