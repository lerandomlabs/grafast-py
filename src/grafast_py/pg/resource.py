"""PgResource: a table/source abstraction with columns, primary key and relations.

Mirrors Grafast's ``@dataplan/pg`` resource concept (minus codecs, which we fold into
plain Python column lists). A :class:`PgResource` names a table, the columns to
SELECT, its primary key, and the relations (hasOne / hasMany) that link it to other
resources by FK columns. Plan resolvers use the resource's factory methods
(:meth:`PgResource.get_single`, :meth:`PgResource.find`) and relation helpers to build
the batched pg steps in :mod:`grafast_py.pg.steps`.

Resources register in a :class:`PgRegistry` so relations resolve their target by
name regardless of declaration order.
"""

from typing import Dict, List, Optional, Sequence

from ..step_model import Step


class PgRelation:
    """A FK link from one resource to another.

    ``local_column`` is the column on the *owning* resource whose value identifies the
    related rows; ``remote_column`` is the matched column on ``target``. For a hasOne
    (``Post.author``) the local column is the FK (``author_id``) and the remote is the
    target's key (``id``). For a hasMany (``Author.posts``) the local column is the
    owner's key (``id``) and the remote is the FK on the target (``author_id``).
    """

    def __init__(
        self,
        name: str,
        target: "PgResource",
        local_column: str,
        remote_column: str,
        kind: str,
    ) -> None:
        self.name = name
        self.target = target
        self.local_column = local_column
        self.remote_column = remote_column
        self.kind = kind  # "has_one" | "has_many"


class PgResource:
    """One Postgres table/source: name, columns, primary key, relations."""

    def __init__(
        self,
        name: str,
        schema: str,
        table: str,
        columns: Sequence[str],
        primary_key: str = "id",
        registry: Optional["PgRegistry"] = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self.table = table
        self.columns: List[str] = list(columns)
        self.primary_key = primary_key
        self.relations: Dict[str, PgRelation] = {}
        if registry is not None:
            registry.add(self)

    @property
    def qualified_table(self) -> str:
        """``schema.table`` for use in SQL."""
        return f"{self.schema}.{self.table}"

    def has_one(
        self, name: str, target: "PgResource", local_column: str, remote_column: str
    ) -> PgRelation:
        """Register a hasOne relation (one related row, matched remote == local)."""
        relation = PgRelation(name, target, local_column, remote_column, "has_one")
        self.relations[name] = relation
        return relation

    def has_many(
        self, name: str, target: "PgResource", local_column: str, remote_column: str
    ) -> PgRelation:
        """Register a hasMany relation (a list of related rows)."""
        relation = PgRelation(name, target, local_column, remote_column, "has_many")
        self.relations[name] = relation
        return relation

    def get_relation(self, name: str) -> PgRelation:
        """Return a registered relation by name (fail loud on a typo)."""
        relation = self.relations.get(name)
        if relation is None:
            raise KeyError(f"resource {self.name!r} has no relation {name!r}")
        return relation

    # ------------------------------------------------------------- step factories
    # The plan-resolver-facing surface. Each builds a batched pg step keyed on a
    # per-entry key step; the SQL emission lives in grafast_py.pg.steps.

    def get_single(self, key_step: Step, match_column: Optional[str] = None) -> Step:
        """A :class:`PgSelectSingleStep`: one row where ``match_column`` == key.

        ``match_column`` defaults to the primary key (``resource.get(id)``).
        """
        from .steps import PgSelectSingleStep

        return PgSelectSingleStep(self, key_step, match_column or self.primary_key)

    def find(
        self,
        key_step: Step,
        match_column: str,
        order_by: Optional[Sequence[str]] = None,
    ) -> Step:
        """A :class:`PgSelectStep`: all rows where ``match_column`` == key.

        Used for hasMany relations (``match_column`` is the FK on this resource).
        """
        from .steps import PgSelectStep

        return PgSelectStep(self, key_step, match_column, order_by=order_by)

    def related_single(self, parent_row_step: Step, relation_name: str) -> Step:
        """Plan a hasOne relation off ``parent_row_step`` (the parent row step)."""
        from ..core_steps import access

        relation = self.get_relation(relation_name)
        key = access(parent_row_step, (relation.local_column,))
        return relation.target.get_single(key, relation.remote_column)

    def related_many(
        self,
        parent_row_step: Step,
        relation_name: str,
        order_by: Optional[Sequence[str]] = None,
    ) -> Step:
        """Plan a hasMany relation off ``parent_row_step`` (the parent row step)."""
        from ..core_steps import access

        relation = self.get_relation(relation_name)
        key = access(parent_row_step, (relation.local_column,))
        default_order = order_by or [relation.target.primary_key]
        return relation.target.find(key, relation.remote_column, order_by=default_order)


class PgRegistry:
    """A name -> :class:`PgResource` map so relations resolve their targets."""

    def __init__(self) -> None:
        self.resources: Dict[str, PgResource] = {}

    def add(self, resource: PgResource) -> None:
        self.resources[resource.name] = resource

    def __getitem__(self, name: str) -> PgResource:
        return self.resources[name]


__all__ = ["PgResource", "PgRelation", "PgRegistry"]
