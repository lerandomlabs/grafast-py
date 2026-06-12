"""Request-input constraints: re-checkable notes a plan records about its request.

Under ``cache_plans`` a finalized plan is shared across requests of the same document.
That is only correct while the plan does not depend on anything request-specific — and
when planning DOES look at a per-request input to decide the plan's shape or field
selection, that look must be written down as a CONSTRAINT: "this plan assumed input X
looked like Y". On every cache hit the stored constraint set is re-validated against
the requesting request (:func:`constraints_match`); a plan whose constraints fail is
simply not a hit for that request — the cache keeps several plan VARIANTS per document
and serves whichever one validates (or plans a fresh variant).

This is the grafast-py port of upstream's ``constraints.ts`` + ``__TrackedValueStep``:
upstream records constraints whenever planning ``eval``s a tracked value, and
``establishOperationPlan`` replays ``matchesConstraints`` per cached candidate. Two
deliberate differences (see AGENTS.md "purposeful distinctions from upstream"):

* matching uses type-pinned Python ``==`` (:func:`values_equal`; upstream ``===``) —
  GraphQL variable coercion pins each variable's Python type per operation, ``==``
  compares coerced dicts/lists sanely where JS reference-equality would always miss,
  and the top-level type check stops bool/int aliasing (``True == 1``);
* the constraint surface is the four cases with consumers here (value / equality /
  exists / keys), not upstream's full eval surface (length/isEmpty have no consumer
  and are out of scope per spec).

Two channels, one rule
----------------------
A plan resolver reads a request input through exactly one of two intents, and the API
carries the intent (the engine cannot infer it from a bare value access):

* **runtime value** — "thread this into the query, I never look at it": a value-less
  placeholder / :class:`ContextToken`. No constraint, no cache split; the per-request
  value rides into the compiled statement at render. This is the common multi-tenant
  path and it MUST stay constraint-free (the hit-rate guarantee).
* **plan-time read** — "show me the value NOW, I am deciding the plan with it":
  ``eval`` / ``eval_is`` / ``eval_has``. Returns the value and records a constraint, so
  the resulting plan is only ever reused for requests where that read would have come
  out the same.

Everything here is pure ``graphql.language`` + stdlib — core stays sqlalchemy-free.
"""

from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from graphql.language import (
    FragmentDefinitionNode,
    FragmentSpreadNode,
    OperationDefinitionNode,
    VariableNode,
)


class _Absent:
    """Sentinel for "no value at this path" (a missing key/attribute).

    Distinct from ``None`` because GraphQL distinguishes an explicit ``null`` from an
    omitted input, and a constraint recorded against an omitted input must only match
    requests where it is omitted again.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "ABSENT"


ABSENT = _Absent()

# distinguishes "gate.get(key)" from "gate.get(key, default)" so the latter can fail loud
# (a value-less token has no default; the dict idiom signals value semantics were expected).
_GET_NO_DEFAULT = object()


class RequestFacts(NamedTuple):
    """The per-request inputs constraints are validated against on a cache hit.

    ``variable_values`` are the graphql-core-COERCED variable values (defaults folded
    in); ``context_value`` is the host's GraphQL context object. Both are read-only
    here — validation is pure and runs on every hit (the hot path).
    """

    variable_values: Mapping[str, Any]
    context_value: Any

    def scope_value(self, scope: str) -> Any:
        if scope == "variables":
            return self.variable_values
        if scope == "context":
            return self.context_value
        raise ValueError(f"unknown constraint scope {scope!r}")


def value_at(obj: Any, path: Sequence[Any]) -> Any:
    """Walk ``path`` through mappings/attributes, returning ``ABSENT`` when it ends.

    Mirrors the render seam's context duality (``where_params``): a Mapping is read by
    key, anything else by attribute — so a constraint recorded against a dict context
    validates the same way the value would be resolved at render. Missing keys yield
    ``ABSENT`` (a mismatch against any recorded value except ``ABSENT`` itself), never
    raise: a hit-time mismatch is a re-plan, not an error.
    """
    value = obj
    for key in path:
        if isinstance(value, Mapping):
            if key not in value:
                return ABSENT
            value = value[key]
        elif value is None or not hasattr(value, str(key)):
            return ABSENT
        else:
            value = getattr(value, str(key))
    return value


def values_equal(a: Any, b: Any) -> bool:
    """Constraint equality: type-pinned ``==``, with the ABSENT sentinel matched by identity.

    The top-level type check approximates upstream's ``===`` where Python ``==`` would
    alias across types (``True == 1``, ``0 == False``) — a context eval of ``1`` must not
    validate a request carrying ``True``. A type mismatch is only ever a re-plan (the safe
    direction); within one operation, GraphQL variable coercion pins each value's type, so
    the variables scope never pays a split for this.
    """
    if a is ABSENT or b is ABSENT:
        return a is b
    if type(a) is not type(b):
        return False
    return a == b


class ValueConstraint(NamedTuple):
    """"The value at ``path`` equals ``value``" — recorded by a full ``eval`` read.

    The most splitting constraint (one plan variant per distinct value); prefer
    :class:`EqualityConstraint` / :class:`ExistsConstraint` where the plan only needed
    a comparison/presence outcome.
    """

    scope: str
    path: Tuple[Any, ...]
    value: Any

    def matches(self, facts: RequestFacts) -> bool:
        return values_equal(value_at(facts.scope_value(self.scope), self.path), self.value)


class EqualityConstraint(NamedTuple):
    """"Comparing the value at ``path`` to ``expected`` yields ``passed``" (eval_is).

    Constrains the comparison OUTCOME, not the value: every request on the same side
    of the comparison shares one plan variant (a million non-admin tenants, one plan).
    """

    scope: str
    path: Tuple[Any, ...]
    expected: Any
    passed: bool

    def matches(self, facts: RequestFacts) -> bool:
        value = value_at(facts.scope_value(self.scope), self.path)
        return values_equal(value, self.expected) == self.passed


class ExistsConstraint(NamedTuple):
    """"A value exists at ``path``" (or doesn't) — recorded by eval_has / membership.

    Present-but-``None`` counts as existing (GraphQL's explicit ``null`` is a value;
    only an OMITTED input is absent).
    """

    scope: str
    path: Tuple[Any, ...]
    exists: bool

    def matches(self, facts: RequestFacts) -> bool:
        return (value_at(facts.scope_value(self.scope), self.path) is not ABSENT) == self.exists


class KeysConstraint(NamedTuple):
    """"The mapping at ``path`` has exactly these keys" — recorded by enumerating reads.

    Iterating ``info.variable_values`` (or taking its length) reveals WHICH variables the
    request carries — a per-request fact a plan may have shaped itself around — so the
    enumeration pins the sorted key set (the analogue of upstream ``evalKeys``).
    """

    scope: str
    path: Tuple[Any, ...]
    keys: Tuple[str, ...]

    def matches(self, facts: RequestFacts) -> bool:
        value = value_at(facts.scope_value(self.scope), self.path)
        if not isinstance(value, Mapping):
            return False
        return tuple(sorted(value)) == self.keys


def constraints_match(constraints: Iterable[Any], facts: RequestFacts) -> bool:
    """Whether EVERY recorded constraint still holds for this request (AND semantics).

    The on-hit replay (upstream ``matchesConstraints``): pure, no I/O, runs on every
    hit. Constraints are duck-typed on ``matches(facts)`` so the pg
    ``CustomizerConstraint`` (which re-invokes its customizer against the pg request
    context and ignores ``facts``) rides the same list without core importing pg.
    """
    return all(constraint.matches(facts) for constraint in constraints)


# the directives whose arguments graphql-core (or our planner) resolves against variable
# VALUES before/during planning, freezing the outcome into the plan: @skip/@include decide
# the collected field selection (collect_fields), @defer decides the deferred partitions
# (build_execution_plan), and @stream's initialCount/label are read into FieldPlan.stream
# at plan time (get_stream_usage) — unlike upstream, where @stream args are runtime steps.
_PLAN_TIME_DIRECTIVES = frozenset({"skip", "include", "defer", "stream"})


def directive_variable_constraints(
    operation: OperationDefinitionNode,
    fragments: Optional[Mapping[str, FragmentDefinitionNode]],
    variable_values: Mapping[str, Any],
) -> List[ValueConstraint]:
    """Constraints for every ``$variable`` used as a plan-time directive argument.

    This closes the R4 hole: ``@skip(if: $hide)`` (and @include / fragment ``if:`` /
    @defer / @stream) is resolved against variable VALUES inside graphql-core's
    ``collect_fields`` BEFORE planning, so the resolved field selection is frozen into
    the plan. That read happens inside foreign code and cannot be routed through a
    tracked accessor — but the document names exactly which variables it will read, so
    we record the constraints upstream's ``evalDirectiveArg`` would have recorded
    (one ``value`` constraint per directive variable; a literal arg records nothing)
    and let graphql-core's collection run unchanged.

    Walks the operation's selections plus every reachable named fragment once (visited
    set: fragments may recurse); duplicate uses of one variable fold into one
    constraint.
    """
    names: set = set()

    def scan_selection_set(selection_set, visited_fragments: set) -> None:
        if selection_set is None:
            return
        for selection in selection_set.selections:
            for directive in selection.directives or ():
                if directive.name.value not in _PLAN_TIME_DIRECTIVES:
                    continue
                for arg in directive.arguments or ():
                    if isinstance(arg.value, VariableNode):
                        names.add(arg.value.name.value)
            if isinstance(selection, FragmentSpreadNode):
                fragment_name = selection.name.value
                if fragments and fragment_name not in visited_fragments:
                    visited_fragments = visited_fragments | {fragment_name}
                    fragment = fragments.get(fragment_name)
                    if fragment is not None:
                        scan_selection_set(fragment.selection_set, visited_fragments)
            else:
                scan_selection_set(getattr(selection, "selection_set", None), visited_fragments)

    scan_selection_set(operation.selection_set, set())
    return [
        ValueConstraint("variables", (name,), variable_values.get(name, ABSENT))
        for name in sorted(names)
    ]


class ContextToken:
    """A value-less handle to one request-context entry — the runtime-value channel.

    ``info.context["tenant_id"]`` hands a plan resolver THIS, never the value: the
    token says *which* context entry flows into the query (``source`` =
    ``"ctx:tenant_id"``, the same source-tag namespace the pg render seam resolves per
    request), while the value itself stays out of the plan entirely. One shared plan
    serves every request; no constraint is recorded.

    Branching on a token is meaningless (it is the same object shape for every
    request), so ``bool()`` fails loud instead of silently taking one branch — the two
    legitimate moves are named in the error. In a pg predicate the token coerces to a
    value-less placeholder bind via ``__clause_element__`` (the SQLAlchemy coercion
    protocol), so ``.where(col == info.context["tenant_id"])`` builds the placeholder
    directly.
    """

    __slots__ = ("source",)

    def __init__(self, source: str) -> None:
        self.source = source

    def __bool__(self) -> bool:
        raise TypeError(
            f"a plan-time context read is a value-less token ({self.source!r}); to branch "
            "the plan on this value use info.context.eval(...) (records a cache "
            "constraint), or thread the token into the query as a placeholder"
        )

    def __eq__(self, other: Any) -> Any:
        # key off the SOURCE tag only (mirror the pagination Placeholder): two same-source
        # tokens are interchangeable handles, and a token never equals a bare value — a
        # comparison against the runtime value is exactly what a token withholds.
        if not isinstance(other, ContextToken):
            return NotImplemented
        return self.source == other.source

    def __hash__(self) -> int:
        return hash(self.source)

    def __repr__(self) -> str:
        return f"ContextToken({self.source!r})"

    def __clause_element__(self):
        # SQLAlchemy's coercion hook: `column == token` builds a value-less placeholder
        # bind tagged with this token's source. Only ever invoked by SQLAlchemy itself,
        # so the pg import cannot fire on a core-only install.
        from .pg.placeholders import pg_placeholder

        return pg_placeholder(self.source)


class ContextGate:
    """The plan-resolver view of the request context: withhold by default, eval by name.

    Plan resolvers receive this as ``info.context`` (in every mode — the API does not
    change shape with caching). A bare read (``[...]`` / ``.get``) yields a
    :class:`ContextToken` — the runtime-value channel, constraint-free. The ``eval``
    family is the explicit "look at the value NOW to decide the plan" read: it returns
    the real value and records a constraint into the plan's request-constraint sink, so
    the plan only ever serves requests where the read comes out the same.

    Upstream's public ``context()`` exposes ONLY the token channel; the eval family is
    engine-internal there. Offering it publicly is a deliberate grafast-py extension
    (same machinery, upstream-documented as usable for plan branching).

    Classic (runtime) resolvers are untouched — they read the real context at execute
    time through graphql-core's own info. Note the token's value resolves at the pg
    render seam from ``current_pg_request().context`` (the pg request context), so a
    host must hand the same object to ``pg_request_context(context=...)`` and
    ``graphql(context_value=...)`` — the same contract ``ContextSources`` placeholders
    already carry.
    """

    __slots__ = ("_context_value", "_sink")

    def __init__(self, context_value: Any, sink: List[Any]) -> None:
        self._context_value = context_value
        self._sink = sink

    def __getitem__(self, key: str) -> ContextToken:
        if not isinstance(key, str):
            # without this, `iter(gate)` falls back to the legacy sequence protocol
            # (__getitem__(0), 1, 2, …) and spins forever yielding tokens.
            raise TypeError(
                "the plan-time context gate is not iterable/indexable; read one entry "
                "(info.context['key']) or use the eval family to branch the plan"
            )
        return ContextToken(f"ctx:{key}")

    def get(self, key: str, default: Any = _GET_NO_DEFAULT) -> ContextToken:
        if default is not _GET_NO_DEFAULT:
            raise TypeError(
                "a plan-time context token has no default — the value (and its absence) "
                "resolves at execute; to branch on presence or value at plan time use "
                f"info.context.eval_has({key!r}) / eval({key!r}) (records a cache constraint)"
            )
        return ContextToken(f"ctx:{key}")

    def __contains__(self, key: object) -> bool:
        raise TypeError(
            "a plan-time membership check on the request context is a planning decision; "
            f"use info.context.eval_has({key!r}) (records a cache constraint)"
        )

    def __iter__(self):
        raise TypeError(
            "the plan-time context gate is not iterable; read one entry "
            "(info.context['key']) or use the eval family to branch the plan"
        )

    def _read(self, key: str) -> Any:
        # fail loud on a missing key (mirrors where_params): an eval of a key the host
        # never provided is a wiring bug, not an ABSENT to silently constrain on.
        value = self._context_value
        if isinstance(value, Mapping):
            return value[key]
        return getattr(value, key)

    def eval(self, key: str) -> Any:
        """The value of ``key`` NOW, recording that this plan assumed exactly it.

        The most splitting read (one plan variant per distinct value) — prefer
        :meth:`eval_is` / :meth:`eval_has` when only an outcome is needed. The evaluated
        value is stored inside the cached plan's constraint set (and compared with ``==``
        on every hit), so eval scalars — evaluating a live object (a session, a model)
        pins it in the cache and splits on its equality semantics.
        """
        value = self._read(key)
        self._sink.append(ValueConstraint("context", (key,), value))
        return value

    def eval_is(self, key: str, expected: Any) -> bool:
        """Whether ``key`` equals ``expected``, constraining only the OUTCOME.

        Also pins the key's PRESENCE: reading it here fails loud when missing (a wiring
        bug), so a warm cache must not silently serve a variant to a request that would
        have raised on a fresh build — the exists constraint forces such a request to
        re-plan (and so to raise the same loud error).
        """
        passed = values_equal(self._read(key), expected)
        self._sink.append(ExistsConstraint("context", (key,), True))
        self._sink.append(EqualityConstraint("context", (key,), expected, passed))
        return passed

    def eval_has(self, key: str) -> bool:
        """Whether ``key`` is present, constraining only the presence."""
        value = self._context_value
        if isinstance(value, Mapping):
            exists = key in value
        else:
            exists = hasattr(value, key)
        self._sink.append(ExistsConstraint("context", (key,), exists))
        return exists

    def __repr__(self) -> str:
        return f"ContextGate({type(self._context_value).__name__})"


class TrackedVariables(Mapping):
    """``info.variable_values`` for plan resolvers: every read is an eval.

    A read of a variable's value at plan time is a planning decision by definition
    (there is no runtime channel through this surface — that is what ``args.source()``
    placeholders are for), so each ``[...]`` / ``.get`` / ``in`` records the matching
    constraint, and ENUMERATING the mapping (``iter`` / ``len`` / ``keys`` /
    ``dict(...)``) pins the whole key set (which variables this request carries is
    itself a per-request fact).
    """

    __slots__ = ("_values", "_sink")

    def __init__(self, variable_values: Mapping[str, Any], sink: List[Any]) -> None:
        self._values = variable_values
        self._sink = sink

    def __getitem__(self, name: str) -> Any:
        value = self._values[name]
        self._sink.append(ValueConstraint("variables", (name,), value))
        return value

    def get(self, name: str, default: Any = None) -> Any:
        value = self._values.get(name, ABSENT)
        self._sink.append(ValueConstraint("variables", (name,), value))
        return default if value is ABSENT else value

    def __contains__(self, name: object) -> bool:
        exists = name in self._values
        self._sink.append(ExistsConstraint("variables", (str(name),), exists))
        return exists

    def _note_keys(self) -> None:
        self._sink.append(
            KeysConstraint("variables", (), tuple(sorted(self._values)))
        )

    def __iter__(self):
        self._note_keys()
        return iter(self._values)

    def __len__(self) -> int:
        self._note_keys()
        return len(self._values)

    def __repr__(self) -> str:
        return f"TrackedVariables({dict(self._values)!r})"


__all__ = [
    "ABSENT",
    "RequestFacts",
    "ValueConstraint",
    "EqualityConstraint",
    "ExistsConstraint",
    "KeysConstraint",
    "values_equal",
    "constraints_match",
    "directive_variable_constraints",
    "ContextToken",
    "ContextGate",
    "TrackedVariables",
    "value_at",
]
