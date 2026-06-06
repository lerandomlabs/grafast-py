"""A registry of :class:`~grafast_py.pg.resource.PgCodec` by Postgres type name.

A codec is the READ/WRITE bridge between a fetched Postgres value and the Python value
a leaf resolver presents (``to_py`` on read, ``to_pg`` on write) PLUS the SQL type its
column declares (``sql_type``), which the keyset cursor path casts a text-origin cursor
value back to. This module names the codec for a given Postgres type — the scalar types
asyncpg already hands back natively (``int4``/``text``/``bool`` need no transform), the
non-native scalars that still need a cast for the keyset path (``timestamptz`` /
``numeric``), and the three RECURSIVE container kinds:

- ARRAY (``int4[]`` / ``text[]`` / ...) — asyncpg returns a Python ``list``; the array
  codec maps the ELEMENT codec's ``to_py`` over every (non-NULL) element, and its
  ``sql_type`` is ``ARRAY(element_sql_type)`` so a cursor over an array column casts to
  the array type;
- RANGE (``int4range`` / ``tstzrange`` / ...) — asyncpg returns an ``asyncpg.Range``;
  the range codec decodes it into a plain ``{lower, upper, lower_inc, upper_inc, empty}``
  dict (applying the BOUND codec's ``to_py`` to each finite bound), so a host needn't
  depend on the asyncpg type;
- COMPOSITE (a row type) — asyncpg returns a positional ``tuple``; the composite codec
  zips it against the named FIELD codecs into a ``{field: decoded}`` dict, recursing into
  each field's codec (so a composite OF an array OF a composite decodes all the way down).

ENUM is a text-backed scalar: its codec is the identity ``to_py`` plus an optional label
SET it validates membership against, so an out-of-range label fails loud rather than
silently passing an unknown string downstream.

The registry is consulted by :meth:`~grafast_py.pg.resource.PgResource.codec_for`, which
hands an attribute its codec by Postgres type NAME; the codec rides the EXISTING
``has_decoders`` / ``decode_rows`` materialisation path (decode is post-grouping, so it is
dedup-neutral — a codec never changes the emitted SQL skeleton or any step's dedup key),
and its ``sql_type`` is fed into the resource ``column_types`` so the keyset CAST keeps
working for a non-native (array/range/numeric/timestamptz) order column.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from sqlalchemy.dialects.postgresql import (
    ARRAY,
    DATERANGE,
    INT4RANGE,
    INT8RANGE,
    NUMRANGE,
    TSRANGE,
    TSTZRANGE,
)
from sqlalchemy.types import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Numeric,
    Time,
    TypeEngine,
)

from .resource import PgCodec

# A codec's ``to_py`` is applied per element/bound/field; type it once for the recursion.
Decoder = Callable[[Any], Any]


def apply_to_py(codec: Optional[PgCodec], value: Any) -> Any:
    """Decode one value via ``codec.to_py`` (a no-op when the codec/hook is absent or NULL).

    The recursive container codecs (array/range/composite) all decode their parts through
    this one rule so a NULL element/bound/field passes through untouched and a codec with
    no ``to_py`` (a native scalar) is a pass-through — keeping the recursion uniform.
    """
    if value is None or codec is None or codec.to_py is None:
        return value
    return codec.to_py(value)


def array_codec(element: PgCodec, *, sql_type: Optional[TypeEngine] = None) -> PgCodec:
    """A codec for a Postgres ARRAY whose elements decode via ``element``.

    asyncpg hands back a Python ``list``; ``to_py`` maps the element codec's ``to_py`` over
    every element (NULL elements pass through). The resulting ``sql_type`` is
    ``ARRAY(element.sql_type)`` (or an explicit override) so a keyset cursor over an array
    column casts the text-origin value back to the array type. A native-element array
    (``int4[]`` whose elements need no transform) still gets an array codec when its column
    is non-native for the cursor cast; with a native element AND no cast type it degenerates
    to a pass-through.
    """
    element_to_py = element.to_py
    array_sql_type = sql_type
    if array_sql_type is None and element.sql_type is not None:
        array_sql_type = ARRAY(element.sql_type)

    if element_to_py is None:
        # native-element array: no per-element transform, only the array sql_type (for the
        # cursor cast) matters; decoding is the identity list copy the materialiser already does.
        return PgCodec(to_py=None, sql_type=array_sql_type)

    def to_py(value: Optional[Sequence[Any]]) -> Optional[List[Any]]:
        # a NULL array column passes through (a nullable text[]/int4[] returning SQL NULL),
        # mirroring the range/composite container guards — only a present list is mapped.
        if value is None:
            return value
        return [apply_to_py(element, item) for item in value]

    return PgCodec(to_py=to_py, sql_type=array_sql_type)


def range_codec(
    bound: PgCodec, *, sql_type: Optional[TypeEngine] = None
) -> PgCodec:
    """A codec for a Postgres RANGE whose finite bounds decode via ``bound``.

    asyncpg returns an ``asyncpg.Range`` (``.lower`` / ``.upper`` / ``.lower_inc`` /
    ``.upper_inc`` / ``.isempty``); ``to_py`` flattens it into a plain dict
    ``{lower, upper, lower_inc, upper_inc, empty}`` so a host depends on no asyncpg type,
    applying the bound codec's ``to_py`` to each FINITE bound (an unbounded/None bound stays
    ``None``). ``sql_type`` is the range's own SQL type (required to cast a range cursor
    value, since a range has no single element type to derive it from).
    """

    def to_py(value: Any) -> Dict[str, Any]:
        if value is None:
            return value
        return {
            "lower": apply_to_py(bound, value.lower),
            "upper": apply_to_py(bound, value.upper),
            "lower_inc": value.lower_inc,
            "upper_inc": value.upper_inc,
            "empty": value.isempty,
        }

    return PgCodec(to_py=to_py, sql_type=sql_type)


def composite_codec(
    fields: Sequence[Tuple[str, PgCodec]],
    *,
    sql_type: Optional[TypeEngine] = None,
) -> PgCodec:
    """A codec for a Postgres COMPOSITE (row) type with named, individually-coded fields.

    asyncpg returns a POSITIONAL ``tuple`` (record); ``to_py`` zips it against ``fields``
    (``(name, codec)`` in column order) into a ``{name: decoded}`` dict, recursing into each
    field's codec — so a composite of an array of a composite decodes all the way down. A
    field count mismatch fails loud (a misdeclared composite would otherwise silently drop
    or misalign columns).
    """
    field_list = list(fields)

    def to_py(value: Any) -> Dict[str, Any]:
        if value is None:
            return value
        if len(value) != len(field_list):
            raise ValueError(
                f"composite codec expects {len(field_list)} fields "
                f"({[n for n, _ in field_list]}) but the row has {len(value)} columns"
            )
        return {
            name: apply_to_py(field_codec, item)
            for (name, field_codec), item in zip(field_list, value)
        }

    return PgCodec(to_py=to_py, sql_type=sql_type)


def enum_codec(labels: Optional[Sequence[str]] = None) -> PgCodec:
    """A codec for a text-backed Postgres ENUM, optionally validating its label set.

    asyncpg returns the enum label as a plain ``str``; the codec is the identity ``to_py``
    plus, when ``labels`` is given, a membership check that fails loud on an unknown label
    (a value outside the declared set is a schema/migration drift, surfaced here rather than
    passed silently downstream). With no ``labels`` it is a pure pass-through (the enum text
    flows through unchanged). ``sql_type`` stays ``None``: an enum binds and casts as text.
    """
    if labels is None:
        return PgCodec(to_py=None)
    label_set = frozenset(labels)

    def to_py(value: Optional[str]) -> Optional[str]:
        # a NULL enum column is the ABSENCE of a value, not a drifted label — pass it
        # through (decode_row calls to_py directly, so the None guard must live here, not
        # only in apply_to_py's recursive short-circuit) before the membership check.
        if value is None:
            return value
        if value not in label_set:
            raise ValueError(
                f"enum value {value!r} is not in the declared label set "
                f"{sorted(label_set)}"
            )
        return value

    return PgCodec(to_py=to_py)


# ----------------------------------------------------------------- scalar registry
# The base scalar codecs keyed by Postgres type NAME (and common aliases). A NATIVE
# scalar (asyncpg already returns the right Python type) carries no ``to_py`` — it is a
# pass-through; its only role in the registry is to declare an ``sql_type`` for the keyset
# CAST when the column is NON-NATIVE for a text-origin cursor (timestamptz / numeric / date
# / time). Native int/text/bool columns bind directly and need no cast, so their codec
# carries no ``sql_type`` either. These scalar codecs are also the ELEMENT/BOUND/FIELD
# building blocks the recursive container codecs compose.
_SCALAR_CODECS: Dict[str, PgCodec] = {
    # native scalars: pass-through, no cast type needed.
    "int2": PgCodec(),
    "smallint": PgCodec(),
    "int4": PgCodec(),
    "integer": PgCodec(),
    "int": PgCodec(),
    "int8": PgCodec(sql_type=BigInteger()),
    "bigint": PgCodec(sql_type=BigInteger()),
    "text": PgCodec(),
    "varchar": PgCodec(),
    "bpchar": PgCodec(),
    "char": PgCodec(),
    "bool": PgCodec(),
    "boolean": PgCodec(sql_type=Boolean()),
    "float4": PgCodec(sql_type=Float()),
    "float8": PgCodec(sql_type=Float()),
    "real": PgCodec(sql_type=Float()),
    "double precision": PgCodec(sql_type=Float()),
    # non-native scalars: a text-origin keyset cursor value must cast back to this type.
    "numeric": PgCodec(sql_type=Numeric()),
    "decimal": PgCodec(sql_type=Numeric()),
    "timestamptz": PgCodec(sql_type=DateTime(timezone=True)),
    "timestamp with time zone": PgCodec(sql_type=DateTime(timezone=True)),
    "timestamp": PgCodec(sql_type=DateTime()),
    "timestamp without time zone": PgCodec(sql_type=DateTime()),
    "date": PgCodec(sql_type=Date()),
    "time": PgCodec(sql_type=Time()),
    "timetz": PgCodec(sql_type=Time(timezone=True)),
}

# The range type name -> its element scalar name, so ``codec_for`` builds the range codec
# from the SCALAR registry rather than re-declaring each bound type. The range's own SQL
# type (needed to cast a range cursor) is paired alongside.
_RANGE_TYPES: Dict[str, Tuple[str, TypeEngine]] = {
    "int4range": ("int4", INT4RANGE()),
    "int8range": ("int8", INT8RANGE()),
    "numrange": ("numeric", NUMRANGE()),
    "tsrange": ("timestamp", TSRANGE()),
    "tstzrange": ("timestamptz", TSTZRANGE()),
    "daterange": ("date", DATERANGE()),
}


def normalize_type_name(pg_type_name: str) -> str:
    """Canonicalise a Postgres type name for registry lookup (lower-cased, trimmed).

    Accepts the array suffix form (``int4[]``) and the ``_int4`` array-OID form, leaving
    the bracket/underscore stripping to :func:`codec_for`; here it only lowercases and
    trims surrounding whitespace so ``TIMESTAMPTZ`` and ``timestamptz`` resolve alike.
    """
    return pg_type_name.strip().lower()


def codec_for(
    pg_type_name: str,
    *,
    composite_fields: Optional[Sequence[Tuple[str, str]]] = None,
    enum_labels: Optional[Sequence[str]] = None,
) -> PgCodec:
    """Return the :class:`PgCodec` for a Postgres type by NAME (recursive on containers).

    Resolution order:

    - an ARRAY (a ``name[]`` suffix or the ``_name`` array-OID form) builds an
      :func:`array_codec` over the ELEMENT type's codec (recursing — ``int4[][]`` nests);
    - a RANGE name (``int4range`` ...) builds a :func:`range_codec` over its bound scalar's
      codec, carrying the range's own ``sql_type``;
    - a COMPOSITE (``composite_fields`` given, mapping ``(field_name, field_type_name)``)
      builds a :func:`composite_codec` whose field codecs are each looked up here BY NAME — so
      array / range / scalar fields recurse, but a field that is ITSELF a composite or enum
      (which would need its own ``composite_fields`` / ``enum_labels``) is not expressible by
      name here; build such a nested codec with the :func:`composite_codec` / :func:`enum_codec`
      constructors directly (those recurse fully over any field codecs you pass);
    - an ENUM (``enum_labels`` given) builds an :func:`enum_codec` validating the labels;
    - otherwise the base SCALAR codec.

    Raises ``KeyError`` LOUDLY for an unknown scalar type rather than returning a silent
    pass-through — an unregistered type is a declaration gap to surface, not to paper over.
    """
    name = normalize_type_name(pg_type_name)

    if composite_fields is not None:
        return composite_codec(
            [(field_name, codec_for(field_type)) for field_name, field_type in composite_fields]
        )
    if enum_labels is not None:
        return enum_codec(enum_labels)

    # ARRAY: a trailing ``[]`` (``int4[]``) or the array-OID underscore form (``_int4``).
    if name.endswith("[]"):
        return array_codec(codec_for(name[:-2]))
    if name.startswith("_"):
        return array_codec(codec_for(name[1:]))

    if name in _RANGE_TYPES:
        element_name, range_sql_type = _RANGE_TYPES[name]
        return range_codec(codec_for(element_name), sql_type=range_sql_type)

    codec = _SCALAR_CODECS.get(name)
    if codec is None:
        raise KeyError(
            f"no codec registered for Postgres type {pg_type_name!r}; register a scalar "
            "codec, or pass composite_fields= / enum_labels= for a composite/enum type"
        )
    return codec


def scalar_codecs() -> Mapping[str, PgCodec]:
    """The base scalar codec registry (read-only view), for inspection / extension tests."""
    return dict(_SCALAR_CODECS)


__all__ = [
    "Decoder",
    "apply_to_py",
    "array_codec",
    "range_codec",
    "composite_codec",
    "enum_codec",
    "codec_for",
    "scalar_codecs",
    "normalize_type_name",
]
