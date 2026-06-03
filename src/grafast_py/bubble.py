"""The Bubble value: a non-null violation propagating to a nullable boundary.

A non-null field/item/wrapper that completes to None produces a `Bubble` carrying
the already-located `GraphQLError`. The bubble travels up through enclosing
non-null completers unchanged and is caught at the first nullable boundary
(nullable field, nullable list item, nullable wrapper), where its error is
appended exactly once and the value becomes None. An uncaught bubble at a field's
top boundary nulls the parent object dict; at the operation root it surfaces as a
raised error (data -> None).
"""

from typing import NamedTuple

from graphql.error import GraphQLError


class Bubble(NamedTuple):
    """A propagating non-null violation; `error` is appended once when caught."""

    error: GraphQLError
