from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from scenetiq.domain.scene import Aspect


class Relation(StrEnum):
    ABOVE = "above"
    BELOW = "below"
    ON = "on"
    UNDER = "under"
    NEAR = "near"
    FAR_FROM = "far_from"
    BETWEEN = "between"
    NEAREST_TO = "nearest_to"
    FARTHEST_FROM = "farthest_from"
    LARGEST = "largest"
    SMALLEST = "smallest"
    TALLEST = "tallest"
    LOWEST = "lowest"

    @property
    def anchor_count(self) -> int:
        if self in (Relation.LARGEST, Relation.SMALLEST, Relation.TALLEST, Relation.LOWEST):
            return 0
        return 2 if self is Relation.BETWEEN else 1


MAX_QUERY_OBJECTS = 4
MAX_CONSTRAINTS = 4


@dataclass(frozen=True)
class QueryObject:
    object_id: str
    phrase: str
    aspects: tuple[Aspect, ...] = ()


@dataclass(frozen=True)
class Constraint:
    relation: Relation
    subject: str
    anchors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedQuery:
    objects: tuple[QueryObject, ...]
    target: str
    constraints: tuple[Constraint, ...] = ()
    unsupported: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("a parsed Query needs at least one object")
        if len(self.objects) > MAX_QUERY_OBJECTS:
            raise ValueError(f"a parsed Query has at most {MAX_QUERY_OBJECTS} objects")
        if len(self.constraints) > MAX_CONSTRAINTS:
            raise ValueError(f"a parsed Query has at most {MAX_CONSTRAINTS} constraints")
        ids = [o.object_id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError(f"a parsed Query repeats an object id: {ids}")
        if self.target not in ids:
            raise ValueError(f"the target {self.target!r} is not one of the objects {ids}")
        for c in self.constraints:
            for named in (c.subject, *c.anchors):
                if named not in ids:
                    raise ValueError(f"{c.relation} names object {named!r}, which is not listed")
            if len(c.anchors) != c.relation.anchor_count:
                raise ValueError(
                    f"{c.relation} takes {c.relation.anchor_count} Anchor(s), got {len(c.anchors)}"
                )
            if c.subject in c.anchors:
                raise ValueError(f"{c.relation} uses object {c.subject!r} as its own Anchor")
            if len(set(c.anchors)) != len(c.anchors):
                raise ValueError(f"{c.relation} names one Anchor twice: {list(c.anchors)}")

    @classmethod
    def without_relations(cls, text: str) -> ParsedQuery:
        return cls(objects=(QueryObject("a", text),), target="a")


class QueryParseError(Exception): ...
