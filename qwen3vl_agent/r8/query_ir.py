"""Topological query compiler with explicit provenance and bounded integer growth."""

from dataclasses import dataclass, field
from fractions import Fraction

from .evidence import DEFINITIONS
from .operators import OPS, calculate
from .types import ModelingError, digest
from .units import IntervalQuantity, Quantity, Unit, encode, number


def typed(value, unit="1", basis="", digits=100):
    if value is None:
        raise ModelingError("unreadable/missing input")
    if isinstance(value, bool):
        return value
    if isinstance(value, list):
        return [typed(v, unit, basis, digits) for v in value]
    if isinstance(value, dict):
        if set(value) == {"value", "unit"}:
            return typed(value["value"], value["unit"], basis, digits)
        return {k: typed(v, "1", "", digits) for k, v in value.items()}
    if isinstance(value, Quantity):
        return value
    try:
        result = number(value, digits)
    except ModelingError:
        # Calendar and clock values stay text until the explicit parsing node.
        if (Unit.parse(unit).dimensions or Unit.parse(unit).factor != 1) and unit not in {
            "date",
            "clock",
        }:
            raise
        if not isinstance(value, str):
            raise
        return value
    return Quantity(result, Unit.parse(unit), basis)


@dataclass
class Execution:
    value: object
    nodes: dict = field(default_factory=dict)
    parents: list = field(default_factory=list)
    checks: list = field(default_factory=list)

    def to_dict(self):
        return {
            "value": encode(self.value),
            "nodes": encode(self.nodes),
            "parents": self.parents,
            "checks": self.checks,
        }


class Executor:
    def __init__(self, store, config, imported=None):
        self.store, self.config = store, config
        self.imported = imported or {}

    def run(self, query):
        if (
            set(query) != {"nodes", "target_node", "output_unit"}
            or len(query["nodes"]) > self.config.max_query_nodes
        ):
            raise ModelingError("invalid query graph")
        values, dependencies, depths, checks = {}, {}, {}, []

        def resolve(arg):
            if isinstance(arg, dict) and set(arg) == {"refs"}:
                if (
                    not isinstance(arg["refs"], list)
                    or not 1 <= len(arg["refs"]) <= 128
                    or any(not isinstance(r, str) for r in arg["refs"])
                ):
                    raise ModelingError("reference vector must contain 1..128 bound IDs")
                parts = [resolve(r) for r in arg["refs"]]
                return (
                    [v[0] for v in parts],
                    sorted({p for v in parts for p in v[1]}),
                    max(v[2] for v in parts),
                )
            if isinstance(arg, str):
                if arg in values:
                    return values[arg], dependencies[arg], depths[arg]
                if arg in self.imported:
                    value = self.imported[arg]
                    return value["value"], value["parents"], 0
                row = self.store.get(arg)
                if row["unresolved"] or row["alternatives"]:
                    raise ModelingError(
                        "ambiguous variable requires candidate evaluation or repair"
                    )
                ref = f"{row['id']}@{row['version']}"
                return (
                    typed(
                        row["value"], row["unit"], row["unit_basis"], self.config.max_numeric_digits
                    ),
                    [ref],
                    0,
                )
            if not isinstance(arg, dict) or set(arg) != {"value", "unit", "source"}:
                raise ModelingError("literal requires value/unit/source")
            source = arg["source"]
            if source.startswith("definition:"):
                key = source.split(":", 1)[1]
                if key not in DEFINITIONS or (str(arg["value"]), arg["unit"]) != DEFINITIONS[key]:
                    raise ModelingError("unregistered mathematical constant")
                return typed(arg["value"], arg["unit"]), [], 0
            row = self.store.get(source)
            if row["origin"] not in {"given", "hypothetical"} or (arg["value"], arg["unit"]) != (
                row["value"],
                row["unit"],
            ):
                raise ModelingError("literal is not an exact sourced question value")
            return typed(arg["value"], arg["unit"]), [f"{row['id']}@{row['version']}"], 0

        for node in query["nodes"]:
            if set(node) != {"id", "op", "args", "params"} or node["op"] not in OPS:
                raise ModelingError("invalid node")
            key = node["id"]
            if (
                not isinstance(key, str)
                or not key
                or key in values
                or key in self.imported
                or key in self.store.state["current"]
                or "@" in key
            ):
                raise ModelingError("duplicate/shadowed query node")
            args = [resolve(v) for v in node["args"]]
            depth = 1 + max((x[2] for x in args), default=0)
            if depth > self.config.max_query_depth:
                raise ModelingError("query depth limit")
            value = calculate(node["op"], [x[0] for x in args], node["params"])
            self._bound(value)
            if node["op"] in {
                "nonzero_denominator",
                "nonnegative",
                "recompute",
                "minimality",
                "target_uniqueness",
                "domain",
            }:
                checks.append({"node": key, "passed": value is True})
                if value is not True:
                    raise ModelingError(f"mathematical check failed: {key}")
            values[key] = value
            dependencies[key] = sorted({ref for a in args for ref in a[1]})
            depths[key] = depth
        target = query["target_node"]
        value, parents, _ = resolve(target)
        if isinstance(value, (Quantity, IntervalQuantity)):
            value = value.convert(query["output_unit"])
        result = Execution(value, values, parents, checks)
        self.store.add_derived(
            "query:" + (target if isinstance(target, str) else digest(target)),
            result.to_dict(),
            parents,
        )
        return result

    def _bound(self, value):
        if isinstance(value, IntervalQuantity):
            number(value.lower, self.config.max_numeric_digits)
            number(value.upper, self.config.max_numeric_digits)
        if isinstance(value, Quantity) and isinstance(value.value, Fraction):
            number(value.value, self.config.max_numeric_digits)
        elif isinstance(value, (list, dict)):
            if len(value) > 4096:
                raise ModelingError("collection size limit")
            for v in value.values() if isinstance(value, dict) else value:
                self._bound(v)
