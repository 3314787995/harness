"""Finite geometry rules with explicit entity mappings, premises and provenance."""

from copy import deepcopy

from .types import ModelingError
from .units import Unit


def ref(name):
    return {"ref": name}


def op(name, *args):
    return {"op": name, "args": list(args)}


RULES = {
    "rectangle": ("rectangle", ("width", "height", "area")),
    "square": ("square", ("side", "area")),
    "segment_addition": ("on_segment", ("left", "right", "whole")),
    "triangle_angles": ("triangle", ("a", "b", "c")),
    "shared_base": ("shared_base", ("area1", "area2", "height1", "height2")),
    "shared_height": ("shared_height", ("area1", "area2", "base1", "base2")),
    "pythagoras": ("right_triangle", ("a", "b", "c")),
    "similar_triangles": ("similar_triangles", ("a1", "b1", "c1", "a2", "b2", "c2")),
    "rectangle_partition": ("rectangle_partition", ("tl", "tr", "bl", "br", "total")),
}


def apply_rules(program, relations, config):
    symbols = {s["id"]: s for s in program["symbols"]}
    constraints, trace, signatures = [], [], set()
    for application in program["rules"]:
        name = application["rule"]
        if name not in RULES:
            raise ModelingError("unsupported geometry rule")
        required, keys = RULES[name]
        mapping = application["mapping"]
        if set(mapping) != set(keys) or any(v not in symbols for v in mapping.values()):
            raise ModelingError("incomplete geometry rule mapping")
        premises = []
        for rid in application["premises"]:
            if rid not in relations:
                raise ModelingError("geometry premise missing")
            relation = relations[rid]
            if relation["snapshot"] != application["snapshot"]:
                raise ModelingError("geometry premises cross snapshots")
            premises.append(relation)
        candidates = [r for r in premises if r["kind"] == required]
        matched = next(
            (
                r
                for r in candidates
                if all(r["objects"].get(k) == symbols[mapping[k]]["entity_id"] for k in keys)
            ),
            None,
        )
        if matched is None:
            raise ModelingError("required relation/object correspondence absent")
        if any(symbols[v]["snapshot"] != application["snapshot"] for v in mapping.values()):
            raise ModelingError("geometry variable snapshot mismatch")
        if required not in {"on_segment", "triangle", "rectangle_partition"} and matched[
            "source_kind"
        ] not in {"explicit_marker", "given"}:
            raise ModelingError("metric theorem premise requires a marker or question stipulation")
        objects = {matched["objects"][key] for key in keys}
        nondegenerate = {
            e
            for r in premises
            if r["kind"] in {"positive", "nondegenerate"}
            for e in r["objects"].values()
        }
        # The required relation is a typed relation: rectangle_partition means a complete 2x2
        # rectangular grid with shared row heights/column widths, not merely four nearby regions.
        covered = objects <= nondegenerate
        if name in {"rectangle", "square"}:
            covered |= matched["objects"]["area"] in nondegenerate
        elif name in {"shared_base", "shared_height"}:
            covered |= {matched["objects"][k] for k in ("area1", "area2")} <= nondegenerate
        elif name in {"triangle_angles", "pythagoras"}:
            covered |= matched["objects"].get("region") in nondegenerate
        elif name == "similar_triangles":
            covered |= all(
                matched["objects"].get(k) in nondegenerate for k in ("region1", "region2")
            )
        elif name == "rectangle_partition":
            covered |= {matched["objects"][k] for k in ("tl", "tr", "bl", "br")} <= nondegenerate
        if not covered:
            raise ModelingError("nondegeneracy premise missing for one or more mapped objects")
        signature = (name, tuple(sorted(mapping.items())), application["snapshot"])
        if signature in signatures:
            continue
        signatures.add(signature)
        if len(signatures) > config.max_geometry_rules:
            raise ModelingError("geometry rule application limit")
        values = {k: ref(v) for k, v in mapping.items()}

        def eq(a, b, application=application):
            constraints.append(
                {
                    "id": f"rule:{application['id']}:{len(constraints)}",
                    "relation": "equal",
                    "args": [a, b],
                    "sources": application["premises"],
                    "snapshot": application["snapshot"],
                }
            )

        def dims(key, unit, mapping=mapping):
            if Unit.parse(symbols[mapping[key]]["unit"]).dimensions != Unit.parse(unit).dimensions:
                raise ModelingError("geometry measurement type/unit mismatch")

        if name in {"rectangle", "square"}:
            dims("area", "m^2")
            if name == "rectangle":
                dims("width", "m")
                dims("height", "m")
                eq(values["area"], op("multiply", values["width"], values["height"]))
            else:
                dims("side", "m")
                eq(values["area"], op("square", values["side"]))
        elif name == "segment_addition":
            for k in keys:
                dims(k, "m")
            eq(values["whole"], op("add", values["left"], values["right"]))
        elif name == "triangle_angles":
            for k in keys:
                dims(k, "degree")
            eq(
                op("add", op("add", values["a"], values["b"]), values["c"]),
                {"constant": "triangle_degrees"},
            )
        elif name in {"shared_base", "shared_height"}:
            for k in keys:
                dims(k, "m^2" if k.startswith("area") else "m")
            k1, k2 = keys[2:]
            eq(
                op("multiply", values["area1"], values[k2]),
                op("multiply", values["area2"], values[k1]),
            )
        elif name == "pythagoras":
            for k in keys:
                dims(k, "m")
            eq(
                op("add", op("square", values["a"]), op("square", values["b"])),
                op("square", values["c"]),
            )
        elif name == "similar_triangles":
            for k in keys:
                dims(k, "m")
            eq(
                op("multiply", values["a1"], values["b2"]),
                op("multiply", values["b1"], values["a2"]),
            )
            eq(
                op("multiply", values["a1"], values["c2"]),
                op("multiply", values["c1"], values["a2"]),
            )
        else:
            for k in keys:
                dims(k, "m^2")
            eq(
                op("multiply", values["tl"], values["br"]),
                op("multiply", values["tr"], values["bl"]),
            )
            eq(
                values["total"],
                op(
                    "add",
                    op("add", values["tl"], values["tr"]),
                    op("add", values["bl"], values["br"]),
                ),
            )
        # All measures in these nondegenerate geometry rules are positive, except angle sums
        # whose positive-angle premise is equally part of the nondegenerate triangle definition.
        for key in keys:
            constraints.append(
                {
                    "id": f"rule:{application['id']}:positive:{key}",
                    "relation": "greater",
                    "args": [values[key], {"constant": "zero"}],
                    "sources": application["premises"],
                    "snapshot": application["snapshot"],
                }
            )
        trace.append(
            {"application": deepcopy(application), "premises": [r["id"] for r in premises]}
        )
    return constraints, trace
