"""Distance semantics, interval ranking and explicit metric conversion."""

import math

from ..types import MissingCapability

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}


def closest_boundary(a, b, *, complete_a=True, complete_b=True):
    if not complete_a or not complete_b:
        raise MissingCapability(
            "boundary", "partial visible surfaces do not determine closest boundaries"
        )
    if not a or not b or len({len(p) for p in [*a, *b]}) != 1:
        raise MissingCapability("boundary", "boundary samples missing or have mixed dimensions")
    return min(math.dist(x, y) for x in a for y in b)


def nearest(intervals):
    if not intervals:
        raise MissingCapability("relation", "no candidate distances")
    winners = [
        key
        for key, (_, upper) in intervals.items()
        if all(key == other or upper < lower for other, (lower, _) in intervals.items())
    ]
    if len(winners) != 1:
        raise MissingCapability("boundary", "candidate distance intervals overlap or tie")
    return winners[0]


def multiply_positive(a, b):
    if a[0] < 0 or b[0] <= 0 or a[0] > a[1] or b[0] > b[1]:
        raise MissingCapability("scale", "invalid positive scale/measurement interval")
    return [a[0] * b[0], a[1] * b[1]]


def metric_scale(value, measured_entity):
    if value["kind"] == "unknown":
        raise MissingCapability("scale", "absolute scale is unknown")
    if value["kind"] == "category_prior" and value["reference_entity"] == measured_entity:
        raise MissingCapability(
            "scale", "target category size cannot independently calibrate the target"
        )
    given = value.get("reference_length_meters")
    scene = value.get("reference_length_scene")
    if value["kind"] in {"readable_ruler", "question_bound_dimension"}:
        if not given or not scene or given[0] <= 0 or scene[0] <= 0:
            raise MissingCapability(
                "scale", "physical dimension lacks a bound scene correspondence"
            )
        return [given[0] / scene[1], given[1] / scene[0]]
    return value["meters_per_scene_unit"]


def convert(interval, source_unit, target_unit, scale=None, power=1):
    def dimension(unit):
        return 2 if (unit or "").replace("²", "2").replace("^2", "2").endswith("2") else 1

    if dimension(source_unit) != power or dimension(target_unit) != power:
        raise MissingCapability("scale", "length and area units cannot be interchanged")

    def base(unit):
        return (unit or "").replace("²", "2").replace("^2", "2").removesuffix("2")

    source, target = base(source_unit), base(target_unit)
    if source == target and source in {*UNITS, "arbitrary_scene_unit"}:
        return list(interval)
    if target not in UNITS:
        raise MissingCapability("scale", f"unsupported output unit: {target_unit}")
    if source in UNITS:
        factor = (UNITS[source] / UNITS[target]) ** power
        return [x * factor for x in interval]
    if source != "arbitrary_scene_unit" or scale is None:
        raise MissingCapability(
            "scale", "scale-free geometry cannot be implicitly converted to metric units"
        )
    scaled = [x**power for x in scale]
    return [x / UNITS[target] ** power for x in multiply_positive(interval, scaled)]
