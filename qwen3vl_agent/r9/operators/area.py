"""Area of supported, simple, disjoint floor polygons; no convex-hull completion."""

from ..types import MissingCapability

EPS = 1e-9


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _edges(p):
    return list(zip(p, p[1:] + p[:1]))


def _proper_intersection(a, b, c, d):
    return _cross(a, b, c) * _cross(a, b, d) < -EPS and _cross(c, d, a) * _cross(c, d, b) < -EPS


def _inside(point, polygon):
    inside = False
    x, y = point
    for a, b in _edges(polygon):
        if (
            abs(_cross(a, b, point)) < EPS
            and min(a[0], b[0]) - EPS <= x <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= y <= max(a[1], b[1]) + EPS
        ):
            return False  # Shared boundaries have zero area.
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            inside = not inside
    return inside


def polygon_area(polygon):
    if len(polygon) < 3 or len(set(map(tuple, polygon))) != len(polygon):
        raise MissingCapability("coverage", "invalid/duplicate floor polygon vertices")
    edges = _edges(polygon)
    if any(
        _proper_intersection(a, b, c, d)
        for i, (a, b) in enumerate(edges)
        for j, (c, d) in enumerate(edges)
        if i + 1 < j and (i, j) != (0, len(edges) - 1)
    ):
        raise MissingCapability("coverage", "self-intersecting floor polygon")
    result = abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in edges)) / 2
    if result <= EPS:
        raise MissingCapability("coverage", "degenerate floor polygon")
    return result


def floor_area(regions, coverage_complete):
    if not coverage_complete:
        raise MissingCapability("coverage", "observed region is not the whole room footprint")
    if len({r["id"] for r in regions}) != len(regions):
        raise MissingCapability("coverage", "duplicate room region / revisit")
    polygons = [r["polygon"] for r in regions]
    areas = [polygon_area(p) for p in polygons]
    for i, a in enumerate(polygons):
        for b in polygons[i + 1 :]:
            crossing = any(
                _proper_intersection(x, y, u, v) for x, y in _edges(a) for u, v in _edges(b)
            )
            contained = any(_inside(p, b) for p in a) or any(_inside(p, a) for p in b)
            # Coincident polygons and partial collinear overlap require explicit union support.
            same = set(map(tuple, a)) == set(map(tuple, b))
            midpoints = [[(x[0] + y[0]) / 2, (x[1] + y[1]) / 2] for x, y in _edges(a)]
            other_midpoints = [[(x[0] + y[0]) / 2, (x[1] + y[1]) / 2] for x, y in _edges(b)]
            if (
                crossing
                or contained
                or same
                or any(_inside(p, b) for p in midpoints)
                or any(_inside(p, a) for p in other_midpoints)
            ):
                raise MissingCapability(
                    "coverage", "overlapping floor regions need a validated union"
                )
    return sum(areas)
