"""Explicit horizontal reference frames and angular intervals."""

import math

from ..types import MissingCapability

EPS = 1e-9


def wrap(degrees):
    return (degrees + 180) % 360 - 180


def label_angle(angle, kind, threshold=None):
    angle = wrap(angle)
    if kind == "left_right_back" and threshold is not None and abs(angle) >= threshold - EPS:
        return "back"
    if abs(math.sin(math.radians(angle))) < EPS:
        raise MissingCapability("reference_frame", "target lies on the query axis")
    horizontal = "right" if angle > 0 else "left"
    if kind in {"left_right", "left_right_back"}:
        return horizontal
    if abs(math.cos(math.radians(angle))) < EPS:
        raise MissingCapability("reference_frame", "target lies on a quadrant boundary")
    return ("front" if abs(angle) < 90 else "back") + "-" + horizontal


def bearing(origin, forward, target, kind="four_quadrants", threshold=None):
    if len(origin) != len(forward) or len(origin) != len(target) or len(origin) not in {2, 3}:
        raise MissingCapability(
            "reference_frame", "points require one common 2D/3D coordinate system"
        )
    # Horizontal plane: scene x/y, with +z up. No image-coordinate interpretation.
    fx, fy = forward[0] - origin[0], forward[1] - origin[1]
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    if math.hypot(fx, fy) < EPS or math.hypot(dx, dy) < EPS:
        raise MissingCapability("reference_frame", "coincident origin/forward/target")
    x, y = dx * fy - dy * fx, dx * fx + dy * fy
    angle = math.degrees(math.atan2(x, y))
    return label_angle(angle, kind, threshold), angle


def interval_bearing(interval, kind, threshold=None):
    lo, hi = interval
    if lo > hi or hi - lo >= 180:
        raise MissingCapability(
            "reference_frame", "angular interval wraps or spans incompatible directions"
        )
    points = [lo, hi, (lo + hi) / 2]
    boundaries = [-180, -90, 0, 90, 180]
    if threshold:
        boundaries += [-threshold, threshold]
    points += [b for b in boundaries if lo < b < hi]
    labels = {label_angle(a, kind, threshold) for a in points}
    if len(labels) != 1:
        raise MissingCapability("reference_frame", "angular interval crosses answer boundaries")
    return labels.pop()


def heading_delta(initial, current):
    lo, hi = current[0] - initial[1], current[1] - initial[0]
    if hi - lo >= 180 or math.floor((lo + 180) / 360) != math.floor((hi + 180) / 360):
        raise MissingCapability("reference_frame", "heading uncertainty crosses the wrap boundary")
    return [wrap(lo), wrap(hi)]


def turn(initial, departure):
    delta = wrap(departure - initial)
    if abs(delta) < EPS:
        return "straight"
    if abs(abs(delta) - 180) < EPS:
        return "back"
    return "right" if delta > 0 else "left"


def transform_point(point, rotation_degrees, translation):
    if len(point) != len(translation):
        raise MissingCapability("reference_frame", "alignment dimensionality mismatch")
    a = math.radians(rotation_degrees)
    # Positive yaw is clockwise relative to +y forward.
    x = math.cos(a) * point[0] + math.sin(a) * point[1] + translation[0]
    y = -math.sin(a) * point[0] + math.cos(a) * point[1] + translation[1]
    return [x, y] + ([point[2] + translation[2]] if len(point) == 3 else [])
