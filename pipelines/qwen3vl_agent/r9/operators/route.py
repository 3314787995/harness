"""Route replay maintains arrival heading and checks stop before next action."""

from itertools import pairwise

from ..types import MissingCapability
from .direction import turn
from .distance import nearest


def edge_for(edges, origin, target):
    found = [e for e in edges if e["from"] == origin and e["to"] == target]
    if not found or any(e != found[0] for e in found[1:]):
        raise MissingCapability("route", f"missing or ambiguous edge {origin} -> {target}")
    return found[0]


def replay(path, heading, edges):
    if len(path) < 2 or heading is None:
        raise MissingCapability("route", "route requires start, waypoints and initial heading")
    turns = []
    for a, b in pairwise(path):
        edge = edge_for(edges, a, b)
        if edge["departure_heading"] is None or edge["arrival_heading"] is None:
            raise MissingCapability("route", "topological edge lacks departure/arrival heading")
        turns.append(turn(heading, edge["departure_heading"]))
        heading = edge["arrival_heading"]
    return turns


def next_action(progress, waypoints, edges):
    if progress["stop_reached"]:
        return "Stop"
    done = progress["completed_waypoints"]
    if waypoints[: len(done)] != done or progress["instruction_index"] != len(done):
        raise MissingCapability("route", "route progress contradicts the instruction sequence")
    if len(done) >= len(waypoints):
        raise MissingCapability("route", "waypoints completed but stop condition not established")
    label = replay([progress["current"], waypoints[len(done)]], progress["heading"], edges)[0]
    return {
        "left": "Turn left and move forward",
        "right": "Turn right and move forward",
        "straight": "Move forward",
        "back": "Turn back and move forward",
    }[label]


def compare_paths(paths, edges):
    costs = {}
    for i, path in enumerate(paths):
        if len(path) < 2:
            raise MissingCapability("route", "empty candidate path")
        total = [0.0, 0.0]
        for a, b in pairwise(path):
            edge = edge_for(edges, a, b)
            cost = edge["cost_seconds"]
            if cost is None or cost[0] < 0 or cost[0] > cost[1]:
                raise MissingCapability(
                    "route", "fastest path needs observed travel-time cost, not turn count"
                )
            total = [total[0] + cost[0], total[1] + cost[1]]
        costs[i] = total
    return paths[nearest(costs)]
