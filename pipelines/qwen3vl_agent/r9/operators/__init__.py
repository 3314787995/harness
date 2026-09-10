"""Typed spatial queries. Missing prerequisites return gaps, never invented inputs."""

import math
from copy import deepcopy

from ..types import MissingCapability, ProtocolError, QueryResult
from . import area, direction, distance, route, viewpoint


class Executor:
    def __init__(self, state, spec, contract, query_time=None, query_scope=None):
        self.state, self.spec, self.contract = state, spec, contract
        self.query_time = query_time
        self.query_scope = query_scope
        self.used = {}
        self.steps = []
        self.scales = []
        self.estimated = False

    def take(self, record):
        self.used[record["record_id"]] = record
        for rid in record["parent_record_ids"]:
            if rid in self.state.data["invalid_records"]:
                raise MissingCapability("relation", "derived record depends on invalid evidence")
            self.take(self.state.data["records"][rid])
        return record["value"]

    def one(self, predicate, arguments=None, **kwargs):
        record = self.state.one(predicate, arguments, **kwargs)
        self.take(record)
        return record

    def event_time(self, description, ordinal=1, *, query=True):
        window = self.query_scope if query else None
        records = self.state.select("event")
        events = [
            r
            for r in records
            if r["value"]["description"].strip().casefold() == description.strip().casefold()
            and not r["value"]["replay_of"]
            and (
                window is None
                or window[0] <= r["value"]["start_s"] <= r["value"]["end_s"] <= window[1]
            )
        ]
        # Repeated observation of the same event is not a new occurrence.
        unique = {}
        for r in events:
            v = r["value"]
            unique.setdefault((v["start_s"], v["end_s"], tuple(r["arguments"])), r)
        events = sorted(unique.values(), key=lambda r: r["value"]["start_s"])
        if len(events) < ordinal:
            raise MissingCapability("time", f"missing occurrence {ordinal} of {description}")
        value = self.take(events[ordinal - 1])
        # Only an explicit sequential scan coverage supports an ordinal; finite sampling remains estimated.
        start = window[0] if window else self.contract.allowed_time_intervals[0][0]
        end = value["end_s"]
        spans = sorted(
            c["span"]
            for c in self.state.data["coverage"]
            if c.get("sequential") and c.get("completed")
        )
        cursor = start
        for a, b in spans:
            if a <= cursor + 1e-6:
                cursor = max(cursor, b)
        if cursor < end - 1e-6:
            raise MissingCapability(
                "time", "ordinal event needs a sequential scan of the preceding permitted interval"
            )
        self.estimated = True  # Finite sampling cannot certify unseen event absence.
        return value["end_s"]

    def anchor_time(self, anchor, *, query=False):
        if not anchor:
            return None
        intervals = (
            (self.query_scope,)
            if query and self.query_scope
            else self.contract.allowed_time_intervals
        )
        if anchor["kind"] == "start":
            return intervals[0][0]
        if anchor["kind"] == "end":
            return intervals[-1][1]
        if anchor["kind"] == "time":
            return anchor["time_s"]
        return self.event_time(anchor["description"], anchor["ordinal"] or 1, query=query)

    def points(self, entities, predicate="point", time=None):
        groups = [self.state.select(predicate, [e], time=time) for e in entities]
        if any(not group for group in groups):
            raise MissingCapability(
                "boundary" if predicate == "boundary" else "relation",
                f"missing common {predicate} geometry",
                entities=entities,
            )
        common = set.intersection(*[{r["reference_frame_id"] for r in group} for group in groups])
        if len(common) != 1:
            # A real alignment record is required; a connected scene graph alone is not a transform.
            if not common:
                return self.aligned_points(entities, groups, predicate)
            raise MissingCapability(
                "reference_frame", "multiple unresolved geometry frames", entities=entities
            )
        fid = common.pop()
        records = [self.one(predicate, [e], time=time, frame=fid) for e in entities]
        if any(r["value"]["space"] != "scene" for r in records):
            raise MissingCapability(
                "reference_frame", "image geometry cannot answer a scene-space query"
            )
        if len({r["unit"] for r in records}) != 1:
            raise MissingCapability("scale", "points use incompatible coordinate units")
        return records, fid

    def aligned_points(self, entities, groups, predicate):
        if predicate != "point" or any(len(g) != 1 for g in groups):
            raise MissingCapability(
                "reference_frame", "disconnected geometry components", entities=entities
            )
        records = [deepcopy(g[0]) for g in groups]
        destination = records[0]["reference_frame_id"]
        for record in records:
            self.take(record)
            if record["value"]["space"] != "scene":
                raise MissingCapability("reference_frame", "ImagePoint cannot become ScenePoint")
            if record["unit"] != records[0]["unit"]:
                raise MissingCapability("scale", "alignment requires the same coordinate scale")
            if record["reference_frame_id"] == destination:
                continue
            matches = [
                r
                for r in self.state.select("alignment")
                if r["value"]["from_frame"] == record["reference_frame_id"]
                and r["value"]["to_frame"] == destination
            ]
            if len(matches) != 1:
                raise MissingCapability(
                    "reference_frame", "missing unique explicit frame alignment"
                )
            alignment = self.take(matches[0])
            if len(set(alignment["landmark_ids"])) < 2:
                raise MissingCapability(
                    "reference_frame", "alignment lacks distinct bridge landmarks"
                )
            record["value"]["coordinates"] = direction.transform_point(
                record["value"]["coordinates"],
                alignment["rotation_degrees"],
                alignment["translation"],
            )
            record["reference_frame_id"] = destination
            self.steps.append(
                {
                    "operator": "deterministic_transform",
                    "parent_record_ids": [matches[0]["record_id"], record["record_id"]],
                }
            )
        return records, destination

    def metric(self, interval, record, target_unit, entity, power=1):
        try:
            return distance.convert(interval, record["unit"], target_unit, power=power)
        except MissingCapability:
            scales = self.state.select("scale", frame=record["reference_frame_id"])
            if len(scales) != 1:
                raise MissingCapability(
                    "scale", "missing unambiguous scale in the geometry frame", entities=[entity]
                ) from None
            s = self.take(scales[0])
            measured_target = (
                entity if self.spec["operation"] == "max_extent" else "queried_measurement"
            )
            factor = distance.metric_scale(s, measured_target)
            self.scales.append(s["kind"])
            return distance.convert(interval, record["unit"], target_unit, factor, power)

    def distance_pair(self, entities, time=None):
        semantics = self.spec["measurement"]["geometry_semantics"]
        if semantics not in {"center", "car_front", "closest_boundary"}:
            raise MissingCapability("task", "distance semantics must be specified")
        direct = self.state.select("distance", entities, time=time)
        if direct:
            r = self.one("distance", entities, time=time)
            if r["value"]["semantics"] != semantics:
                raise MissingCapability(
                    "boundary", "distance record uses a different geometric definition"
                )
            return r["value"]["interval"], r
        kind = "boundary" if semantics == "closest_boundary" else "point"
        records, _ = self.points(entities, kind, time)
        if kind == "boundary":
            a, b = [r["value"] for r in records]
            value = distance.closest_boundary(
                a["points"], b["points"], complete_a=a["complete"], complete_b=b["complete"]
            )
        else:
            if semantics == "car_front":
                raise MissingCapability(
                    "boundary", "object center points cannot stand in for car fronts"
                )
            value = math.dist(*[r["value"]["coordinates"] for r in records])
        return [value, value], records[0]

    def execute(self, op, entities, inputs=(), params=None):
        params = params or {}
        if op == "event_select":
            return self.event_time(params["event_description"], params.get("ordinal", 1)), "s"
        if op == "collect":
            return [v for v, _ in inputs], None
        if op in {"add", "subtract", "compare", "unit_convert"}:
            if not inputs or any(
                not isinstance(v, (int, float)) or isinstance(v, bool) for v, _ in inputs
            ):
                raise MissingCapability("task", "numeric DAG helper requires scalar numeric inputs")
            if op == "unit_convert":
                if len(inputs) != 1:
                    raise MissingCapability("task", "unit_convert requires one input")
                v, u = inputs[0]
                return distance.convert([v, v], u, params["output_unit"])[0], params["output_unit"]
            if len({u for _, u in inputs}) != 1:
                raise MissingCapability("scale", "numeric DAG inputs have incompatible units")
            values = [v for v, _ in inputs]
            if op == "add":
                return sum(values), inputs[0][1]
            if len(values) != 2:
                raise MissingCapability("task", "binary helper requires two inputs")
            if op == "subtract":
                return values[0] - values[1], inputs[0][1]
            return (
                "less" if values[0] < values[1] else "greater" if values[0] > values[1] else "equal"
            ), None
        query_time = (
            self.query_time
            if self.query_time is not None
            else self.anchor_time(self.spec["time"]["query_anchor"], query=True)
        )
        if params.get("event_description"):
            query_time = self.event_time(params["event_description"], params.get("ordinal", 1))
        if inputs and op == "heading_delta":
            query_time = inputs[0][0]
        if self.query_scope and (
            query_time is None or not self.query_scope[0] <= query_time <= self.query_scope[1]
        ):
            raise MissingCapability(
                "time", "query window needs a resolved time/event anchor inside it"
            )
        if op == "next_action":
            progress_records = self.state.select("progress", time=query_time)
            if progress_records:
                progress = self.one("progress", time=query_time)
                if progress["value"]["stop_reached"]:
                    instruction = self.spec["route"]
                    if not instruction or not instruction["stop_condition"]:
                        raise MissingCapability(
                            "route", "stopping needs the question's stop condition"
                        )
                    for e in progress["arguments"]:
                        self.state.bound(e, query_time)
                    return "Stop", None
        for entity in entities:
            self.state.bound(entity, query_time)
        if op == "bearing":
            if len(entities) != 3:
                raise MissingCapability(
                    "task", "bearing needs origin, forward target and query target"
                )
            rule = self.spec["direction_rule"]
            records = self.state.select("bearing", entities, time=query_time, frame="query")
            if records:
                r = self.one("bearing", entities, time=query_time, frame="query")
                v = r["value"]
                if v["angle_interval"] is not None:
                    return direction.interval_bearing(
                        v["angle_interval"], rule["kind"], rule["back_threshold_degrees"]
                    ), None
                if v["horizontal"] in {"left", "right"}:
                    if rule["kind"] == "left_right":
                        return v["horizontal"], None
                    if rule["kind"] == "four_quadrants" and v["depth"] in {"front", "back"}:
                        return v["depth"] + "-" + v["horizontal"], None
                raise MissingCapability(
                    "reference_frame", "qualitative evidence does not resolve the requested sector"
                )
            points, fid = self.points(entities, time=query_time)
            if self.state.data["frames"][fid]["plane"] not in {"horizontal", "3d"}:
                raise MissingCapability(
                    "reference_frame", "bearing requires a horizontal scene plane"
                )
            label, angle = direction.bearing(
                *[p["value"]["coordinates"] for p in points],
                rule["kind"],
                rule["back_threshold_degrees"],
            )
            self.steps.append({"angle_degrees": angle, "reference_frame_id": fid})
            return label, None
        if op == "heading_delta":
            reference = self.anchor_time(self.spec["time"]["reference_anchor"])
            if reference is None or query_time is None:
                raise MissingCapability("time", "heading update needs initial and query anchors")
            initial = self.one("heading", entities, time=reference)
            current = self.one("heading", entities, time=query_time)
            if initial["reference_frame_id"] != current["reference_frame_id"]:
                raise MissingCapability(
                    "reference_frame", "shot-separated headings need an explicit common frame"
                )
            kind = self.spec["query_frame"]["heading_kind"]
            if initial["value"]["kind"] != kind or current["value"]["kind"] != kind:
                raise MissingCapability(
                    "reference_frame", "camera, body and motion headings cannot be substituted"
                )
            lo, hi = direction.heading_delta(
                initial["value"]["degrees"], current["value"]["degrees"]
            )
            if abs(lo - hi) > 1e-6:
                raise MissingCapability(
                    "reference_frame", "heading interval does not determine an exact angle"
                )
            return (
                "unchanged"
                if abs(lo) < 1e-9
                else ("right" if lo > 0 else "left") + f" {abs(lo):g} degrees"
            ), None
        if op in {"absolute_distance", "nearest_object"}:
            if len(entities) < 2:
                raise MissingCapability("task", "distance query needs an origin and candidates")
            if op == "absolute_distance":
                interval, r = self.distance_pair(entities[:2], query_time)
                unit = self.spec["measurement"]["unit"]
                converted = self.metric(interval, r, unit, entities[0])
                return self.scalar(converted), unit
            intervals, units = {}, set()
            frames = set()
            for e in entities[1:]:
                interval, r = self.distance_pair([entities[0], e], query_time)
                units.add(r["unit"])
                frames.add(r["reference_frame_id"])
                intervals[e] = interval
            if len(units) != 1 or len(frames) != 1:
                raise MissingCapability(
                    "reference_frame", "candidate distances need a common frame and unit"
                )
            winner = distance.nearest(intervals)
            return self.description(winner), None
        if op == "max_extent":
            r = self.one("extent", entities, time=query_time)
            if not r["value"]["complete"]:
                raise MissingCapability("boundary", "max extent requires full object boundaries")
            dims = r["value"]["dimensions"]
            if any(a < 0 or a > b for a, b in dims):
                raise MissingCapability("boundary", "invalid extent interval")
            interval = [max(v[0] for v in dims), max(v[1] for v in dims)]
            unit = self.spec["measurement"]["unit"]
            return self.scalar(self.metric(interval, r, unit, entities[0])), unit
        if op == "floor_area":
            r = self.one("area", entities, time=query_time)
            if self.state.data["frames"][r["reference_frame_id"]]["plane"] != "horizontal":
                raise MissingCapability(
                    "reference_frame", "floor area requires horizontal footprint geometry"
                )
            value = area.floor_area(r["value"]["regions"], r["value"]["coverage_complete"])
            unit = self.spec["measurement"]["unit"]
            return self.scalar(self.metric([value, value], r, unit, entities[0], power=2)), unit
        if op in {"fill_turns", "next_action", "compare_paths"}:
            instruction = self.spec["route"]
            if not instruction:
                raise MissingCapability("task", "navigation query lacks a route contract")
            edge_records = self.state.select("edge", time=query_time)
            if len({r["reference_frame_id"] for r in edge_records}) > 1:
                raise MissingCapability("reference_frame", "route headings use disconnected frames")
            edges = [self.take(r) for r in edge_records]
            if op == "compare_paths":
                path = route.compare_paths(instruction["candidate_paths"], edges)
                return [self.description(e) for e in path], None
            progress = self.one("progress", time=query_time)
            if (
                edge_records
                and progress["reference_frame_id"] != edge_records[0]["reference_frame_id"]
            ):
                raise MissingCapability(
                    "reference_frame", "arrival heading and route use different frames"
                )
            if op == "next_action":
                return route.next_action(progress["value"], instruction["waypoints"], edges), None
            start = instruction["start_entity"]
            if progress["value"]["current"] != start:
                raise MissingCapability("route", "route replay must begin at the specified origin")
            forward = instruction["forward_entity"]
            if forward:
                roles = {e["role"]: e["id"] for e in self.spec["entities"]}
                qframe = self.spec["query_frame"]
                if (
                    progress["reference_frame_id"] == "query"
                    and roles.get(qframe["origin_role"]) == start
                    and roles.get(qframe["forward_role"]) == forward
                ):
                    expected_heading = 0.0
                else:
                    points, fid = self.points([start, forward], time=query_time)
                    if fid != progress["reference_frame_id"]:
                        raise MissingCapability(
                            "reference_frame", "initial facing target uses another frame"
                        )
                    a, b = [p["value"]["coordinates"] for p in points]
                    if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-9:
                        raise MissingCapability(
                            "reference_frame", "initial facing target coincides with origin"
                        )
                    expected_heading = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
                heading = progress["value"]["heading"]
                if heading is None or abs(direction.wrap(heading - expected_heading)) > 1e-6:
                    raise MissingCapability(
                        "reference_frame",
                        "route progress contradicts question-defined initial facing",
                    )
            turns = route.replay(
                [start, *instruction["waypoints"]], progress["value"]["heading"], edges
            )
            indices = instruction.get("turn_indices", list(range(len(turns))))
            if any(i >= len(turns) for i in indices):
                raise MissingCapability("route", "turn blank index exceeds the supplied route")
            return [turns[i] for i in indices], None
        if op == "viewpoint_relation":
            r = self.one("viewpoint", entities, time=query_time)
            return viewpoint.viewpoint(r["value"], self.state.data["frames"]), None
        raise ProtocolError(f"unknown whitelisted operation: {op}")

    def scalar(self, interval):
        # Midpoint remains an estimate and interval is retained in the execution trace.
        self.estimated |= interval[0] != interval[1]
        self.steps.append(
            {"numeric_interval": list(interval), "output_policy": "midpoint_estimate"}
        )
        return sum(interval) / 2

    def description(self, entity):
        return next(e["description"] for e in self.spec["entities"] if e["id"] == entity)

    def root_entities(self):
        entities = self.spec["entities"]
        if self.spec["operation"] == "bearing":
            roles = self.spec["query_frame"]
            names = [roles["origin_role"], roles["forward_role"], "target"]
            return [
                next((e["id"] for e in entities if e["role"] == role), "missing") for role in names
            ]
        return [e["id"] for e in entities]

    def run(self):
        try:
            nodes = self.spec["nodes"]
            results = {}
            if nodes:
                for n in nodes:
                    results[n["id"]] = self.execute(
                        n["operation"],
                        n["entity_ids"],
                        [results[i] for i in n["input_nodes"]],
                        n["parameters"],
                    )
                    self.steps.append(
                        {
                            "node_id": n["id"],
                            "operation": n["operation"],
                            "value": results[n["id"]][0],
                            "unit": results[n["id"]][1],
                        }
                    )
                value, unit = results[self.spec["output_node"]]
            else:
                value, unit = self.execute(self.spec["operation"], self.root_entities())
            estimated = self.estimated or any(
                r["status"] != "evidence_supported"
                or r["method"]
                in {"category_prior", "multiview_visual_estimate", "predicted_geometry"}
                for r in self.used.values()
            )
            return QueryResult(
                "estimated" if estimated else "evidence_supported",
                value,
                unit,
                sorted({s for r in self.used.values() for s in r["source_observation_ids"]}),
                list(self.used),
                [],
                self.steps,
                sorted(set(self.scales)),
            )
        except MissingCapability as exc:
            if exc.gap.kind == "time" and self.query_scope:
                exc.gap.time_interval = list(self.query_scope)
            return QueryResult(gaps=[exc.gap], record_ids=list(self.used), derivation=self.steps)


def run_query(spec, state, contract, query_time=None, query_scope=None):
    return Executor(state, spec, contract, query_time, query_scope).run()
