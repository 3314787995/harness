"""Transactional, source-linked observations and spatial records."""

from copy import deepcopy

from .schema import SCHEMAS, validate
from .types import MissingCapability, ProtocolError, digest, plain


class SpatialState:
    def __init__(self, spec, data=None):
        self.spec = spec
        self.data = (
            deepcopy(data)
            if data is not None
            else {
                "state_version": 0,
                "entities": deepcopy(spec["entities"]),
                "observations": {},
                "entity_links": {},
                "link_history": [],
                "frames": {
                    "query": {
                        "id": "query",
                        "kind": spec["query_frame"]["kind"],
                        "plane": spec["query_frame"]["plane"],
                        "component_id": "query",
                        "parent_frame_id": None,
                        "source_observation_ids": [],
                    }
                },
                "records": {},
                "coverage": [],
                "unresolved_links": [],
                "events": [],
                "audit_verdicts": {},
                "invalid_records": [],
            }
        )

    def _commit(self, draft):
        draft["entities"] = deepcopy(self.spec["entities"])
        draft["scene_components"] = sorted({f["component_id"] for f in draft["frames"].values()})
        for name, predicates in {
            "local_geometry": {"point", "boundary", "alignment", "extent", "area"},
            "topology": {"edge", "progress"},
            "scale_records": {"scale"},
            "relations": {"bearing", "heading", "distance", "viewpoint"},
        }.items():
            draft[name] = [
                i
                for i, r in draft["records"].items()
                if r["predicate"] in predicates and i not in draft["invalid_records"]
            ]
        draft["state_version"] += 1
        self.data = plain(draft)

    def observe(self, value, evidence, namespace, max_hypotheses=3):
        validate(value, SCHEMAS["observe"])
        draft = deepcopy(self.data)
        observations = draft["observations"]
        ids = [o["id"] for o in value["observations"]]
        if len(ids) != len(set(ids)):
            raise ProtocolError("duplicate observation ID")
        obs_ids = {i: f"{namespace}.{i}" for i in ids}
        candidates = {
            o["entity_candidate"]: f"{namespace}.{o['entity_candidate']}"
            for o in value["observations"]
        }
        for raw in value["observations"]:
            source = evidence.get(raw["frame_id"])
            if not source:
                raise ProtocolError("observation cites a frame not presented in this call")
            item = deepcopy(raw)
            item["id"] = obs_ids[raw["id"]]
            item["entity_candidate"] = candidates[raw["entity_candidate"]]
            item["frame_id"] = source["id"]
            item["source_frame_id"] = source["source_frame_id"]
            item["timestamp_s"] = source["timestamp_seconds"]
            item["source_observation_call"] = namespace
            item["shot_id"] = f"{namespace}.{raw['shot_id']}"
            box = item["bbox_xyxy"]
            if box and not (box[0] < box[2] and box[1] < box[3]):
                raise ProtocolError("invalid observation box")
            if box and item["coordinate_space"] == "presented_image_normalized":
                x0, y0, x1, y1 = source["view_box"]
                width, height = source["source_size"]
                item["bbox_xyxy"] = [
                    (x0 + box[0] * (x1 - x0)) / width,
                    (y0 + box[1] * (y1 - y0)) / height,
                    (x0 + box[2] * (x1 - x0)) / width,
                    (y0 + box[3] * (y1 - y0)) / height,
                ]
            item["coordinate_space"] = "original_image_normalized"
            item["crop_to_original_transform"] = source["view_box"]
            if item["id"] in observations and observations[item["id"]] != item:
                raise ProtocolError("observation IDs are immutable")
            observations[item["id"]] = item
        entity_ids = {e["id"] for e in self.spec["entities"]}
        for raw in value["links"]:
            link = deepcopy(raw)
            entity = link["entity_id"]
            if entity not in entity_ids:
                raise ProtocolError("binding references unknown query entity")
            link["source_observation_ids"] = [
                obs_ids.get(i, i) for i in link["source_observation_ids"]
            ]
            link["candidate_ids"] = [candidates.get(i, i) for i in link["candidate_ids"]]
            self._sources(link, observations)
            supported = {
                observations[i]["entity_candidate"] for i in link["source_observation_ids"]
            }
            if set(link["candidate_ids"]) - supported:
                raise ProtocolError("binding candidate has no cited observation")
            if link["status"] == "confirmed" and (
                link["basis"] in {"appearance_only", "none"}
                or link["basis"] in {"continuous", "overlap"}
                and len(set(link["source_observation_ids"])) < 2
            ):
                link["status"] = "tentative"
            old = draft["entity_links"].get(entity, [])
            if link not in old:
                draft["link_history"].append(deepcopy(link))
                for previous in old:
                    if set(previous["candidate_ids"]) & set(link["candidate_ids"]):
                        if (
                            link["status"] in {"distinct", "unresolved", "tentative"}
                            and previous["status"] == "confirmed"
                        ):
                            previous["status"] = link["status"]
                            self._invalidate_entity(draft, entity)
                        elif link["status"] == "confirmed" and previous["status"] == "tentative":
                            previous["status"] = "confirmed"
                # Confirmations of new temporal observations do not silently replace identities.
                if link["status"] == "confirmed":
                    conflicting = [
                        p
                        for p in old
                        if p["status"] == "confirmed"
                        and _overlap(p["valid_time"], link["valid_time"])
                        and not set(p["candidate_ids"]) & set(link["candidate_ids"])
                    ]
                    if conflicting:
                        link["status"] = "tentative"
                        for p in conflicting:
                            p["status"] = "tentative"
                        self._invalidate_entity(draft, entity)
                revised = [*old, link]
                active = [
                    p
                    for p in revised
                    if p["status"] not in {"distinct", "unresolved"}
                    and _overlap(p["valid_time"], link["valid_time"])
                ]
                hypotheses = []
                for p in active:
                    group = set(p["candidate_ids"])
                    overlaps = [h for h in hypotheses if h & group]
                    for h in overlaps:
                        group |= h
                        hypotheses.remove(h)
                    hypotheses.append(group)
                if len(hypotheses) > max_hypotheses:
                    draft["unresolved_links"].append(
                        {"entity_id": entity, "reason": "binding_hypothesis_limit"}
                    )
                    for p in revised:
                        if p["status"] == "confirmed":
                            p["status"] = "tentative"
                    self._invalidate_entity(draft, entity)
                    retained = active[-max_hypotheses:]
                    revised = [p for p in revised if p not in active] + retained
                elif link["status"] == "confirmed":
                    draft["unresolved_links"] = [
                        p for p in draft["unresolved_links"] if p["entity_id"] != entity
                    ]
                draft["entity_links"][entity] = revised
        self._commit(draft)
        return list(obs_ids.values())

    @staticmethod
    def _sources(record, observations):
        sources = record["source_observation_ids"]
        if not sources or set(sources) - set(observations):
            raise ProtocolError("unknown/empty observation references")
        a, b = record["valid_time"]
        if a > b or any(not a <= observations[i]["timestamp_s"] <= b for i in sources):
            raise ProtocolError("record valid time excludes its supporting observations")

    @staticmethod
    def _invalidate_entity(draft, entity):
        changed = {k for k, r in draft["records"].items() if entity in r["arguments"]}
        while True:
            more = {k for k, r in draft["records"].items() if set(r["parent_record_ids"]) & changed}
            if more <= changed:
                break
            changed |= more
        draft["invalid_records"] = sorted(set(draft["invalid_records"]) | changed)

    def relations(self, value, presented_frame_ids, namespace):
        validate(value, SCHEMAS["relations"])
        draft = deepcopy(self.data)
        observations = draft["observations"]
        records = draft["records"]
        frame_map = {
            f["id"]: (f["id"] if f["id"] in draft["frames"] else f"{namespace}.{f['id']}")
            for f in value["frames"]
        }
        record_map = {r["record_id"]: f"{namespace}.{r['record_id']}" for r in value["records"]}
        if len(record_map) != len(value["records"]) or len(frame_map) != len(value["frames"]):
            raise ProtocolError("duplicate relation/frame IDs")
        for f in value["frames"]:
            f = deepcopy(f)
            f["id"] = frame_map[f["id"]]
            if set(f["source_observation_ids"]) - set(observations):
                raise ProtocolError("frame cites unknown observations")
            if f["id"] not in draft["frames"] and not f["source_observation_ids"]:
                raise ProtocolError("new reference frame needs visual sources")
            if f["parent_frame_id"]:
                f["parent_frame_id"] = frame_map.get(f["parent_frame_id"], f["parent_frame_id"])
                if f["parent_frame_id"] not in draft["frames"]:
                    raise ProtocolError("unknown parent reference frame")
            if f["id"] in draft["frames"] and draft["frames"][f["id"]] != f:
                raise ProtocolError("reference frames are immutable")
            draft["frames"][f["id"]] = f
        entity_ids = {e["id"] for e in self.spec["entities"]}
        for raw in value["records"]:
            r = deepcopy(raw)
            r["record_id"] = record_map[r["record_id"]]
            r["reference_frame_id"] = frame_map.get(
                r["reference_frame_id"], r["reference_frame_id"]
            )
            for key in (
                "camera_frame",
                "observer_frame",
                "mirror_plane_id",
                "from_frame",
                "to_frame",
            ):
                if r["value"].get(key):
                    r["value"][key] = frame_map.get(r["value"][key], r["value"][key])
            r["parent_record_ids"] = [record_map.get(i, i) for i in r["parent_record_ids"]]
            if set(r["arguments"]) - entity_ids or r["reference_frame_id"] not in draft["frames"]:
                raise ProtocolError("unknown entity or reference frame")
            if set(r["parent_record_ids"]) - set(records):
                raise ProtocolError("parent records must already exist (acyclic dependency)")
            self._sources(r, observations)
            if any(
                observations[i]["frame_id"] not in presented_frame_ids
                for i in r["source_observation_ids"]
            ):
                raise ProtocolError("relation source was not visually presented to the builder")
            if r["method"] not in {
                "direct_observation",
                "multiview_visual_estimate",
                "category_prior",
            }:
                raise ProtocolError(
                    "model cannot create question constraints, tool results or deterministic derivations"
                )
            # World coordinates/metric magnitudes generated by Qwen are estimates, not measurements.
            if r["method"] != "direct_observation" or r["predicate"] in {
                "point",
                "boundary",
                "heading",
                "distance",
                "extent",
                "area",
                "scale",
                "alignment",
                "viewpoint",
            }:
                r["status"] = "estimated" if r["status"] != "unresolved" else "unresolved"
            if (
                r["predicate"] == "bearing"
                and draft["frames"][r["reference_frame_id"]]["kind"] == "query"
            ):
                r["status"] = "estimated" if r["status"] != "unresolved" else "unresolved"
            if r["predicate"] == "edge":
                r["status"] = "estimated" if r["status"] != "unresolved" else "unresolved"
            if any(
                draft["records"][i]["status"] != "evidence_supported"
                for i in r["parent_record_ids"]
            ):
                r["status"] = "estimated" if r["status"] != "unresolved" else "unresolved"
            self._validate_value(r, draft)
            if r["record_id"] in records and records[r["record_id"]] != r:
                raise ProtocolError("record IDs are immutable")
            records[r["record_id"]] = r
            if r["predicate"] == "event" and r["record_id"] not in draft["events"]:
                draft["events"].append(r["record_id"])
        self._commit(draft)
        return list(record_map.values())

    @staticmethod
    def _validate_value(r, draft):
        p, v = r["predicate"], r["value"]
        frame = draft["frames"][r["reference_frame_id"]]
        known_entities = {e["id"] for e in draft["entities"]}
        supported_candidates = {
            draft["observations"][i]["entity_candidate"] for i in r["source_observation_ids"]
        }
        referenced = [
            v[k]
            for k in ("reference_entity", "position_entity", "from", "to", "current", "previous")
            if v.get(k)
        ]
        referenced += v.get("completed_waypoints", [])
        if set(referenced) - known_entities:
            raise ProtocolError("spatial value references an unknown entity")
        if set(v.get("landmark_ids", [])) - (known_entities | supported_candidates):
            raise ProtocolError("landmark must be a task entity or a cited observed candidate")
        if p == "scale" and v["kind"] == "frozen_prediction":
            raise ProtocolError(
                "a disabled geometry backend cannot supply frozen scale predictions"
            )
        if p in {"point", "boundary"}:
            if v["space"] == "scene" and frame["kind"] == "image":
                raise ProtocolError("ImagePoint cannot become ScenePoint")
            points = [v["coordinates"]] if p == "point" else v["points"]
            if len({len(x) for x in points}) != 1:
                raise ProtocolError("mixed geometry dimensions")
        if p in {"distance", "scale", "heading"}:
            interval = v.get("interval", v.get("meters_per_scene_unit", v.get("degrees")))
            if interval[0] > interval[1] or (p != "heading" and interval[0] < 0):
                raise ProtocolError("invalid numeric interval")
        if (
            p == "event"
            and not r["valid_time"][0] <= v["start_s"] <= v["end_s"] <= r["valid_time"][1]
        ):
            raise ProtocolError("event outside sourced interval")
        for key in ("camera_frame", "observer_frame", "mirror_plane_id", "from_frame", "to_frame"):
            if v.get(key) and v[key] not in draft["frames"]:
                raise ProtocolError("value references unknown frame")

    def invalidate(self, record_ids, verdict="contradicted"):
        draft = deepcopy(self.data)
        pending = set(record_ids)
        if pending - set(draft["records"]):
            raise ProtocolError("audit references unknown records")
        while True:
            more = {k for k, r in draft["records"].items() if set(r["parent_record_ids"]) & pending}
            if more <= pending:
                break
            pending |= more
        draft["invalid_records"] = sorted(set(draft["invalid_records"]) | pending)
        for i in record_ids:
            draft["audit_verdicts"][i] = verdict
        self._commit(draft)

    def bound(self, entity, time=None):
        links = [
            p
            for p in self.data["entity_links"].get(entity, [])
            if p["status"] == "confirmed"
            and (time is None or p["valid_time"][0] <= time <= p["valid_time"][1])
        ]
        if not links or any(p["entity_id"] == entity for p in self.data["unresolved_links"]):
            raise MissingCapability(
                "identity", f"unresolved instance binding: {entity}", entities=[entity]
            )
        if any(
            p["status"] == "tentative"
            and (time is None or p["valid_time"][0] <= time <= p["valid_time"][1])
            for p in self.data["entity_links"].get(entity, [])
        ):
            raise MissingCapability(
                "identity", f"ambiguous instance binding: {entity}", entities=[entity]
            )
        return links

    def select(self, predicate, arguments=None, *, time=None, frame=None):
        return [
            r
            for rid, r in self.data["records"].items()
            if rid not in self.data["invalid_records"]
            and r["status"] != "unresolved"
            and r["predicate"] == predicate
            and (arguments is None or r["arguments"] == list(arguments))
            and (time is None or r["valid_time"][0] <= time <= r["valid_time"][1])
            and (frame is None or r["reference_frame_id"] == frame)
        ]

    def one(self, predicate, arguments=None, **kwargs):
        records = self.select(predicate, arguments, **kwargs)
        if not records:
            raise MissingCapability(
                "relation", f"missing {predicate} for {arguments}", entities=arguments or []
            )
        signatures = {
            digest({k: r[k] for k in ("value", "unit", "reference_frame_id")}) for r in records
        }
        if len(signatures) != 1:
            raise MissingCapability(
                "relation",
                f"conflicting {predicate} records",
                record_ids=[r["record_id"] for r in records],
            )
        return records[-1]

    def context(self, observation_limit=24):
        # Persistent evidence is never truncated; only the role's working view is bounded.
        return {
            "state_version": self.data["state_version"],
            "entities": self.spec["entities"],
            "observations": list(self.data["observations"].values())[-observation_limit:],
            "entity_links": self.data["entity_links"],
            "frames": list(self.data["frames"].values()),
            "records": [
                r for k, r in self.data["records"].items() if k not in self.data["invalid_records"]
            ][-24:],
        }

    def binding_gaps(self):
        gaps = []
        for e in self.spec["entities"]:
            try:
                self.bound(e["id"])
            except MissingCapability as exc:
                gaps.append(exc.gap)
        return gaps


def _overlap(a, b):
    return max(a[0], b[0]) <= min(a[1], b[1])
