"""Necessary evidence gates, not an oracle for natural-language option entailment."""

from .observation import ObservationSpec
from .types import ProtocolError


def assessment_policy(query, payload):
    operations = {}
    if query:
        slots = {s["id"]: s for s in query["slots"]}
        derived = {o["operation_id"]: o for o in payload["derived"].get("operations", [])}
        for operation in query["operations"]:
            result = derived.get(operation["id"], {})
            spec = ObservationSpec.from_query({**query, "operations": [operation]})
            evidence = []
            for row in payload["observations"]:
                if (
                    row["slot_id"] not in operation["slot_ids"]
                    or row.get("basis") != "visual_observation"
                ):
                    continue
                visible = row.get("visibility") == "visible"
                negative = (
                    row.get("visibility") == "absent"
                    and operation["op"] == "motion_condition_filter"
                )
                if not (visible or negative):
                    continue
                # Geometry requires measured fields, not a posture label in value.
                view = {**row, "frame_id": row["source_frame_id"], "slot": slots[row["slot_id"]]}
                if spec.record_missing(view, stored=True):
                    continue
                if row["id"] in result.get("evidence_ids", []):
                    evidence.append(row["id"])
            operations[operation["id"]] = {
                "status": result.get("status", "unresolved"),
                "evidence_ids": evidence,
                "minimum_distinct_times": 3
                if operation["op"] in {"direction_sequence", "path_shape"}
                else 2
                if operation["op"]
                in {
                    "endpoint_delta",
                    "direction_sequence",
                    "path_shape",
                    "rotation_pattern",
                    "relation_transition",
                    "motion_property_trend",
                    "periodic_continuation",
                }
                else 1,
            }
    return {
        "basis": "program_necessary_evidence_conditions",
        "operations": operations,
        "complete_support_allowed": bool(payload["derived"].get("sufficient")),
        "instruction": "Assess whole options, not isolated phrases. Unresolved operations require unknown; prediction may still be a best guess. Passing these checks does not prove option meaning.",
    }


def validate_assessment_evidence(value, payload):
    policy = payload.get("assessment_policy")
    if not policy:
        return
    supported = [a for a in value["assessments"] if a["status"] == "supported"]
    if len(supported) > 1:
        raise ProtocolError(
            "Single-choice final supports multiple whole options; reassess complete meanings, not subclaims"
        )
    if supported and supported[0]["label"] != value["prediction"]:
        raise ProtocolError("Prediction disagrees with the supported whole option")
    times = {r["id"]: r["timestamp"] for r in payload["observations"]}
    for assessment in value["assessments"]:
        if assessment["status"] == "unknown":
            continue
        if not policy["complete_support_allowed"]:
            raise ProtocolError(
                "Query evidence/identity/coverage is unresolved; use unknown for whole-option assessments"
            )
        ids = assessment.get("operation_ids", [])
        if not ids or len(ids) != len(set(ids)):
            raise ProtocolError(
                "Decisive assessments require unique operation_ids from assessment_policy"
            )
        covered = set()
        for oid in ids:
            operation = policy["operations"].get(oid)
            if not operation or operation["status"] != "supported":
                raise ProtocolError("Assessment cites an unresolved or unknown operation")
            refs = set(assessment["evidence_ids"]) & set(operation["evidence_ids"])
            if len({times[r] for r in refs}) < operation["minimum_distinct_times"]:
                raise ProtocolError("Assessment lacks ordered measurement evidence for " + oid)
            covered.update(refs)
        if set(assessment["evidence_ids"]) - covered:
            raise ProtocolError(
                "Assessment references are not measurements supporting its cited operations"
            )
