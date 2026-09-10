"""Deterministic checks and narrow raw-frame visual audit validation."""

from .types import Gap, ProtocolError, Verification


def deterministic_audit(spec, state, result):
    gaps = list(result.gaps)
    for rid in result.record_ids:
        record = state.data["records"].get(rid)
        if not record or rid in state.data["invalid_records"]:
            gaps.append(Gap("relation", "query depends on invalid record", record_ids=[rid]))
    if result.value is None:
        return Verification(False, False, gaps=gaps)
    if not result.source_ids:
        gaps.append(Gap("relation", "spatial answer has no visual observation sources"))
    missing = [r for r in result.record_ids if state.data["audit_verdicts"].get(r) != "supported"]
    if missing:
        gaps.append(
            Gap(
                "relation", "decisive spatial claims still need raw-frame audit", record_ids=missing
            )
        )
    return Verification(not gaps, not result.gaps, gaps=gaps)


def validate_visual_audit(value, requested, evidence):
    ids = [c["record_id"] for c in value["checks"]]
    if len(ids) != len(set(ids)) or set(ids) != set(requested):
        raise ProtocolError("audit must check every requested atomic claim exactly once")
    for check in value["checks"]:
        if set(check["frame_ids"]) - set(evidence):
            raise ProtocolError("audit cites an unpresented frame")
    return value
