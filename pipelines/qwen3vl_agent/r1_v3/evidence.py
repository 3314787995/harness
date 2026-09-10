"""Evidence qualifications, modality diagnostics and stable field-binding review tasks."""

import hashlib
import json
from dataclasses import asdict

from qwen3vl_agent.r1_v3.types import QueryBindingTask, V3Packet


def refresh_binding_tasks(packet, query):
    filled = {q for f in packet.fact_eligibility if f.answer_eligible for q in f.query_fields}
    missing = {f.field_id for f in query.fields} - filled
    records = {r.record_id: r for r in packet.observations}
    for task in packet.query_binding_tasks:
        if not missing.intersection(task.field_ids):
            task.status, task.reason = "resolved", "fields_now_supported_by_new_evidence"
        elif task.attempted:
            visual = any(
                g["kind"] in {"detail", "temporal_context", "temporal_selection"}
                for rid in task.result_record_ids
                for g in records[rid].gaps
            )
            task.status = "visual_gap" if visual else "unresolved"
            task.reason = "explicit_visual_gap" if visual else "query_binding_unresolved"
        elif task.status != "blocked":
            task.status, task.reason = "pending", "query_binding_unresolved"

    if packet.target_binding == "confirmed" and missing:
        eligible_seeds = {
            f.fact_id
            for f in packet.fact_eligibility
            if f.source_valid
            and f.observation_clear
            and f.target_confirmed
            and not f.refuted
            and not f.query_fields
        }
        for record in packet.observations:
            # A review response is not itself new source evidence. Its facts are retained,
            # but cannot create another task merely by obtaining a fresh record/fact ID.
            if record.task_id or record.purpose == "query_binding":
                continue
            facts = [f for f in record.facts if f.fact_id in eligible_seeds]
            if not facts:
                continue
            sources = tuple(
                sorted(
                    {x for f in facts for x in f.original_frame_ids}
                    | {
                        record.batch.crops.get(x, {}).get("source_frame_id", x)
                        for x in record.target["source_frame_ids"]
                    }
                )
            )
            fields = tuple(sorted(missing))
            if any(
                set(sources) <= set(t.source_frame_ids) and set(fields) <= set(t.field_ids)
                for t in packet.query_binding_tasks
            ):
                continue
            identity = [packet.candidate_id, record.record_id, fields, sources]
            digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
            packet.query_binding_tasks.append(
                QueryBindingTask(
                    "query_binding_" + digest,
                    packet.candidate_id,
                    (record.record_id,),
                    tuple(f.fact_id for f in facts),
                    fields,
                    sources,
                )
            )
    packet.unresolved = [
        x for x in packet.unresolved if not x.startswith("query_binding_unresolved:")
    ]
    packet.unresolved.extend(
        f"query_binding_unresolved:{t.task_id}:{','.join(sorted(missing.intersection(t.field_ids)))}"
        for t in packet.query_binding_tasks
        if t.status != "resolved"
    )
    return [t for t in packet.query_binding_tasks if t.status == "pending" and not t.attempted]


def modality_state(s, bundle):
    packets = [p for b in s.bundles for p in b.packets if isinstance(p, V3Packet)]
    # The selected bundle is included for standalone audits and merged candidates.
    packets += [p for p in bundle.packets if isinstance(p, V3Packet) and p not in packets]
    records = {r.record_id: asdict(r) for p in packets for r in p.observations}
    # _finish freezes all observations before retiring candidates. Their readings remain
    # actual input history even when their facts cannot compete in the answer bundle.
    records.update(
        (r["record_id"], r) for r in getattr(s, "trace", {}).get("observation_records", [])
    )
    shown = {f["id"] for r in records.values() for f in r["batch"]["frames"] if r["call_id"]}
    shown.update(getattr(getattr(s, "context", None), "shown_frame_ids", ()))
    shown = sorted(shown)
    raw = [f for r in records.values() for f in r["facts"]]
    eligible = {
        f.fact_id
        for p in bundle.packets
        if isinstance(p, V3Packet)
        for f in p.fact_eligibility
        if f.answer_eligible
    }
    result = {}
    for modality in dict.fromkeys(("video", "screen_text", *s.query.required_modalities)):
        kinds = {"visual", "screen_text"} if modality == "video" else {modality}
        facts = [f for f in raw if f["source_kind"] in kinds]
        readable = [
            f["fact_id"]
            for f in facts
            if f["observation_status"] == "clear" and not f["uncertain_characters"]
        ]
        result[modality] = {
            "allowed": modality in s.request.available_modalities,
            "input_present": bool(shown) if modality in {"video", "screen_text"} else False,
            "shown_source_frame_ids": shown if modality in {"video", "screen_text"} else [],
            "reading_status": "readable"
            if readable
            else "insufficient"
            if facts
            else "not_reported",
            "readable_fact_ids": readable,
            "answer_fact_ids": sorted(eligible.intersection(f["fact_id"] for f in facts)),
            "unassociated_fact_ids": [
                f["fact_id"] for f in facts if not f["supports_query_fields"]
            ],
        }
    return result
