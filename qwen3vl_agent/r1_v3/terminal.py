"""Frozen-evidence terminal audit and at most one conditional, image-backed recovery."""

import copy
import re
from dataclasses import asdict

from qwen3vl_agent.r1.control import BudgetExhausted, ProtocolError
from qwen3vl_agent.r1_v3.types import TerminalAudit


def audit_terminal(
    payload,
    data,
    *,
    raw="",
    attempt=1,
    call_ids=(),
    limited=False,
    truncated=False,
    parser=None,
    error=None,
):
    checks = {
        key: True
        for key in (
            "json",
            "fact_references",
            "frozen_claims",
            "option_coverage",
            "exclusion_basis",
            "support_consistency",
            "model_certainty",
            "media_quality",
            "frozen_evidence",
            "no_competing_candidates",
        )
    }
    failures, output_errors = [], []

    def fail(check, reason, *, repairable=True):
        checks[check] = False
        failures.append(reason)
        if repairable:
            output_errors.append(reason)

    if not payload["evidence_audit"]["sufficient"]:
        fail("frozen_evidence", "terminal_frozen_evidence_insufficient", repairable=False)
    if payload.get("competing_bundles_unresolved"):
        fail(
            "no_competing_candidates", "terminal_competing_candidates_unresolved", repairable=False
        )
    if limited:
        fail("media_quality", "terminal_media_quality_limited", repairable=False)
    if truncated:
        fail("json", "terminal_output_truncated")
    parsed = None
    if not isinstance(data, dict):
        fail(
            "json",
            "terminal_json_unparseable"
            if isinstance(error, ProtocolError)
            else "terminal_call_failed:" + str(error),
            repairable=isinstance(error, ProtocolError),
        )
    else:
        frozen = payload["frozen_bundle"]
        facts = {f["fact_id"]: f for p in frozen["packets"] for f in p["facts"]}
        usable = {
            fid
            for fid, f in facts.items()
            if f["observation_status"] == "clear"
            and not f.get("uncertain_characters")
            and fid not in frozen["refuted_fact_ids"]
        }

        def references(value, label, *, required=True):
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                fail("fact_references", "terminal_references_invalid:" + label)
                return []
            if required and not value:
                fail("fact_references", "terminal_references_empty:" + label)
            if set(value) - facts.keys():
                fail("fact_references", "terminal_references_unfrozen:" + label)
            if (set(value) & facts.keys()) - usable:
                fail("fact_references", "terminal_references_ineligible:" + label)
            return value

        references(data.get("evidence_fact_ids"), "answer")
        claims = data.get("claims")
        if not isinstance(claims, list) or not claims:
            fail("frozen_claims", "terminal_claims_missing_or_invalid")
        else:
            for i, claim in enumerate(claims):
                if not isinstance(claim, dict):
                    fail("frozen_claims", f"terminal_claim_invalid:{i}")
                    continue
                ids = references(claim.get("fact_ids"), f"claim:{i}")
                if not isinstance(claim.get("statement"), str) or claim["statement"] not in {
                    facts[r]["statement"] for r in ids if r in facts
                }:
                    fail("frozen_claims", f"terminal_claim_outside_frozen_facts:{i}")

        choices = {c["label"] for c in payload["choices"]}
        assessments = data.get("choice_assessments")
        mapped = {}
        if not isinstance(assessments, list):
            fail("option_coverage", "terminal_choice_assessments_invalid")
        else:
            for item in assessments:
                if not isinstance(item, dict) or not isinstance(item.get("label"), str):
                    fail("option_coverage", "terminal_choice_assessment_invalid")
                    continue
                label = item["label"]
                if label not in choices or label in mapped:
                    fail("option_coverage", "terminal_choice_duplicate_or_unknown:" + label)
                mapped[label] = item
                status = item.get("status")
                if status not in {"supported", "rejected", "unresolved"}:
                    fail("support_consistency", "terminal_choice_status_invalid:" + label)
                references(
                    item.get("fact_ids"),
                    "choice:" + label,
                    required=status in {"supported", "rejected"},
                )
                if status in {"supported", "rejected"} and (
                    not isinstance(item.get("basis"), str) or not item["basis"].strip()
                ):
                    fail("exclusion_basis", "terminal_choice_basis_missing:" + label)
                if status == "unresolved":
                    fail("model_certainty", "terminal_choice_unresolved:" + label, repairable=False)
        for label in sorted(choices - mapped.keys()):
            fail("option_coverage", "terminal_choice_missing:" + label)
        prediction = data.get("prediction")
        statuses_complete = not choices or (
            set(mapped) == choices
            and all(
                item.get("status") == ("supported" if label == prediction else "rejected")
                for label, item in mapped.items()
            )
        )
        judged = data.get("answer_supported")
        if not isinstance(judged, bool):
            fail("support_consistency", "terminal_answer_supported_invalid")
        if judged is not True:
            fail("model_certainty", "terminal_model_uncertainty", repairable=False)
        if choices and (
            judged is True and not statuses_complete or judged is False and statuses_complete
        ):
            fail("support_consistency", "terminal_support_assessments_contradict")
        excluded = data.get("alternatives_excluded")
        if choices and (not isinstance(excluded, bool) or excluded != statuses_complete):
            fail("support_consistency", "terminal_alternative_flag_contradicts_assessments")
        if parser:
            try:
                parsed = parser(data, limited)
            except (ProtocolError, KeyError, TypeError, ValueError) as exc:
                fail("json", "terminal_public_protocol:" + str(exc))
    audit = TerminalAudit(
        attempt,
        list(call_ids),
        raw,
        checks,
        list(dict.fromkeys(failures)),
        list(dict.fromkeys(output_errors)),
        all(checks.values()),
    )
    if parsed is not None:
        parsed["supported"] = bool(parsed["supported"] and audit.passed)
    return audit, parsed


def _media_recovery_reason(agent, s, batch):
    if batch is None or not batch.frames:
        return "terminal_recovery_original_media_unavailable"
    try:
        prepared = agent.media.prepare(batch)
    except (OSError, ValueError, BudgetExhausted) as exc:
        return "terminal_recovery_media_unavailable:" + str(exc)
    ctx, budget = s.context, s.request.budget
    if (
        ctx.frame_exposures + len(prepared.frames) > budget.max_frame_exposures
        or ctx.media_pixels + prepared.pixels > budget.max_media_pixels
        or budget.max_visual_tokens is not None
        and ctx.visual_tokens_estimated + (prepared.pixels + 1023) // 1024
        > budget.max_visual_tokens
    ):
        return "terminal_recovery_media_budget"
    return None


def _fallback(payload, data, raw, previous):
    """Preserve a usable prediction, never choose the more favourable support assessment."""
    labels = {c["label"] for c in payload["choices"]}
    value = data.get("prediction") if isinstance(data, dict) else None
    if labels and (not isinstance(value, str) or value not in labels):
        value = next(
            (
                label
                for label in sorted(labels)
                if raw.strip().strip('`"') == label
                or re.search(r'"prediction"\s*:\s*"' + re.escape(label) + '"', raw)
            ),
            None,
        )
    if not isinstance(value, str) or not value.strip() or labels and value not in labels:
        value = previous["prediction"] if previous else None
    if payload["output_protocol"] == "numeric":
        valid = {
            f["structured_value"].strip()
            for p in payload["frozen_bundle"]["packets"]
            for f in p["facts"]
            if f["observation_status"] == "clear"
            and not f.get("uncertain_characters")
            and f["fact_id"] not in payload["frozen_bundle"]["refuted_fact_ids"]
        }
        if value not in valid or not re.search(r"\d", value or ""):
            value = previous["prediction"] if previous else None
    if value is None:
        raise ProtocolError("terminal_no_usable_prediction")
    return {"prediction": value, "claims": [], "choice_assessments": [], "supported": False}


def terminal_response(agent, s, payload, batch, parser):
    frozen_payload = copy.deepcopy(payload)
    attempts = s.trace.setdefault("terminal_audits", [])
    previous, current_payload = None, frozen_payload
    for attempt in (1, 2):
        role = "final" if attempt == 1 else "terminal_review"
        before = len(s.context.calls)
        data, raw, error, limited = None, "", None, False
        try:
            response = s.session.call(role, current_payload, batch=batch, terminal=True)
            data, raw = response.value, response.raw
            limited = bool(response.prepared and response.prepared.quality_limited)
        except (ProtocolError, RuntimeError, OSError, TypeError, ValueError) as exc:
            error = exc
        calls = s.context.calls[before:]
        if calls:
            raw = calls[-1].get("raw_response", raw)
            limited = limited or calls[-1].get("quality_limited", False)
        truncated = bool(
            calls
            and calls[-1].get("metadata", {}).get("output_tokens", 0) >= agent.config.final_tokens
        )
        audit, parsed = audit_terminal(
            frozen_payload,
            data,
            raw=raw,
            attempt=attempt,
            call_ids=[c["call_id"] for c in calls],
            limited=limited,
            truncated=truncated,
            parser=parser,
            error=error,
        )
        if audit.passed and parsed is not None:
            attempts.append(asdict(audit))
            s.trace["terminal_call_count"] = len(s.session.terminal_calls)
            return parsed
        reason = None
        if attempt == 2:
            reason = "terminal_recovery_failed"
        elif not audit.checks["frozen_evidence"] or not audit.checks["no_competing_candidates"]:
            reason = "terminal_recovery_evidence_not_ready"
        elif any(
            x in s.context.issues
            for x in (
                "query_compiler_unresolved",
                "discriminant_compiler_unresolved",
                "terminal_media_unavailable",
            )
        ) or agent._missing_modalities(
            s, next(b for b in s.bundles if b.bundle_id == payload["frozen_bundle"]["bundle_id"])
        ):
            reason = "terminal_recovery_input_unresolved"
        elif not audit.output_errors:
            reason = "terminal_recovery_no_correctable_output_error"
        elif len(s.session.terminal_calls) >= 2:
            reason = "terminal_call_budget"
        elif len(s.context.calls) >= s.request.budget.max_model_calls:
            reason = "model_call_budget"
        else:
            reason = _media_recovery_reason(agent, s, batch)
        if reason:
            audit.recovery = reason
            attempts.append(asdict(audit))
            s.context.issues.extend([*audit.failures, reason])
            s.trace["terminal_call_count"] = len(s.session.terminal_calls)
            result = (
                parsed if parsed is not None else _fallback(frozen_payload, data, raw, previous)
            )
            result["supported"] = False
            return result
        audit.recovery = "requested_output_recovery"
        attempts.append(asdict(audit))
        previous = parsed if parsed is not None else previous
        current_payload = {
            **copy.deepcopy(frozen_payload),
            "terminal_review": {
                "first_output": raw,
                "failed_checks": audit.output_errors,
                "instruction": "Re-evaluate all original choices against this same frozen evidence.",
            },
        }
    raise AssertionError("terminal loop exceeded its fixed bound")
