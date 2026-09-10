"""Narrow shared-CLI adapter; unrelated strategy flags are rejected."""

import json
from dataclasses import replace
from pathlib import Path

from qwen3vl_agent.factory import build_model

from .controller import R6VideoAgent
from .evaluate import model_and_config, write_json
from .types import R6Request


def run_generic(args):
    if (
        len(args.video) != 1
        or args.image
        or args.output_protocol not in {"auto", "multiple_choice"}
    ):
        raise SystemExit("R6 requires one video and original multiple-choice options")
    if any(
        (
            args.r7_protocol,
            args.r8_protocol,
            args.r7_mode != "B4",
            args.r8_mode != "G",
            args.r9_mode != "B4",
            args.r9_unit,
            args.r9_allow_unresolved,
            args.facts_input,
            args.variables_input,
            args.r3_policy,
            args.r4_policy,
            args.diagnostic,
            args.require_choice,
            args.replay_output,
            args.group_id,
            args.query_time,
            args.protocol_id != "full_video",
        )
    ):
        raise SystemExit("incompatible non-R6 flags")
    provided = sum(
        bool(v) for v in (args.allowed_interval, args.allowed_scope, args.given_interval)
    )
    if provided > 1:
        raise SystemExit("choose one allowed interval convention")
    spans = args.allowed_interval or (
        [args.allowed_scope or args.given_interval] if provided else []
    )
    modalities = ["video"] + [k for k in ("subtitle", "asr") if getattr(args, k)]
    protocol = (
        json.loads(Path(args.r6_protocol).read_text(encoding="utf-8-sig"))
        if args.r6_protocol
        else {}
    )
    allowed = {
        "allowed_intervals",
        "reference_scope",
        "history_cutoff",
        "allowed_modalities",
        "subtitle_policy",
        "cross_question_cache_policy",
        "answer_protocol",
    }
    if not isinstance(protocol, dict) or set(protocol) - allowed:
        raise SystemExit("R6 protocol accepts only public scope/modality/output fields")
    request = R6Request(
        args.video[0],
        args.query,
        args.choice,
        request_id=args.request_id or "r6-request",
        video_id=args.video_id or "video",
        subtitle_path=args.subtitle,
        asr_path=args.asr,
        mode=args.r6_mode,
        subtype=args.execution_subtype or "auto",
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        **(
            {
                "allowed_intervals": spans,
                "reference_scope": args.query_scope,
                "history_cutoff": args.observation_cutoff,
                "allowed_modalities": modalities,
                "subtitle_policy": "aligned_only" if len(modalities) > 1 else "disabled",
            }
            | protocol
        ),
    )
    settings, config = model_and_config(args.config)
    if args.max_model_calls is not None:
        if args.max_model_calls > config.max_model_calls_total:
            raise SystemExit("CLI may only lower the configured call cap")
        config = replace(config, max_model_calls_total=args.max_model_calls)
    agent = R6VideoAgent(build_model(settings), config)
    agent.load()
    try:
        result = agent.solve(request)
    finally:
        agent.unload()
    print(result.text)
    if args.show_metadata:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.trace_output:
        write_json(args.trace_output, result.to_dict())
