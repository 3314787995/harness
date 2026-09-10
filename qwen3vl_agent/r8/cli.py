"""Narrow adapter from the existing shared --strategy CLI to the public R8 request."""

import json
from pathlib import Path

from qwen3vl_agent.factory import build_model

from .controller import R8VideoAgent
from .evaluate import model_and_config, write_json
from .types import R8Request


def run_generic(args):
    if len(args.video) != 1 or args.image:
        raise SystemExit("R8 requires exactly one --video and no --image")
    if any(
        (
            args.asr,
            args.r7_protocol,
            args.r7_mode != "B4",
            args.facts_input,
            args.r3_policy,
            args.r4_policy,
            args.replay_output,
            args.force_choice,
        )
    ):
        raise SystemExit("incompatible non-R8 arguments; use --require-choice for R8 fallback")
    values = {
        "allowed_scope": args.allowed_scope or args.given_interval,
        "allowed_time_intervals": args.allowed_interval or (),
        "query_scope": args.query_scope,
        "query_time": args.query_time,
        "observation_cutoff": args.observation_cutoff,
        "protocol_id": args.protocol_id,
        "available_modalities": ("video", "screen_text", "subtitle")
        if args.subtitle
        else ("video", "screen_text"),
    }
    if args.r8_protocol:
        protocol = json.loads(Path(args.r8_protocol).read_text(encoding="utf-8"))
        allowed = {
            "allowed_scope",
            "allowed_time_intervals",
            "observation_cutoff",
            "protocol_id",
            "protocol_source",
            "available_modalities",
        }
        if not isinstance(protocol, dict) or set(protocol) - allowed:
            raise SystemExit("R8 protocol accepts only public media permission fields")
        for key, value in protocol.items():
            old = values.get(key)
            supplied = (
                key in {"allowed_scope", "allowed_time_intervals", "observation_cutoff"}
                and old is not None
                and old != ()
            )
            if supplied and json.loads(json.dumps(old)) != value:
                raise SystemExit("R8 protocol conflicts with explicit CLI permission")
        values.update(protocol)
    request = R8Request(
        args.video[0],
        args.query,
        choices=args.choice,
        **values,
        subtitle_path=args.subtitle,
        execution_subtype=args.execution_subtype,
        output_protocol=args.output_protocol,
        require_choice=args.require_choice,
        max_model_calls=args.max_model_calls,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        request_id=args.request_id or "r8-request",
        video_id=args.video_id or "video",
        group_id=args.group_id,
        mode=args.r8_mode,
        variables_input=args.variables_input,
        diagnostic=args.diagnostic,
    )
    model_config, config = model_and_config(args.config)
    agent = R8VideoAgent(build_model(model_config), config)
    agent.load()
    try:
        result = agent.solve(request)
    finally:
        agent.unload()
    print(result.text)
    if args.show_metadata:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.trace_output:
        write_json(args.trace_output, {"r8": result.to_dict()})
    if result.prediction is None:
        raise SystemExit(2)
