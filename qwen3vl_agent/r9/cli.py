"""Narrow adapter for the shared strategy entry point."""

from qwen3vl_agent.factory import build_model

from .controller import R9VideoAgent
from .evaluate import model_and_config, write_json
from .types import R9Request


def run_generic(args):
    if len(args.video) != 1 or args.image or args.subtitle or args.asr:
        raise SystemExit("R9 requires one video; external images/subtitles/ASR are not enabled")
    if any(
        (
            args.r7_protocol,
            args.r8_protocol,
            args.r7_mode != "B4",
            args.r8_mode != "G",
            args.facts_input,
            args.variables_input,
            args.r3_policy,
            args.r4_policy,
            args.diagnostic,
            args.require_choice,
            args.replay_output,
            args.force_choice,
        )
    ):
        raise SystemExit("incompatible non-R9 flags")
    request = R9Request(
        args.video[0],
        args.query,
        choices=args.choice,
        allowed_scope=args.allowed_scope or args.given_interval,
        allowed_time_intervals=args.allowed_interval or (),
        query_scope=args.query_scope,
        query_time=args.query_time,
        observation_cutoff=args.observation_cutoff,
        output_protocol=args.output_protocol,
        output_unit=args.r9_unit,
        force_answer=not args.r9_allow_unresolved,
        mode=args.r9_mode,
        max_model_calls=args.max_model_calls,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        request_id=args.request_id or "r9-request",
        video_id=args.video_id or "video",
    )
    settings, config = model_and_config(args.config)
    agent = R9VideoAgent(build_model(settings), config)
    agent.load()
    try:
        result = agent.solve(request)
    finally:
        agent.unload()
    print(result.text)
    if args.show_metadata:
        import json

        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.trace_output:
        write_json(args.trace_output, {"r9": result.to_dict()})
    if result.prediction is None:
        raise SystemExit(2)
