from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.r1_v3 import R1V3VideoAgent
from qwen3vl_agent.r2 import R2VideoAgent
from qwen3vl_agent.r3 import R3VideoAgent
from qwen3vl_agent.r4 import R4VideoAgent
from qwen3vl_agent.r5 import R5VideoAgent
from qwen3vl_agent.r7 import R7VideoAgent

AGENTS = {'r1': R1V3VideoAgent, 'r2': R2VideoAgent, 'r3': R3VideoAgent,
          'r4': R4VideoAgent, 'r5': R5VideoAgent, 'r7': R7VideoAgent}

def normalize_strategy(strategy):
    return 'r1' if strategy == 'r1-v3' else strategy

def pipeline_settings(config, strategy):
    # Preserve the existing 3.1 config and trace namespace.
    if strategy == 'r1':
        return config.get('r1_v3', config.get('r1'))
    return config.get(strategy)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one of nine current Qwen3-VL video pipelines")
    parser.add_argument("--config", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument(
        "--strategy",
        choices=('r1', 'r1-v3', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'),
        default='r1',
        help='Select a current pipeline; r1-v3 is an alias of r1 (3.1).',
    )
    parser.add_argument("--choice", action="append", default=[])
    parser.add_argument("--subtitle", default=None)
    parser.add_argument("--asr", default=None, help="Existing ASR text; requires a pipeline with explicit text permissions")
    parser.add_argument(
        "--given-interval",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        default=None,
    )
    parser.add_argument("--trace-output", default=None)
    parser.add_argument("--allowed-scope", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--query-scope", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--output-protocol", choices=("auto", "multiple_choice", "free_text", "numeric", "interval", "list"),
                        default="auto")
    parser.add_argument("--execution-subtype", default=None, help="Optional R2/R3/R4/R5 operation hint")
    parser.add_argument("--observation-cutoff", type=float, default=None)
    parser.add_argument("--r3-policy", default=None, help="JSON file containing public task conventions")
    parser.add_argument("--r4-policy", default=None, help="JSON file containing public inventory conventions")
    parser.add_argument("--checkpoint", default=None, help="R2/R3/R4/R5 append-only checkpoint path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-model-calls", type=int, default=None, help="Lower the R2/R3/R4/R5 model-call hard cap")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--video-id", default=None)
    parser.add_argument("--group-id", default=None)
    parser.add_argument("--allowed-interval", action="append", nargs=2, type=float, metavar=("START", "END"))
    parser.add_argument("--query-time", type=float, default=None)
    parser.add_argument("--protocol-id", default="full_video")
    parser.add_argument("--r7-protocol", help="JSON containing R7 public observation permissions")
    parser.add_argument("--r6-protocol", help="JSON containing R6 public evidence permissions")
    parser.add_argument("--r6-mode", choices=("pipeline", "direct", "captions", "question_only"),
                        default="pipeline")
    parser.add_argument("--r8-protocol", help="JSON containing R8 public observation permissions")
    parser.add_argument("--r8-mode", choices=tuple("ABCDEFG"), default="G")
    parser.add_argument("--r9-mode", choices=("B0", "B1", "B2", "B3", "B4"), default="B4")
    parser.add_argument("--r9-unit", help="Public numeric answer unit for R9")
    parser.add_argument("--r9-allow-unresolved", action="store_true", help="Allow null R9 output when evidence is insufficient")
    parser.add_argument("--variables-input", help="R8 variable replay artifact; oracle data requires --diagnostic")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--require-choice", action="store_true", help="Reserve an explicit forced fallback for unresolved R8 MCQs")
    parser.add_argument("--r7-mode", choices=("B0", "B1", "B2", "B3", "B4", "B4-uniform"), default="B4")
    parser.add_argument("--facts-input", help="R7 validated factual observation replay artifact")
    parser.add_argument("--show-metadata", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    strategy = normalize_strategy(args.strategy)
    args.strategy = strategy
    if len(args.video) != 1 or args.image:
        raise SystemExit('All nine pipelines require exactly one --video and no --image')
    if strategy == 'r6':
        from qwen3vl_agent.r6.cli import run_generic
        run_generic(args)
        return
    if args.r6_protocol or args.r6_mode != 'pipeline':
        raise SystemExit('R6-specific flags require --strategy r6')
    if strategy == 'r9':
        from qwen3vl_agent.r9.cli import run_generic
        run_generic(args)
        return
    if args.r9_mode != 'B4' or args.r9_unit or args.r9_allow_unresolved:
        raise SystemExit('R9-specific flags require --strategy r9')
    if strategy == 'r8':
        from qwen3vl_agent.r8.cli import run_generic
        run_generic(args)
        return
    if args.r8_protocol or args.r8_mode != 'G' or args.variables_input or args.diagnostic or args.require_choice:
        raise SystemExit('R8-specific flags require --strategy r8')
    if strategy != 'r7' and (args.r7_protocol or args.facts_input or args.r7_mode != 'B4'):
        raise SystemExit('R7-specific flags require --strategy r7')
    if strategy == 'r7' and (not args.choice or args.output_protocol not in {'auto', 'multiple_choice'}):
        raise SystemExit('R7 requires original choices and multiple_choice output')
    if args.asr and strategy not in {'r2', 'r5'}:
        raise SystemExit('--asr requires r2, r5 or r6')
    if args.subtitle and strategy not in {'r2', 'r4', 'r5', 'r7'}:
        raise SystemExit('--subtitle requires a pipeline supporting aligned text')
    if strategy in {'r2','r5'} and args.output_protocol not in {'auto','multiple_choice','free_text'}:
        raise SystemExit('R2/R5 support multiple_choice and free_text output')
    if (args.output_protocol == 'list' and strategy != 'r4') or (args.output_protocol == 'interval' and strategy == 'r4'):
        raise SystemExit('list output requires r4; r4 does not support interval output')
    if args.r3_policy and strategy != 'r3':
        raise SystemExit('--r3-policy requires r3')
    if args.r4_policy and strategy != 'r4':
        raise SystemExit('--r4-policy requires r4')
    if strategy == 'r1' and any((args.execution_subtype, args.observation_cutoff is not None,
                               args.checkpoint, args.resume, args.max_model_calls is not None,
                               args.request_id, args.video_id, args.group_id)):
        raise SystemExit('R1 does not accept batch checkpoint or execution overrides')
    if strategy not in {'r2','r7'} and (args.allowed_interval or args.query_time is not None or args.protocol_id != 'full_video'):
        raise SystemExit('--allowed-interval, --query-time and --protocol-id require r2, r6 or r7')
    config = load_config(args.config) if args.config else {}
    model_config = dict(config.get('model') or {})
    model_config.setdefault('path', 'Qwen/Qwen3-VL-8B-Instruct')
    model_config.setdefault('dtype', 'bfloat16')
    model_config.setdefault('attn_implementation', 'flash_attention_2')
    if strategy == 'r7':
        from qwen3vl_agent.r7.config import R7Config
        model_config.setdefault('revision', R7Config.from_mapping(config.get('r7')).model_revision)
    model = build_model(model_config)
    agent = AGENTS[strategy](model, config=pipeline_settings(config, strategy))
    query = args.query
    agent.load()
    try:
        if strategy == "r7":
            protocol = json.loads(Path(args.r7_protocol).read_text(encoding="utf-8")) if args.r7_protocol else {}
            allowed = {"allowed_scope", "allowed_time_intervals", "observation_cutoff", "protocol_id",
                       "protocol_source", "available_modalities", "target_visibility"}
            if not isinstance(protocol, dict) or set(protocol) - allowed:
                raise SystemExit("--r7-protocol only accepts public observation protocol fields")
            defaults = {"allowed_scope": args.allowed_scope or args.given_interval,
                        "allowed_time_intervals": args.allowed_interval or (),
                        "observation_cutoff": args.observation_cutoff, "protocol_id": args.protocol_id,
                        "available_modalities": ("video", "screen_text", "subtitle") if args.subtitle else ("video", "screen_text")}
            for key, value in protocol.items():
                old = defaults.get(key)
                supplied = key in {"allowed_scope", "allowed_time_intervals", "observation_cutoff"} and old is not None and old != ()
                if supplied and json.loads(json.dumps(old)) != value:
                    raise SystemExit("R7 protocol conflicts with command-line scope")
            defaults.update(protocol)
            result = agent.generate(
                [{"role": "user", "content": query}], videos=args.video, choices=args.choice,
                **defaults, query_scope=args.query_scope, query_time=args.query_time,
                subtitle_path=args.subtitle, execution_subtype=args.execution_subtype,
                checkpoint_path=args.checkpoint, resume=args.resume,
                max_model_calls=args.max_model_calls, request_id=args.request_id or "r7-request",
                video_id=args.video_id or "video", group_id=args.group_id,
                output_protocol=args.output_protocol, mode=args.r7_mode, facts_input=args.facts_input,
            )
        elif strategy == "r2":
            from dataclasses import replace
            budget = agent.config.budget
            if args.max_model_calls is not None:
                budget = replace(budget, max_model_calls=min(args.max_model_calls, budget.max_model_calls))
            modalities = ["video", "screen_text"]
            if args.subtitle:
                modalities.append("subtitle")
            if args.asr:
                modalities.append("asr")
            result = agent.generate(
                [{"role": "user", "content": query}], videos=args.video, choices=args.choice,
                allowed_scope=args.allowed_scope or args.given_interval,
                allowed_time_intervals=args.allowed_interval or (), query_scope=args.query_scope,
                query_time=args.query_time, observation_cutoff=args.observation_cutoff,
                protocol_id=args.protocol_id, available_modalities=tuple(modalities),
                subtitle_path=args.subtitle, asr_path=args.asr,
                execution_subtype=args.execution_subtype, checkpoint_path=args.checkpoint,
                resume=args.resume, budget=budget, request_id=args.request_id or "r2-request",
                video_id=args.video_id or "video", group_id=args.group_id,
                output_protocol=args.output_protocol,
            )
        elif strategy == "r5":
            from dataclasses import replace

            budget = agent.config.budget
            if args.max_model_calls is not None:
                budget = replace(budget, max_model_calls=args.max_model_calls)
            result = agent.generate(
                [{"role": "user", "content": query}], videos=args.video, choices=args.choice,
                given_interval=args.given_interval, allowed_scope=args.allowed_scope,
                query_scope=args.query_scope, observation_cutoff=args.observation_cutoff,
                execution_subtype=args.execution_subtype, checkpoint_path=args.checkpoint,
                resume=args.resume, budget=budget, request_id=args.request_id or "r5-request",
                video_id=args.video_id or "video", group_id=args.group_id,
                output_protocol=args.output_protocol, subtitle_path=args.subtitle, asr_path=args.asr,
            )
        elif strategy in {"r3", "r4"}:
            from dataclasses import replace

            policy_path = args.r3_policy if strategy == "r3" else args.r4_policy
            policy = json.loads(Path(policy_path).read_text(encoding="utf-8")) if policy_path else {}
            budget = agent.config.budget
            if args.max_model_calls is not None:
                budget = replace(budget, max_model_calls=args.max_model_calls)
            external = {"subtitle_path": args.subtitle} if strategy == "r4" and args.subtitle else {}
            result = agent.generate(
                [{"role": "user", "content": query}], videos=args.video, choices=args.choice,
                given_interval=args.given_interval, allowed_scope=args.allowed_scope,
                query_scope=args.query_scope, observation_cutoff=args.observation_cutoff,
                execution_subtype=args.execution_subtype, benchmark_policy=policy,
                checkpoint_path=args.checkpoint, resume=args.resume, budget=budget,
                request_id=args.request_id or f"{strategy}-request", video_id=args.video_id or "video",
                group_id=args.group_id, output_protocol=args.output_protocol, **external,
            )
        elif strategy == "r1":
            result = agent.generate(
                [{"role": "user", "content": query}], videos=args.video,
                choices=args.choice, given_interval=args.given_interval,
                allowed_scope=args.allowed_scope, query_scope=args.query_scope,
                output_protocol=args.output_protocol,
            )
    finally:
        agent.unload()

    print(result.text)
    if args.show_metadata:
        print(json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str))
    if args.trace_output:
        output_path = Path(args.trace_output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if strategy == "r2" and result.metadata["r2"]["prediction"] is None:
        raise SystemExit(2)
    if strategy == "r7" and result.metadata["r7"]["prediction"] is None:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
