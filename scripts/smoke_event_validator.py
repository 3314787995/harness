from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3vl_agent.active_tree import ActiveTreeConfig, ActiveTreeVideoAgent
from qwen3vl_agent.active_tree.types import (
    EvidenceLedger,
    EvidenceSlot,
    PlannedAction,
    ResourceLedger,
    TaskContract,
)
from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke-test localized sequence-event validation without a full QA run."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--probe",
        action="append",
        required=True,
        metavar="NODE_ID::SLOT_ID::EVENT",
    )
    parser.add_argument("--output")
    return parser


def parse_probe(value: str) -> tuple[str, str, str]:
    parts = value.split("::", 2)
    if len(parts) != 3 or not all(item.strip() for item in parts):
        raise argparse.ArgumentTypeError(
            "probe must use NODE_ID::SLOT_ID::EVENT format"
        )
    node_id, slot_id, event = (item.strip() for item in parts)
    return node_id, slot_id, event


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    model = build_model(config.get("model"))
    agent = ActiveTreeVideoAgent(
        model,
        config=ActiveTreeConfig.from_mapping(config.get("active_tree")),
    )
    probes = [parse_probe(value) for value in args.probe]
    agent.load()
    try:
        cached = agent.cache.prepare(args.video)
        tree = agent.tree_builder.build(cached, None)
        records: list[dict[str, object]] = []
        for node_id, slot_id, event in probes:
            if node_id not in tree.nodes:
                raise KeyError(f"unknown node ID: {node_id}")
            ledger = EvidenceLedger()
            resources = ResourceLedger(agent.config.max_model_calls)
            trace: dict[str, object] = {"protocol_failures": [], "events": []}
            contract = TaskContract(
                "sequence",
                [EvidenceSlot(slot_id, event, True, "ground the visible transition")],
                ["visual"],
            )
            action = PlannedAction(
                "observe",
                node_id,
                "event_verify",
                slot_id,
                expected_new_evidence="localized event-validator smoke probe",
            )
            added = agent._execute_observation(
                "",
                [],
                contract,
                action,
                cached,
                None,
                tree,
                ledger,
                resources,
                trace,
            )
            records.append(
                {
                    "node_id": node_id,
                    "slot_id": slot_id,
                    "event": event,
                    "ledger_items_added": added,
                    "active_evidence_count": len(ledger.active_items),
                    "routing_clue_count": len(ledger.routing_items),
                    "active_evidence": [item.to_dict() for item in ledger.active_items],
                    "routing_clues": [item.to_dict() for item in ledger.routing_items],
                    "model_call": resources.to_dict()["calls"][0],
                    "observation": trace["events"],
                }
            )
    finally:
        agent.unload()

    payload = {"video": str(Path(args.video).resolve()), "records": records}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
