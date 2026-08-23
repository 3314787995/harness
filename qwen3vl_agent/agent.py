from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from qwen3vl_agent.models.base import BaseVideoModel, ModelOutput
from qwen3vl_agent.tools import BaseTool, ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

PLANNER_PROMPT = """You are the tool planner for a Qwen3-VL assistant.

Available tools (JSON):
{tools}

User question:
{question}

Select only tools that are necessary to answer the question. Arguments must follow each
tool's JSON Schema. Return exactly one JSON object in this form:
{{"tools": [{{"name": "tool_name", "arguments": {{}}}}]}}
Return {{"tools": []}} when no tool is needed. Do not add markdown or explanation.
"""


class Qwen3VLAgent:
    """Compose Qwen3-VL with a dynamically extensible set of tools."""

    def __init__(
        self,
        model: BaseVideoModel,
        *,
        tools: ToolRegistry | None = None,
        use_tools: bool = True,
        max_tool_calls: int = 2,
        planning_max_new_tokens: int = 128,
        planning_temperature: float = 0.0,
        max_tool_result_chars: int = 12_000,
    ) -> None:
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        self.model = model
        self.tools = tools if tools is not None else ToolRegistry()
        self.use_tools = use_tools
        self.max_tool_calls = max_tool_calls
        self.planning_max_new_tokens = planning_max_new_tokens
        self.planning_temperature = planning_temperature
        self.max_tool_result_chars = max_tool_result_chars
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self.model.load()
        try:
            self.tools.load_all()
        except Exception:
            self.model.unload()
            raise
        self._loaded = True

    def unload(self) -> None:
        try:
            self.tools.unload_all()
        finally:
            self.model.unload()
            self._loaded = False

    def register_tool(self, tool: BaseTool, *, replace: bool = False) -> BaseTool:
        registered = self.tools.register(tool, replace=replace)
        if self._loaded:
            registered.load()
        return registered

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        videos: list[str] | None = None,
        images: list[str] | None = None,
        use_tools: bool | None = None,
        tool_metadata: Mapping[str, Any] | None = None,
        **generation: Any,
    ) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Agent is not loaded. Call load() first.")

        should_use_tools = self.use_tools if use_tools is None else use_tools
        if not should_use_tools or not self.tools or self.max_tool_calls == 0:
            return self.model.generate(
                messages,
                videos=videos,
                images=images,
                **generation,
            )

        question = self._latest_user_text(messages)
        try:
            planner_output, plan = self._plan(question, videos=videos, images=images)
        except Exception as exc:  # noqa: BLE001 - planner failure must degrade to direct inference
            logger.warning("Tool planning failed; falling back to direct generation: %s", exc)
            output = self.model.generate(
                messages,
                videos=videos,
                images=images,
                **generation,
            )
            output.metadata["tool_planner"] = {
                "error": f"{type(exc).__name__}: {exc}",
                "requested_calls": [],
            }
            output.metadata["tool_calls"] = []
            return output
        context = ToolContext(
            question=question,
            messages=messages,
            videos=tuple(videos or ()),
            images=tuple(images or ()),
            metadata=tool_metadata or {},
        )
        records = self._execute(plan, context)
        final_messages = self._append_evidence(messages, records)
        output = self.model.generate(
            final_messages,
            videos=videos,
            images=images,
            **generation,
        )
        output.metadata["tool_planner"] = {
            "raw_response": planner_output.text,
            "requested_calls": plan,
        }
        output.metadata["tool_calls"] = records
        return output

    def _plan(
        self,
        question: str,
        *,
        videos: list[str] | None,
        images: list[str] | None,
    ) -> tuple[ModelOutput, list[dict[str, Any]]]:
        prompt = PLANNER_PROMPT.format(
            tools=json.dumps(self.tools.manifests(), ensure_ascii=False),
            question=question,
        )
        output = self.model.generate(
            [{"role": "user", "content": prompt}],
            max_new_tokens=self.planning_max_new_tokens,
            temperature=self.planning_temperature,
        )
        payload = self._parse_json_object(output.text)
        raw_calls = payload.get("tools", [])
        if not isinstance(raw_calls, list):
            raise TypeError("Planner response field 'tools' must be a list")

        calls: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            name = raw_call.get("name")
            arguments = raw_call.get("arguments", {})
            if not isinstance(name, str) or name not in self.tools or name in seen:
                continue
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append({"name": name, "arguments": arguments})
            seen.add(name)
            if len(calls) >= self.max_tool_calls:
                break
        return output, calls

    def _execute(
        self,
        calls: Sequence[Mapping[str, Any]],
        context: ToolContext,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for call in calls:
            name = str(call["name"])
            arguments = dict(call.get("arguments") or {})
            tool = self.tools.get(name)
            if tool is None:
                continue
            started = time.perf_counter()
            record: dict[str, Any] = {"name": name, "arguments": arguments}
            try:
                result = tool.invoke(context, arguments)
                record["ok"] = True
                record["result"] = self._json_safe(result.data)
                if result.metadata:
                    record["metadata"] = self._json_safe(result.metadata)
            except Exception as exc:
                logger.exception("Tool %s failed; continuing to final answer", name)
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["latency_seconds"] = time.perf_counter() - started
            records.append(record)
        return records

    def _append_evidence(
        self,
        messages: list[dict[str, Any]],
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        successful = [record for record in records if record.get("ok")]
        if not successful:
            return messages

        evidence = json.dumps(successful, ensure_ascii=False, default=str)
        if len(evidence) > self.max_tool_result_chars:
            evidence = evidence[: self.max_tool_result_chars] + "...<truncated>"
        block = (
            "The following tool results are supplementary evidence. Use them when relevant, "
            "but prefer direct visual evidence if they conflict:\n" + evidence
        )

        updated = [dict(message) for message in messages]
        for index in range(len(updated) - 1, -1, -1):
            if updated[index].get("role") != "user":
                continue
            content = updated[index].get("content", "")
            if isinstance(content, list):
                updated[index]["content"] = list(content) + [{"type": "text", "text": block}]
            else:
                updated[index]["content"] = f"{content}\n\n{block}"
            return updated
        return [*updated, {"role": "user", "content": block}]

    @staticmethod
    def _latest_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
        return ""

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.removeprefix("```json").removeprefix("```")
            candidate = candidate.removesuffix("```").strip()
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            value = None
            for index, character in enumerate(candidate):
                if character != "{":
                    continue
                try:
                    value, _ = decoder.raw_decode(candidate[index:])
                    break
                except json.JSONDecodeError:
                    continue
            if value is None:
                raise ValueError(f"Planner did not return valid JSON: {text!r}")
        if not isinstance(value, dict):
            raise TypeError("Planner response must be a JSON object")
        return value

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return cls._json_safe(asdict(value))
        if isinstance(value, Mapping):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
