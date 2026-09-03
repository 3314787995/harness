from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.p01.config import P01Config
from qwen3vl_agent.p01.smoke import BENCHMARKS, load_smoke_questions


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str
    details: dict[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"pass", "warning", "error"}:
            raise ValueError(f"unsupported preflight status: {self.status}")


def _gib(value: float) -> float:
    return round(float(value) / (1024**3), 3)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_packages(attention: str | None) -> list[PreflightCheck]:
    requirements = (
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("av", "av"),
        ("Pillow", "PIL"),
        ("qwen-vl-utils", "qwen_vl_utils"),
    )
    checks: list[PreflightCheck] = []
    for distribution, module in requirements:
        version = _package_version(distribution)
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001  # import health is the check
            checks.append(
                PreflightCheck(
                    f"package:{distribution}",
                    "error",
                    f"cannot import {module}: {type(exc).__name__}: {exc}",
                    {"version": version},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    f"package:{distribution}",
                    "pass",
                    f"{distribution} {version or 'installed'}",
                    {"version": version},
                )
            )
    if attention == "flash_attention_2":
        version = _package_version("flash-attn")
        try:
            importlib.import_module("flash_attn")
        except Exception as exc:  # noqa: BLE001  # import health is the check
            checks.append(
                PreflightCheck(
                    "package:flash-attn",
                    "error",
                    f"flash_attention_2 is configured but import failed: {exc}",
                    {"version": version},
                )
            )
        else:
            checks.append(
                PreflightCheck(
                    "package:flash-attn",
                    "pass",
                    f"flash-attn {version or 'installed'}",
                    {"version": version},
                )
            )
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except Exception as exc:  # noqa: BLE001  # import health is the check
        checks.append(
            PreflightCheck(
                "transformers:qwen3_vl",
                "error",
                f"Qwen3VLForConditionalGeneration is unavailable: {exc}",
                {},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "transformers:qwen3_vl",
                "pass",
                "Qwen3-VL model class is available",
                {},
            )
        )
    return checks


def _nvidia_smi() -> dict[str, Any]:
    fields = (
        "index,name,memory.total,utilization.gpu,temperature.gpu,"
        "driver_version,power.draw,clocks.sm"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    visible_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_indices = [
        int(item.strip())
        for item in visible_raw.split(",")
        if item.strip().isdigit()
    ]
    parsed: list[dict[str, Any]] = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 8:
            continue
        parsed.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": float(parts[2]),
                "utilization_percent": float(parts[3]),
                "temperature_c": float(parts[4]),
                "driver_version": parts[5],
                "power_draw_w": float(parts[6]),
                "sm_clock_mhz": float(parts[7]),
            }
        )
    selected_devices = (
        [item for item in parsed if item["index"] in visible_indices]
        if visible_indices
        else parsed
    )
    return {
        "selected": selected_devices[0] if selected_devices else None,
        "selected_devices": selected_devices,
        "visible_physical_indices": visible_indices,
        "all_device_count": len(parsed),
    }


def _check_gpu(min_vram_gib: float, *, required_gpus: int = 1) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001  # import health is the check
        return [
            PreflightCheck(
                "gpu:torch",
                "error",
                f"cannot import torch: {exc}",
                {},
            )
        ]
    if not torch.cuda.is_available():
        return [
            PreflightCheck(
                "gpu:cuda",
                "error",
                "torch.cuda.is_available() is false",
                {"torch_cuda_version": torch.version.cuda},
            )
        ]
    device_count = torch.cuda.device_count()
    checks.append(
        PreflightCheck(
            "gpu:device_count",
            "pass" if device_count >= required_gpus else "error",
            f"torch sees {device_count} CUDA device(s); {required_gpus} required",
            {
                "device_count": device_count,
                "required_gpus": required_gpus,
                "torch_cuda_version": torch.version.cuda,
            },
        )
    )
    devices: list[dict[str, Any]] = []
    compatible = True
    for device in range(device_count):
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        total_gib = _gib(properties.total_memory)
        okay = total_gib >= min_vram_gib and capability[0] >= 8
        compatible = compatible and okay
        details = {
            "device_index": device,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "total_memory_gib": total_gib,
            "compute_capability": list(capability),
        }
        devices.append(details)
        checks.append(
            PreflightCheck(
                f"gpu:device:{device}",
                "pass" if okay else "error",
                (
                    f"{properties.name}, {total_gib:.3f} GiB, "
                    f"compute capability {capability}"
                ),
                details,
            )
        )
    checks.append(
        PreflightCheck(
            "gpu:bf16_flash_attention",
            "pass" if compatible and device_count >= required_gpus else "error",
            (
                "all visible GPUs satisfy the BF16/VRAM profile"
                if compatible
                else "one or more visible GPUs fail the BF16/VRAM profile"
            ),
            {"devices": devices},
        )
    )
    smi = _nvidia_smi()
    if smi.get("error"):
        checks.append(
            PreflightCheck(
                "gpu:nvidia_smi",
                "warning",
                f"nvidia-smi telemetry unavailable: {smi['error']}",
                smi,
            )
        )
    else:
        selected_devices = smi.get("selected_devices") or []
        hot_idle = any(
            float(selected.get("utilization_percent", 100)) <= 5
            and float(selected.get("temperature_c", 0)) >= 70
            for selected in selected_devices
        )
        checks.append(
            PreflightCheck(
                "gpu:nvidia_smi",
                "warning" if hot_idle else "pass",
                (
                    "GPU is at least 70C while utilization is at most 5%; "
                    "watch for cooling or throttling"
                    if hot_idle
                    else "nvidia-smi telemetry looks usable"
                ),
                smi,
            )
        )
    return checks


def _check_model_path(model_path: str) -> PreflightCheck:
    candidate = Path(model_path).expanduser()
    if candidate.is_dir():
        size = sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file())
        config_exists = (candidate / "config.json").is_file()
        return PreflightCheck(
            "model:path",
            "pass" if config_exists else "error",
            (
                f"local model directory is present ({_gib(size):.3f} GiB)"
                if config_exists
                else "local model directory is missing config.json"
            ),
            {"path": str(candidate.resolve()), "size_bytes": size},
        )
    if "/" in model_path and not candidate.is_absolute():
        return PreflightCheck(
            "model:path",
            "warning",
            "model is a remote Hugging Face ID; first load will download weights",
            {"model_id": model_path},
        )
    return PreflightCheck(
        "model:path",
        "error",
        "configured local model path does not exist",
        {"path": str(candidate)},
    )


def _load_model_check(config: dict[str, Any]) -> PreflightCheck:
    model = build_model(config.get("model"))
    try:
        model.load()
        details: dict[str, Any] = {}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                details = {
                    "devices": [
                        {
                            "device_index": device,
                            "memory_allocated_bytes": torch.cuda.memory_allocated(device),
                            "memory_reserved_bytes": torch.cuda.memory_reserved(device),
                            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device),
                        }
                        for device in range(torch.cuda.device_count())
                    ],
                    "hf_device_map": getattr(model, "hf_device_map", {}),
                }
        except Exception:  # noqa: BLE001, S110  # telemetry is best-effort
            pass
    except Exception as exc:  # noqa: BLE001  # model load is the check boundary
        return PreflightCheck(
            "model:load",
            "error",
            f"model load failed: {type(exc).__name__}: {exc}",
            {},
        )
    finally:
        try:
            model.unload()
        except Exception:  # noqa: BLE001, S110  # cleanup must not hide load result
            pass
    return PreflightCheck(
        "model:load",
        "pass",
        "model loaded and unloaded successfully",
        details,
    )


def run_preflight(
    *,
    config_path: str | Path,
    data_root: str | Path,
    work_dir: str | Path,
    min_free_gib: float = 8.0,
    min_vram_gib: float = 23.0,
    static_only: bool = False,
    load_model: bool = False,
    required_gpus: int | None = None,
) -> dict[str, Any]:
    config_file = Path(config_path).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    workspace = Path(work_dir).expanduser().resolve()
    checks: list[PreflightCheck] = []
    try:
        config = load_config(config_file)
        P01Config.from_mapping(config.get("p01"))
    except Exception as exc:  # noqa: BLE001  # report malformed config uniformly
        config = {}
        checks.append(
            PreflightCheck(
                "config",
                "error",
                f"configuration is invalid: {type(exc).__name__}: {exc}",
                {"path": str(config_file)},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "config",
                "pass",
                "P01 v2 configuration loaded and validated",
                {"path": str(config_file), "strategy": "p01-v2"},
            )
        )
    try:
        questions = load_smoke_questions(root, benchmarks=BENCHMARKS)
        video_paths = {item.video_path for item in questions}
    except Exception as exc:  # noqa: BLE001  # report malformed data uniformly
        checks.append(
            PreflightCheck(
                "data",
                "error",
                f"smoke data validation failed: {type(exc).__name__}: {exc}",
                {"data_root": str(root)},
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "data",
                "pass" if len(questions) == 25 and len(video_paths) == 20 else "warning",
                f"loaded {len(questions)} questions and {len(video_paths)} unique videos",
                {
                    "data_root": str(root),
                    "question_count": len(questions),
                    "unique_video_count": len(video_paths),
                },
            )
        )
    if not workspace.exists():
        checks.append(
            PreflightCheck(
                "disk",
                "error",
                "work directory does not exist",
                {"path": str(workspace)},
            )
        )
    else:
        usage = shutil.disk_usage(workspace)
        free_gib = _gib(usage.free)
        checks.append(
            PreflightCheck(
                "disk",
                "pass" if free_gib >= min_free_gib else "error",
                f"{free_gib:.3f} GiB free at work directory",
                {
                    "path": str(workspace),
                    "free_bytes": usage.free,
                    "free_gib": free_gib,
                    "required_free_gib": min_free_gib,
                },
            )
        )
    if config:
        model_config = dict(config.get("model") or {})
        model_path = str(model_config.get("path") or "Qwen/Qwen3-VL-8B-Instruct")
        checks.append(_check_model_path(model_path))
        if not static_only:
            checks.extend(_check_packages(model_config.get("attn_implementation")))
            inferred_required = required_gpus
            if inferred_required is None:
                configured = model_config.get("required_cuda_devices") or ()
                inferred_required = len(configured) or 1
            checks.extend(_check_gpu(min_vram_gib, required_gpus=inferred_required))
            if load_model and not any(item.status == "error" for item in checks):
                checks.append(_load_model_check(config))
    counts = {
        status: sum(item.status == status for item in checks)
        for status in ("pass", "warning", "error")
    }
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "static_only": static_only,
        },
        "summary": {
            **counts,
            "ready": counts["error"] == 0,
        },
        "checks": [asdict(item) for item in checks],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a rental GPU host before running the P01 smoke set."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--json-output")
    parser.add_argument("--min-free-gib", type=float, default=8.0)
    parser.add_argument("--min-vram-gib", type=float, default=23.0)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument(
        "--required-gpus",
        type=int,
        default=None,
        help="Override the GPU count inferred from model.required_cuda_devices.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_preflight(
        config_path=args.config,
        data_root=args.data_root,
        work_dir=args.work_dir,
        min_free_gib=args.min_free_gib,
        min_vram_gib=args.min_vram_gib,
        static_only=args.static_only,
        load_model=args.load_model,
        required_gpus=args.required_gpus,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_output:
        target = Path(args.json_output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["summary"]["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
