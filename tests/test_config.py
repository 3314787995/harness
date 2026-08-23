from __future__ import annotations

from pathlib import Path

from qwen3vl_agent.config import load_config
from qwen3vl_agent.factory import build_model
from qwen3vl_agent.paths import default_videomme_paths


def test_config_expands_environment_values_and_shell_style_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("QWEN3VL_TEST_MODEL", "local/model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """model:
  path: ${QWEN3VL_TEST_MODEL:-fallback/model}
cache: ${QWEN3VL_TEST_CACHE:-.cache/test}
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["model"]["path"] == "local/model"
    assert config["cache"] == ".cache/test"


def test_model_default_can_be_overridden_by_environment(monkeypatch) -> None:
    monkeypatch.setenv("QWEN3VL_MODEL_PATH", "local/qwen3-vl")

    model = build_model()

    assert model.model_path == "local/qwen3-vl"


def test_videomme_layout_can_be_overridden_by_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("VIDEOMME_ROOT", str(tmp_path))

    paths = default_videomme_paths()

    assert paths.annotation == tmp_path / "videomme" / "test-00000-of-00001.parquet"
    assert paths.videos == tmp_path / "videos"
    assert paths.subtitles == tmp_path / "subtitle"
