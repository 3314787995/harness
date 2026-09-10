"""Necessary V2 regressions: real indexing/decoding, controlled model responses, no GPU."""

import importlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import av
import pytest
import yaml
from PIL import Image
from test_r1 import FakeModel, locate, observe, query
from test_r1345_debug_runner import Model, agent_factory, case

from qwen3vl_agent import cli, debug12
from qwen3vl_agent.config import load_config
from qwen3vl_agent.p01 import P01Config, TimeSpan
from qwen3vl_agent.r1 import R1Config, R1Request, R1VideoAgent
from qwen3vl_agent.r1_v2 import R1V2Config, R1V2VideoAgent
from qwen3vl_agent.r1_v2.media import R1V2IndexBuilder


@pytest.fixture(scope="module")
def real_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("r1-v2-real") / "shots.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=12)
        stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
        for i in range(65 * 12):
            # Different luminance triggers the existing grayscale change metric; 12 fps
            # also supports the unchanged 6/8 fps precision-observation requirements.
            colour = "red" if i < 25 * 12 else "white" if i < 52 * 12 else "blue"
            frame = av.VideoFrame.from_image(Image.new("RGB", (160, 96), colour))
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return path


def build(real_video, tmp_path, **handlers):
    model = FakeModel(**handlers)
    config = R1V2Config(media=P01Config(cache_dir=str(tmp_path / "cache")))
    return R1V2VideoAgent(model, config=config), model


def request(real_video, **kwargs):
    return R1Request(
        str(real_video),
        "What colour is the target person's clothing?",
        choices=("red", "green"),
        **kwargs,
    )
