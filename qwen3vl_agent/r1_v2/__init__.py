"""R1 V2; input, output and evidence contracts remain compatible with R1."""

from qwen3vl_agent.r1 import R1Budget, R1Choice, R1Request, R1Result, TimeSpan
from qwen3vl_agent.r1_v2.agent import R1V2VideoAgent
from qwen3vl_agent.r1_v2.config import R1V2Config

__all__ = [
    "R1Budget",
    "R1Choice",
    "R1Request",
    "R1Result",
    "R1V2Config",
    "R1V2VideoAgent",
    "TimeSpan",
]
