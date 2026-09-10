"""Public latest-only routing and trace compatibility; no GPU required."""
import importlib
import json
import sys
from pathlib import Path

import pytest

from qwen3vl_agent import cli
from qwen3vl_agent.config import load_config
from qwen3vl_agent.models.base import ModelOutput
from qwen3vl_agent.r1_v3 import R1V3VideoAgent


@pytest.mark.parametrize('strategy', ['r1', 'r1-v3'])
def test_r1_alias_dispatches_latest_and_keeps_trace(strategy, monkeypatch, tmp_path):
    received = {}

    class Agent:
        def __init__(self, model, *, config):
            received['config'] = config

        def load(self):
            received['loaded'] = True

        def unload(self):
            received['unloaded'] = True

        def generate(self, messages, **kwargs):
            received.update(kwargs)
            return ModelOutput(text='Y', metadata={'r1_v3': {'policy_id': 'r1-local-evidence/3.1'}})

    assert cli.AGENTS['r1'] is R1V3VideoAgent
    monkeypatch.setitem(cli.AGENTS, 'r1', Agent)
    monkeypatch.setattr(cli, 'build_model', lambda _: object())
    output = tmp_path / 'trace.json'
    monkeypatch.setattr(sys, 'argv', ['cli', '--strategy', strategy, '--video', 'video.mp4',
                        '--query', 'Read the sign.', '--choice', 'Y: yes', '--choice', 'N: no',
                        '--config', str(Path(__file__).parents[1] / 'configs/r1_8b.yaml'),
                        '--allowed-scope', '2', '8', '--trace-output', str(output)])
    cli.main()
    assert received['choices'] == ['Y: yes', 'N: no']
    assert received['allowed_scope'] == [2, 8]
    assert received['config']['observer_tokens'] == 3072
    assert received['loaded'] and received['unloaded']
    assert json.loads(output.read_text())['r1_v3']['policy_id'] == 'r1-local-evidence/3.1'


@pytest.mark.parametrize('strategy', ['p01', 'p01-v3', 'p05', 'r1-v2', 'active_tree', 'coarse_to_fine'])
def test_removed_standalone_strategies_rejected(strategy):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['--query', 'q', '--strategy', strategy])


@pytest.mark.parametrize('number', range(1, 10))
def test_current_config_and_public_api(number):
    root = Path(__file__).parents[1]
    module = importlib.import_module('qwen3vl_agent')
    config = load_config(root / f'configs/r{number}_8b.yaml')
    cls = getattr(module, f'R{number}Config')
    value = cls.from_mapping(cli.pipeline_settings(config, f'r{number}'))
    assert value is not None
    assert getattr(module, f'R{number}VideoAgent') is not None
    assert getattr(module, f'R{number}Request') is not None


def test_default_and_alias_use_same_current_configuration():
    assert cli.normalize_strategy(cli.build_parser().parse_args(['--query', 'q']).strategy) == 'r1'
    assert cli.normalize_strategy('r1-v3') == 'r1'
    root = Path(__file__).parents[1]
    assert load_config(root / 'configs/r1_8b.yaml') == load_config(root / 'configs/r1_v3_8b.yaml')
