"""Check current APIs, versions, source links and publication hashes without a model."""
import argparse
import hashlib
import importlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
DIRECTORIES = {'qwen3vl_agent', 'configs', 'docs', 'examples', 'tests', 'tools', 'scripts'}
ROOT_FILES = {'README.md', 'CONTRIBUTING.md', 'pyproject.toml'}


def release_files():
    for name in ('README.md', '.gitignore', '.gitattributes', '.editorconfig'):
        yield REPO_ROOT / name
    yield from sorted(p for p in (REPO_ROOT / '.github').rglob('*') if p.is_file())
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT)
        if not p.is_file() or any(x in rel.parts for x in ('.git', '__pycache__', '.pytest_cache', '.ruff_cache')):
            continue
        if rel.parts[0] in DIRECTORIES or p.name in ROOT_FILES or (len(rel.parts) == 1 and p.name.startswith('requirements-')):
            yield p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest', action='store_true', help='Freeze reviewed publication files before verification')
    args = parser.parse_args()
    path = ROOT / 'release_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    hashes = {p.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in release_files()}
    if args.write_manifest:
        manifest['files'] = hashes
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    if hashes != manifest['files']:
        changed = sorted(k for k in hashes.keys() | manifest['files'].keys() if hashes.get(k) != manifest['files'].get(k))
        raise SystemExit('Publication hash mismatch: ' + ', '.join(changed))

    from qwen3vl_agent import cli
    from qwen3vl_agent.config import load_config

    package = importlib.import_module('qwen3vl_agent')
    sources = {'R1': ('r1_v3.version', 'POLICY_ID'), 'R3': ('r3.prompts', 'PROMPT_VERSION'),
               'R4': ('r4.prompts', 'VERSION'), 'R5': ('r5.observation', 'PROTOCOL_VERSION')}
    for number in range(1, 10):
        name = f'R{number}'
        module, constant = sources.get(name, (f'r{number}.types', 'VERSION'))
        actual = getattr(importlib.import_module('qwen3vl_agent.' + module), constant)
        assert actual == manifest['pipelines'][name]['protocol'], (name, actual)
        config = load_config(ROOT / f'configs/r{number}_8b.yaml')
        getattr(package, name + 'Config').from_mapping(cli.pipeline_settings(config, name.lower()))
        assert callable(getattr(package, name + 'VideoAgent'))
    options = next(x for x in cli.build_parser()._actions if x.dest == 'strategy').choices
    assert set(options) == {f'r{n}' for n in range(1, 10)} | {'r1-v3'}
    assert cli.normalize_strategy('r1-v3') == 'r1'

    broken = []
    for p in release_files():
        if p.suffix != '.md':
            continue
        text = p.read_text(encoding='utf-8-sig')
        targets = re.findall(r'\]\(([^)]+)\)', text)
        targets += re.findall(r'^\[[^\]]+\]:\s*(\S+)', text, re.M)
        for target in targets:
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target):
                continue
            if not (p.parent / unquote(target)).exists():
                broken.append((p.relative_to(REPO_ROOT).as_posix(), target))
    assert not broken, broken
    text = (ROOT / 'docs/benchmark_mapping.md').read_text(encoding='utf-8')
    section = text.split('## 4.')[1].split('## 5.')[0]
    assert len(re.findall(r'^\| `[^`]+` \|', section, re.M)) == 12
    for benchmark in ('EgoLifeQA','EgoSchema','LongVideoBench','LVBench','MLVU','MVBench','TOMATO','TVBench','Video-MME','Video-MME-v2','VSI-Bench'):
        assert benchmark in text
    print(f'PASS: {len(hashes)} file hashes; 9 versions/configs/APIs; latest-only CLI; local Markdown links; 11 benchmarks and 12 Video-MME entries.')


if __name__ == '__main__':
    main()
