# Contributing

Run all commands from the repository's `pipelines/` directory. Use Python 3.11 and install `requirements-cpu.txt` for engineering tests. Run `python -m pytest -q`, `python -m ruff check qwen3vl_agent tests scripts tools`, and `python tools/check_release.py` before publishing.

Preserve original choices, evidence provenance, modality permissions and observation bounds. Keep answers outside inference manifests. Never replace a failed or unobserved window with a negative observation. Update the per-pipeline version, current runbook and benchmark mapping when behavior changes. Record known failures explicitly in `docs/validation.md`; do not drop applicable regressions merely to obtain a green run.

Regenerate release_manifest.json file hashes for reviewed changes, excluding the manifest itself. Do not commit model weights, videos, local paths, credentials, caches, or run logs. Historical standalone solvers belong in Git history; required bases may remain internal.
