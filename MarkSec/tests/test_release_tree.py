from __future__ import annotations

from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[1]


def iter_text_files():
    ignored_dirs = {
        '.git',
        '.codex',
        '.agents',
        '.pytest_cache',
        '.ruff_cache',
        '__pycache__',
        'node_modules',
        '.next',
        '.venv',
        'build',
        'dist',
        'output',
        'tests',
    }
    suffixes = {
        '.py',
        '.md',
        '.toml',
        '.txt',
        '.yaml',
        '.yml',
        '.json',
        '.in',
    }
    for directory, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [name for name in dirs if name not in ignored_dirs and not name.endswith('.egg-info')]
        for filename in files:
            path = Path(directory) / filename
            if path.suffix in suffixes or path.name in {'NOTICE', 'LICENSE', '.gitignore'}:
                yield path


def test_markllm_vendor_tree_is_not_present():
    assert not (REPO_ROOT / 'libs' / 'MarkLLM').exists()


def test_no_runtime_references_to_removed_markllm_or_project_manager():
    banned_patterns = [
        'libs/MarkLLM',
        'AutoWatermark',
        'TransformersConfig',
        'markllm_path',
        'markllm_root',
        'project-manager',
        'job_manager',
        'gen_script',
        'state/main',
    ]
    allowlist = {
        'README.md',
        'NOTICE',
    }
    offenders = []
    for path in iter_text_files():
        if path.relative_to(REPO_ROOT).as_posix() in allowlist:
            continue
        text = path.read_text(encoding='utf-8', errors='ignore')
        for pattern in banned_patterns:
            if pattern in text:
                offenders.append(f'{path.relative_to(REPO_ROOT)}: {pattern}')

    assert offenders == []


def test_tests_are_excluded_from_distribution_manifest():
    manifest = (REPO_ROOT / 'MANIFEST.in').read_text(encoding='utf-8')
    assert 'prune tests' in manifest

    pyproject = (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'include = ["src*"]' in pyproject
