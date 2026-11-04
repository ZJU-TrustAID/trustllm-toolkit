from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_HF_MIRROR_ENDPOINT = 'https://hf-mirror.com'


def _load_hf_download_helpers():
    from src.util.hf_download import (
        build_download_plan,
        discover_project_model_references,
        download_repo_snapshot,
    )

    return build_download_plan, discover_project_model_references, download_repo_snapshot


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _resolve_models_root(arg_value: str | None) -> Path:
    candidate = (arg_value or os.environ.get('HF_MODELS_ROOT') or '').strip()
    if not candidate:
        raise ValueError(
            'No models root configured. Pass `--models-root` or set `HF_MODELS_ROOT` in the environment.'
        )
    return Path(candidate).expanduser().resolve()


def _resolve_endpoint(arg_value: str | None) -> str:
    endpoint = (arg_value or os.environ.get('HF_ENDPOINT') or DEFAULT_HF_MIRROR_ENDPOINT).strip()
    if not endpoint:
        raise ValueError('HF endpoint cannot be empty.')
    return endpoint


def _resolve_token(arg_value: str | None) -> str | None:
    token = (
        arg_value
        or os.environ.get('HF_TOKEN')
        or os.environ.get('HUGGING_FACE_HUB_TOKEN')
        or os.environ.get('HUGGINGFACE_HUB_TOKEN')
    )
    if token is None:
        return None
    normalized = token.strip()
    return normalized or None


def _manual_model_references(models: list[str]) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = {}
    for repo_id in models:
        normalized = repo_id.strip()
        if not normalized:
            continue
        refs.setdefault(normalized, set()).add('cli:--model')
    return refs


def _print_plan(tasks, *, models_root: Path, endpoint: str) -> None:
    print(f'Model root: {models_root}')
    print(f'HF endpoint: {endpoint}')
    print(f'Total models: {len(tasks)}')
    print('')
    for task in tasks:
        print(f'[{task.action}] {task.repo_id} -> {task.target_dir}')
        for source in task.sources:
            print(f'  source: {source}')


def main() -> int:
    _load_dotenv_if_available()

    parser = argparse.ArgumentParser(
        description='Download Hugging Face models for MarkSec via a configurable mirror.'
    )
    parser.add_argument(
        '--models-root',
        type=str,
        default=None,
        help='Target root directory. Defaults to HF_MODELS_ROOT from env or .env.',
    )
    parser.add_argument(
        '--mode',
        choices=['sync', 'missing'],
        default='sync',
        help='`sync` resumes or completes every selected model; `missing` only downloads absent model directories.',
    )
    parser.add_argument(
        '--endpoint',
        type=str,
        default=None,
        help=f'HF endpoint or mirror base URL. Defaults to HF_ENDPOINT or {DEFAULT_HF_MIRROR_ENDPOINT}.',
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Optional HF token. Defaults to HF_TOKEN / HUGGING_FACE_HUB_TOKEN if present.',
    )
    parser.add_argument(
        '--model',
        action='append',
        default=[],
        help='Explicit repo id to download. Repeat to provide multiple models. When omitted, models are auto-discovered from this repo.',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=8,
        help='Concurrent download workers per model snapshot.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the resolved plan without downloading.',
    )
    args = parser.parse_args()

    build_download_plan, discover_project_model_references, download_repo_snapshot = (
        _load_hf_download_helpers()
    )

    models_root = _resolve_models_root(args.models_root)
    endpoint = _resolve_endpoint(args.endpoint)
    token = _resolve_token(args.token)

    if args.model:
        refs = _manual_model_references(args.model)
    else:
        refs = discover_project_model_references(PROJECT_ROOT)

    if not refs:
        raise RuntimeError('No Hugging Face model ids were found for download.')

    tasks = build_download_plan(refs, models_root=models_root, mode=args.mode)
    _print_plan(tasks, models_root=models_root, endpoint=endpoint)

    if args.dry_run:
        return 0

    failures: list[str] = []
    for task in tasks:
        if task.action == 'skip':
            continue
        print(f'Downloading {task.repo_id} ...', flush=True)
        try:
            download_repo_snapshot(
                repo_id=task.repo_id,
                target_dir=task.target_dir,
                endpoint=endpoint,
                max_workers=args.max_workers,
                token=token,
            )
        except Exception as exc:
            failures.append(f'{task.repo_id}: {exc}')
            print(f'  failed: {exc}', flush=True)
            continue
        print(f'  done: {task.target_dir}', flush=True)

    if failures:
        print('')
        print('Some downloads failed:')
        for item in failures:
            print(f'- {item}')
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
