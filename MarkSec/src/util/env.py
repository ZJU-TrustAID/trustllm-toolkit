# =======================================================
# src/util/env.py
# Description: Environment, System, and Hardware utilities
# =======================================================

import os
import random
import logging
import numpy as np
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Return the repository root for the current checkout."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path | None, project_root: str | Path | None = None) -> str:
    """
    Resolve a repo-relative path against the current project root.

    Absolute paths are returned unchanged so existing machine-local paths keep
    working. Empty values are normalized to an empty string.
    """
    if path is None:
        return ''

    raw = str(path).strip()
    if not raw:
        return ''

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    root = Path(project_root).expanduser() if project_root is not None else get_project_root()
    return str((root / candidate).resolve())


def _load_dotenv_if_needed() -> None:
    """Load .env only when HF_MODELS_ROOT is not already present."""
    if 'HF_MODELS_ROOT' in os.environ:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def set_seed(seed: int):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_best_free_gpu() -> str:
    """
    Identifies the GPU with the most free memory.
    Returns 'cpu' if no GPU is available.
    """
    if not torch.cuda.is_available():
        return 'cpu'

    best_gpu_index = 0
    max_free_memory = 0

    try:
        for i in range(torch.cuda.device_count()):
            # torch.cuda.mem_get_info(i) -> (free, total) bytes
            free_memory, _ = torch.cuda.mem_get_info(i)
            if free_memory > max_free_memory:
                max_free_memory = free_memory
                best_gpu_index = i

        logger.info(
            f'GPU Status: Found {torch.cuda.device_count()} GPUs. Best GPU: {best_gpu_index} with {max_free_memory / 1024**3:.2f} GiB free.'
        )
        return f'cuda:{best_gpu_index}'
    except Exception as e:
        logger.warning(f'Failed to query GPU info: {e}. Defaulting to cuda:0')
        return 'cuda:0'


def extract_model_name(model_name_or_path: str) -> str:
    """
    Extracts a clean model name from a Hugging Face Hub ID or a local path.

    Examples:
        - "facebook/opt-1.3b" -> "opt-1.3b"
        - "/path/to/models/llama-2-7b" -> "llama-2-7b"
        - "bert-base-uncased" -> "bert-base-uncased"
    """
    if not model_name_or_path:
        return 'UnknownModel'

    # .name retrieves the last component of the path
    clean_name = Path(model_name_or_path.strip(os.sep)).name

    return clean_name


def extract_full_model_id(model_name_or_path: str) -> str:
    """
    Extracts a structure similar to Hugging Face Repo ID (Org/Model) from a path.
    If the path ends with two meaningful components, it returns them joined by '/'.
    """
    if not model_name_or_path:
        return ''

    # Strip trailing separator
    path_str = model_name_or_path.strip(os.sep)
    parts = path_str.split(os.sep)

    if len(parts) >= 2:
        return f'{parts[-2]}/{parts[-1]}'
    elif len(parts) == 1:
        return parts[0]
    return ''


def configure_models_root(models_root: str | None) -> None:
    """
    Configure the runtime model root from project config.

    A non-empty value overrides HF_MODELS_ROOT for this process. Empty values
    are ignored so callers may still rely on an existing environment setting.
    """
    normalized = (models_root or '').strip()
    if not normalized:
        return
    os.environ['HF_MODELS_ROOT'] = normalized
    logger.info('Configured HF_MODELS_ROOT from experiment config.')


def resolve_model_path(model_name_or_path: str, models_root: str | None = None) -> str:
    """
    Resolve a model id against a configured local models root when available.

    Resolution order:
    1. Absolute paths are returned unchanged.
    2. Explicit models_root argument.
    3. HF_MODELS_ROOT from the environment or .env.
    4. Fallback to the original input string.
    """
    if not model_name_or_path:
        return model_name_or_path

    # If it's already an absolute path, return it
    if os.path.isabs(model_name_or_path):
        return model_name_or_path

    resolved_models_root = (models_root or '').strip()
    if not resolved_models_root:
        _load_dotenv_if_needed()
        resolved_models_root = os.environ.get('HF_MODELS_ROOT', '').strip()

    if resolved_models_root:
        full_path = os.path.join(resolved_models_root, model_name_or_path)
        if os.path.exists(full_path):
            return full_path

    # If not found locally, return the original name (likely for HF Hub download)
    return model_name_or_path
