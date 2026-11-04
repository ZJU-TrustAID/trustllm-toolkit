"""
Defines threat models (attacker capabilities) for different victim models.
Maps a victim model (base model) to an appropriate surrogate/attack model.
"""

from __future__ import annotations

from pathlib import Path

THREAT_MODEL_MAPPING = {
    # Llama Family
    'meta-llama/Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.2-1B-Instruct',
    # Qwen Family
    'Qwen/Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-1.5B-Instruct',
    # Mistral Family
    # Since Mistral doesn't have a direct 1B/3B variant in the same line,
    # we use Llama-3.2-1B as a generic strong surrogate.
    'mistralai/Mistral-7B-Instruct-v0.3': 'meta-llama/Llama-3.2-1B-Instruct',
}

AUTO_ATTACK_MODEL_SENTINELS = {
    '',
    'auto',
    'threat_model',
    'threat-model',
    'surrogate',
    'base',
    'base_model',
    'base-model',
    'same_as_base',
    'same-as-base',
    'victim',
}


def _candidate_base_names(base_model_name: str) -> list[str]:
    """Build candidate ids from model id or local path for robust mapping lookup."""
    if not isinstance(base_model_name, str):
        return []

    raw = base_model_name.strip()
    if not raw:
        return []

    candidates: list[str] = [raw]
    path = Path(raw)
    path_parts = [part for part in path.parts if part and part != '/']
    if len(path_parts) >= 2:
        # Typical local mirror format: .../<org>/<model-name>
        tail_repo_id = f'{path_parts[-2]}/{path_parts[-1]}'
        candidates.append(tail_repo_id)
        candidates.append(path_parts[-1])

    # Preserve order while deduplicating.
    seen: set[str] = set()
    unique_candidates = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            unique_candidates.append(item)
    return unique_candidates


def _is_auto_request(base_model_name: str, requested_attack_model: str | None) -> bool:
    """Return whether requested attack model means 'resolve from threat model'."""
    if requested_attack_model is None:
        return True

    requested = str(requested_attack_model).strip()
    if not requested:
        return True

    if requested == str(base_model_name).strip():
        return True

    return requested.lower() in AUTO_ATTACK_MODEL_SENTINELS


def resolve_attack_model(base_model_name: str, requested_attack_model: str | None = None) -> str:
    """
    Resolves the attack model to use.

    Args:
        base_model_name: The name of the victim/base model.
        requested_attack_model: The attack model explicitly requested in config (optional).

    Returns:
        The resolved attack model path/name.
    """
    # Respect explicit manual override unless caller requested auto resolution.
    if not _is_auto_request(base_model_name, requested_attack_model):
        return str(requested_attack_model).strip()

    base_candidates = _candidate_base_names(base_model_name)
    if not base_candidates:
        return base_model_name

    # Exact mapping first.
    for candidate in base_candidates:
        mapped = THREAT_MODEL_MAPPING.get(candidate)
        if mapped:
            return mapped

    # Substring/fuzzy matching for local path variants.
    for candidate in base_candidates:
        for victim, attacker in THREAT_MODEL_MAPPING.items():
            if victim in candidate or candidate in victim:
                return attacker

    # Fallback to the victim/base model if no mapping is known.
    return base_model_name
