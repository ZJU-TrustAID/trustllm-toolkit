# src/config/loader.py
import yaml
from typing import Any, Dict

def deep_merge_dicts(base: Dict, new: Dict) -> Dict:
    """Recursively merges `new` dict into `base` dict."""
    merged = base.copy()
    for key, value in new.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

def load_and_merge_configs(
    general_path: str,
    experiment_path: str,
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Loads and merges configurations.
    Uses print instead of logger here, as logging system is not yet ready.
    """
    print(f"[Loader] Loading general config from: {general_path}")
    with open(general_path, 'r', encoding='utf-8') as f:
        general_config = yaml.safe_load(f) or {}

    print(f"[Loader] Loading experiment config from: {experiment_path}")
    with open(experiment_path, 'r', encoding='utf-8') as f:
        experiment_config = yaml.safe_load(f) or {}

    merged_config = deep_merge_dicts(general_config, experiment_config)
    
    if overrides:
        print(f"[Loader] Applying {len(overrides)} CLI overrides.")
        for key, value in overrides.items():
            keys = key.split('.')
            d = merged_config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value

    return merged_config
