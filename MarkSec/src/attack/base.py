# ===============================================================
# base.py
# Description: This file contains the base configuration and attack
#              classes that will be inherited by specific watermark
#              attack implementations.
# ===============================================================
import os
from src.util.data import load_config_file


class BaseConfig:
    """Base configuration class for watermark algorithms."""

    def __init__(self, config_path: str, *args, **kwargs) -> None:
        # Load config file
        self.cfg_dict = (
            load_config_file(f'configs/attack/{self.algorithm_name}.json')
            if config_path is None
            else load_config_file(config_path)
        )
        # Update config with kwargs
        if kwargs:
            self.cfg_dict.update(kwargs)
        # Initialize algorithm-specific parameters
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        """Initialize algorithm-specific parameters. Should be overridden by subclasses."""
        raise NotImplementedError

    @property
    def algorithm_name(self) -> str:
        """Return the algorithm name. Should be overridden by subclasses."""
        raise NotImplementedError


class BaseAttack:
    def __init__(self, watermark_model=None, watermark_tokenizer=None, *args, **kwargs) -> None:
        self.watermark_model = watermark_model
        self.watermark_tokenizer = watermark_tokenizer
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        self.output_dir = os.path.join(project_root, 'output')

    def prepare(self) -> bool:
        raise NotImplementedError('prepare method must be implemented in subclasses')

    def get_attack_cache_dir(self) -> str:
        path = os.path.join(self.output_dir, self.cfg.algorithm_name)
        os.makedirs(path, exist_ok=True)
        return path

    def learn(self, training_data, *args, **kwargs) -> bool:
        # Default implementation for attacks that don't need learning
        self.is_trained = True
        return True

    def edit(self, text: str, prompt: str, *args, **kwargs) -> str:
        raise NotImplementedError

    def save_learned_data(self):
        raise NotImplementedError('save_learned_data method must be implemented in subclasses')

    def load_learned_model(self):
        raise NotImplementedError('save_learned_data method must be implemented in subclasses')
