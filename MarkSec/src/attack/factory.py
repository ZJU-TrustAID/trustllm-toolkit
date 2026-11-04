# ===============================================================
# src/attack/factory.py
# Description: This file contains the factory class for creating
#              attack classes.
# ===============================================================
from typing import Any, Optional
import importlib

# Assuming BaseConfig is defined in src.attack.base
from src.attack.base import BaseConfig
from src.exceptions.exceptions import InvalidAlgorithmNameError, FactoryInstantiationError

# use init to expose the classes instead of importing from each file
CONFIG_MAPPING_NAMES = {
    'DIPPER': 'src.attack.dipper.DipperParaphraserConfig',
    'MIP': 'src.attack.mip.MIPConfig',
    'SA': 'src.attack.smoothing.SmoothingConfig',
    'SCTS': 'src.attack.scts.SCTSConfig',
    'SIRA': 'src.attack.sira.SIRAConfig',
    'B4': 'src.attack.b4.B4Config',
    'RW': 'src.attack.rw.RandomWalkConfig',
    'JSV': 'src.attack.jsv.JSVConfig',
    'LLMP': 'src.attack.llmp.llmp.LLMParaphraseConfig',
    'DEMARK': 'src.attack.demark.demark.DemarkConfig',
}
ATTACK_MAPPING_NAMES = {
    'DIPPER': 'src.attack.dipper.DipperParaphraser',
    'MIP': 'src.attack.mip.MIPAttack',
    'SA': 'src.attack.smoothing.SmoothingAttack',
    'SCTS': 'src.attack.scts.SCTSAttack',
    'SIRA': 'src.attack.sira.SIRAAttack',
    'B4': 'src.attack.b4.B4Attack',
    'RW': 'src.attack.rw.RandomWalkAttack',
    'JSV': 'src.attack.jsv.JSVAttack',
    'LLMP': 'src.attack.llmp.llmp.LLMParaphraseAttack',
    'DEMARK': 'src.attack.demark.demark.DemarkAttack',
}


def config_name_from_alg_name(name: str) -> Optional[str]:
    """Get the config class name from the algorithm name."""
    if name in CONFIG_MAPPING_NAMES:
        return CONFIG_MAPPING_NAMES[name]
    else:
        raise InvalidAlgorithmNameError(name)


def attack_name_from_alg_name(name: str) -> Optional[str]:
    """Get the attack class name from the algorithm name."""
    if name in ATTACK_MAPPING_NAMES:
        return ATTACK_MAPPING_NAMES[name]
    else:
        raise InvalidAlgorithmNameError(name)


class AutoAttackConfig:
    """Factory for loading configuration classes."""

    def __init__(self):
        raise FactoryInstantiationError('AutoAttackConfig')

    @classmethod
    def load(cls, algorithm_name: str, algorithm_config_path=None, **kwargs) -> Any:
        config_name = config_name_from_alg_name(algorithm_name)
        if config_name is None:
            raise InvalidAlgorithmNameError(algorithm_name)

        module_name, class_name = config_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        config_class = getattr(module, class_name)

        if algorithm_config_path is None:
            default_path = f'configs/attack/{algorithm_name}.json'
            print(f'Algorithm config path is None, using default: {default_path}')
            algorithm_config_path = default_path

        # This part will be updated when we switch to dataclasses
        config_instance = config_class(algorithm_config_path, **kwargs)
        return config_instance


class AutoAttack:
    """Factory for creating attack instances."""

    def __init__(self):
        raise FactoryInstantiationError('AutoAttack')

    @staticmethod
    def load(
        algorithm_name: str,
        algorithm_config: BaseConfig,  # Expect a config object
        **kwargs: Any,
    ) -> Any:  # Use flexible kwargs for dependencies
        """
        Loads an attack instance, injecting all necessary dependencies.

        Args:
            algorithm_name (str): The name of the attack.
            algorithm_config (BaseConfig): The configuration object for the attack.
            **kwargs: A dictionary of dependencies (e.g., models, tokenizers)
                      to be passed to the attack's constructor.
        """
        attack_name = attack_name_from_alg_name(algorithm_name)
        module_name, class_name = attack_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        attack_class = getattr(module, class_name)

        # The factory now passes the config object and all other keyword arguments
        # directly to the attack class's constructor.
        attack_instance = attack_class(config=algorithm_config, **kwargs)

        return attack_instance
