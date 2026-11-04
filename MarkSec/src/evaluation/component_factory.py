import logging
from src.attack.factory import AutoAttack, AutoAttackConfig
from src.evaluation.threat_models import resolve_attack_model
from src.config import ExperimentConfig, CapabilityRegistry
from src.watermark import load_watermark

logger = logging.getLogger('MarkSec')


class ComponentFactory:
    """
    Initializes Watermark and Attack components.
    """

    def __init__(self, config: ExperimentConfig, device, model_manager):
        self.config = config
        self.device = device
        self.model_manager = model_manager

    def initialize_watermark(self, base_model, base_tokenizer):
        logger.info('  -> Initializing watermark module...')

        wm_name = self.config.watermark
        algorithm_config_path = self.config.watermark_config_path

        # Capability Check
        capability = CapabilityRegistry.get('watermark', wm_name)
        if capability:
            # Add future checks here (e.g. supported models)
            pass

        watermark_instance = load_watermark(
            algorithm_name=wm_name,
            algorithm_config_path=algorithm_config_path,
            model=base_model,
            tokenizer=base_tokenizer,
            device=self.device,
            max_new_tokens=self.config.max_tokens,
            min_new_tokens=self.config.min_len,
        )
        return watermark_instance

    def initialize_attack(self, dataset, watermark_instance):
        """
        Initializes the attack instance based on the configuration.
        """
        logger.info('  -> Initializing attack module...')

        attack_name = self.config.attack
        attack_config_path = f'configs/attack/{attack_name}.json'

        # Capability Check
        capability = CapabilityRegistry.get('attack', attack_name)
        if capability:
            if capability.is_evaluate_only and self.config.pipeline_mode == 'generate':
                logger.warning(
                    f"Attack '{attack_name}' is marked as evaluate-only but pipeline is in generate mode."
                )

        # Identify parameters from the runtime config that should override the attack config.
        # We exclude known keys related to the pipeline infrastructure.
        # With Pydantic, we can check .model_extra or use dict() and filter
        config_dict = self.config.model_dump()
        excluded_keys = {
            'project_root',
            'configs_dir',
            'output_dir',
            'data_dir',
            'models_root',
            'watermark_data',
            'runner_settings',
            'database',
            'evaluation',
            'base_model',
            'dataset',
            'watermark',
            'attack',
            'experiment_name',
            'swanlab_project',
            'pipeline_mode',
            'data_path',
            'max_tokens',
            'min_len',
            'watermark_config_path',
        }
        # Include any extra fields that might have been passed
        override_params = {
            k: v for k, v in config_dict.items() if k not in excluded_keys and v is not None
        }

        # Also check extra fields if they exist
        if self.config.model_extra:
            override_params.update(self.config.model_extra)

        if override_params:
            logger.info(
                f'Overriding attack configuration with runtime parameters: {list(override_params.keys())}'
            )

        attack_cfg_specific = AutoAttackConfig.load(
            attack_name, attack_config_path, **override_params
        )

        base_model_name = self.config.base_model
        base_model, base_tokenizer = self.model_manager.load_model_and_tokenizer(base_model_name)

        attack_dependencies = {
            'base_model': base_model,
            'tokenizer': base_tokenizer,
            'dataset': dataset,
            'watermark_instance': watermark_instance,
            'device': self.device,
        }
        if attack_name in {'MIP', 'JSV', 'B4'}:
            attack_dependencies['watermark_name'] = str(getattr(self.config, 'watermark', '') or '')

        if (
            hasattr(attack_cfg_specific, 'model_requirements')
            and attack_cfg_specific.model_requirements
        ):
            logger.info(f"Attack '{attack_name}' requires special models. Preparing them...")
            base_model_name = self.config.base_model
            tokenizer_requirements = dict(
                getattr(attack_cfg_specific, 'tokenizer_requirements', {}) or {}
            )
            model_parallel_roles = set(
                getattr(attack_cfg_specific, 'model_parallel_roles', []) or []
            )
            role_torch_dtype = getattr(attack_cfg_specific, 'torch_dtype', None)

            for role, model_name in attack_cfg_specific.model_requirements.items():
                original_model_name = model_name
                # Apply threat-model surrogate resolution for attacker roles.
                if role in {'attack', 'attacker'}:
                    resolved_model_name = resolve_attack_model(base_model_name, model_name)
                    if resolved_model_name != model_name:
                        logger.info(
                            f"  [Threat Model] Mapped '{model_name}' -> '{resolved_model_name}' based on victim model '{base_model_name}'."
                        )
                    model_name = resolved_model_name

                tokenizer_name = tokenizer_requirements.get(role, original_model_name)
                if role in {'attack', 'attacker'} and tokenizer_name == original_model_name:
                    tokenizer_name = model_name

                allow_model_parallel = role in model_parallel_roles

                logger.info(
                    "  - Loading/Retrieving model '%s' (tokenizer='%s', model_parallel=%s) for role '%s'.",
                    model_name,
                    tokenizer_name,
                    allow_model_parallel,
                    role,
                )
                model, tokenizer = self.model_manager.load_model_and_tokenizer(
                    model_name,
                    tokenizer_name=tokenizer_name,
                    allow_model_parallel=allow_model_parallel,
                    torch_dtype=role_torch_dtype,
                )

                attack_dependencies[f'{role}_model'] = model
                attack_dependencies[f'{role}_tokenizer'] = tokenizer
                logger.info(f"    -> Model for role '{role}' is ready.")
        else:
            logger.info(f"Attack '{attack_name}' does not require special models.")

        try:
            attack_instance = AutoAttack.load(
                algorithm_name=attack_name,
                algorithm_config=attack_cfg_specific,
                **attack_dependencies,
            )
            # Some attacks might have a `prepare` method (e.g., for stealing).
            if hasattr(attack_instance, 'prepare'):
                attack_instance.prepare()

        except Exception as e:
            logger.error(
                f"Failed to load attack algorithm '{attack_name}'. Error: {e}", exc_info=True
            )
            raise
        logger.info(f"Attack module '{attack_name}' initialized successfully.")
        return attack_instance
