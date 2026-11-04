import argparse
import sys
import os
import yaml

project_root = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(project_root, 'src')):
    project_root = os.path.dirname(project_root)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import ExperimentConfig
from src.config.loader import load_and_merge_configs
from src.exceptions.exceptions import InvalidOverrideFormatError, ConflictingPipelineModeError


def _apply_quality_metrics_only(final_config_dict: dict, metric_names: list[str]) -> None:
    normalized_names = [str(name).strip() for name in metric_names if str(name).strip()]
    if not normalized_names:
        raise ValueError('quality_metrics_only requires at least one non-empty metric name.')

    final_config_dict['pipeline_mode'] = 'evaluate'
    eval_cfg = final_config_dict.setdefault('evaluation', {})
    quality_metrics = eval_cfg.get('quality_metrics') or {}
    eval_cfg['quality_metrics'] = {
        metric_name: quality_metrics.get(metric_name) or {} for metric_name in normalized_names
    }
    eval_cfg['skip_detection_in_evaluate'] = True


def main():
    parser = argparse.ArgumentParser(
        description='Run a watermark attack experiment from a set of YAML config files.',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('-c', '--config', type=str, required=True, help='Experiment config path')
    parser.add_argument(
        '-g', '--general-config', type=str, default='configs/general.yaml', help='Base config path'
    )
    parser.add_argument(
        '-o', '--override', nargs='+', action='append', metavar='KEY=VALUE', help='CLI overrides'
    )

    # New arguments for pipeline control
    parser.add_argument(
        '--generate-only',
        action='store_true',
        help='Only generate and attack, save results to file.',
    )
    parser.add_argument(
        '--evaluate-only', action='store_true', help='Only evaluate from saved file.'
    )
    parser.add_argument(
        '--prometheus-only',
        action='store_true',
        help='Evaluate only Prometheus from saved data and reuse existing detection outputs.',
    )
    parser.add_argument(
        '--quality-metrics-only',
        nargs='+',
        metavar='METRIC',
        help='Evaluate only the listed quality metrics from saved data and reuse existing detection outputs.',
    )
    parser.add_argument(
        '--data-path', type=str, help='Path to save/load data for separated execution.'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        help='Reuse an existing sqlite-registered run_id for evaluate/backfill flows.',
    )
    parser.add_argument(
        '--source-run-id',
        type=str,
        help='Bind evaluate/backfill outputs to an existing source run while keeping a new run_id.',
    )

    args = parser.parse_args()

    try:
        cli_overrides = {}
        if args.override:
            flat_overrides = [item for sublist in args.override for item in sublist]
            for override in flat_overrides:
                if '=' not in override:
                    raise InvalidOverrideFormatError(override)
                key, value_str = override.split('=', 1)
                cli_overrides[key.strip()] = yaml.safe_load(value_str)

        final_config_dict = load_and_merge_configs(
            general_path=args.general_config, experiment_path=args.config, overrides=cli_overrides
        )

        # Inject pipeline configuration
        if args.generate_only and (
            args.evaluate_only or args.prometheus_only or args.quality_metrics_only
        ):
            raise ConflictingPipelineModeError()

        if args.prometheus_only and args.quality_metrics_only:
            raise ValueError('Use either --prometheus-only or --quality-metrics-only, not both.')

        if args.prometheus_only:
            _apply_quality_metrics_only(final_config_dict, ['Prometheus'])
        elif args.quality_metrics_only:
            _apply_quality_metrics_only(final_config_dict, args.quality_metrics_only)
        elif args.generate_only:
            final_config_dict['pipeline_mode'] = 'generate'
        elif args.evaluate_only:
            final_config_dict['pipeline_mode'] = 'evaluate'

        if args.data_path:
            final_config_dict['data_path'] = args.data_path

        requested_run_id = str(args.run_id or '').strip()
        requested_source_run_id = str(args.source_run_id or '').strip()

        if requested_run_id or requested_source_run_id:
            if requested_run_id == '' and args.run_id is not None:
                raise ValueError('--run-id must be a non-empty string.')
            if requested_source_run_id == '' and args.source_run_id is not None:
                raise ValueError('--source-run-id must be a non-empty string.')
            if final_config_dict.get('pipeline_mode') != 'evaluate':
                raise ValueError(
                    '--run-id and --source-run-id are only supported in evaluate mode. '
                    'Use --evaluate-only / --prometheus-only / --quality-metrics-only, '
                    'or set pipeline_mode=evaluate in YAML.'
                )
            dashboard_cfg = final_config_dict.get('dashboard')
            if not isinstance(dashboard_cfg, dict):
                dashboard_cfg = {}
                final_config_dict['dashboard'] = dashboard_cfg
            if requested_run_id:
                dashboard_cfg['run_id'] = requested_run_id
            if requested_source_run_id:
                dashboard_cfg['source_run_id'] = requested_source_run_id

        # Create Pydantic Model
        print('[Main] Validating configuration with ExperimentConfig...')
        experiment_config = ExperimentConfig(**final_config_dict)

        # Initialize and run experiment
        print(f'[Main] Initializing ExperimentRunner (Mode: {experiment_config.pipeline_mode})...')
        from src.evaluation.launcher import run_experiment

        run_experiment(experiment_config)

        if experiment_config.pipeline_mode == 'all':
            print(
                f'[Main] Experiment complete. Generated samples are in: {experiment_config.data_path}'
            )

    except FileNotFoundError as e:
        print(f'Configuration Error: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'Critical Error: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
