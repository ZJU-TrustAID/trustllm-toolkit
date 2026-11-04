# ===============================================================
# src/util/log.py
# Description: This file contains the centralized configuration
#              for the project's logging system.
# ===============================================================

import logging
import os
import time
from typing import Optional


def get_logger(name: str = 'MarkSec') -> logging.Logger:
    """
    Simple wrapper to get the global logger instance without resetting handlers.
    Use this in sub-modules.
    """
    return logging.getLogger(name)


def setup_logger(
    name: str = 'MarkSec',
    log_dir: str = 'output/logs',
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.DEBUG,
    run_name: Optional[str] = None,
    experiment_config: Optional[dict] = None,
) -> logging.Logger:
    """
    Configures the logger. If called multiple times, it will switch the file handler
    to the new location/name, preserving the console handler.

    Args:
        run_name: Explicit filename (e.g., "my_experiment.log").
        experiment_config: Optional dict to auto-generate meaningful names
                           (e.g., "{method}_{model}_{timestamp}.log").
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if run_name:
        filename = run_name
    elif experiment_config:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{timestamp}.log'
    else:
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'run_{timestamp}.log'

    if not filename.endswith('.log'):
        filename += '.log'

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, filename)

    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )

    if not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(file_log_level)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Filter noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('absl').setLevel(logging.WARNING)

    # Silence vLLM and Prometheus logs as requested
    logging.getLogger('vllm').setLevel(logging.ERROR)
    logging.getLogger('prometheus').setLevel(logging.ERROR)
    logging.getLogger('prometheus_eval').setLevel(logging.ERROR)

    logger.info(f'Logger configured. Output file: {log_file_path}')

    return logger
