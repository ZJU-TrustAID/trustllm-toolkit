# src/evaluation/launcher.py
from src.config import ExperimentConfig
from src.evaluation.loggers.composite import CompositeExperimentLogger
from src.evaluation.loggers.swanlab_logger import SwanLabExperimentLogger
from src.evaluation.loggers.null_logger import NullExperimentLogger
from src.evaluation.exp_runner import ExperimentRunner

def build_exp_logger(cfg: ExperimentConfig):
    loggers = []
    
    # Check if SwanLab is enabled/configured
    # Assuming if project is set, we use it.
    if cfg.runner_settings.swanlab_project:
        loggers.append(SwanLabExperimentLogger(
            project=cfg.runner_settings.swanlab_project
        ))
    
    # Add other loggers here (e.g. JSONL, Console)
    
    if not loggers:
        return NullExperimentLogger()
    
    if len(loggers) == 1:
        return loggers[0]
        
    return CompositeExperimentLogger(loggers)

def run_experiment(config: ExperimentConfig):
    exp_logger = build_exp_logger(config)
    runner = ExperimentRunner(config, exp_logger=exp_logger)
    runner.execute_pipeline()
