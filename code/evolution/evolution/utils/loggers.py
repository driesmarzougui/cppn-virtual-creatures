import json
import logging
from pathlib import Path
from shutil import copy2
from typing import Dict, List

import neat
from neat.reporting import BaseReporter

from AESHN.shared import TBLogger


def setup_loggers_and_dirs(params: Dict, experiment_name: str) -> List[BaseReporter]:
    """
    Sets up the tensorboard logging and logging / result directories.
    Also copies the given experiment parameters and NEAT config to the logging directory.
    """

    # Setup logger and environment.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_path = Path(params["experiment"]["log_dir"]) / experiment_name
    log_path.mkdir(exist_ok=True, parents=True)
    log_path = str(log_path)

    dir_index = TBLogger.get_unique_experiment_name(log_path)
    run_name = f"run_{dir_index}"

    results_path = Path(params["experiment"]["results_dir"]) / experiment_name / run_name
    (results_path / "overal_best").mkdir(exist_ok=True, parents=True)
    (results_path / "checkpoints").mkdir(exist_ok=True, parents=True)
    results_path = str(results_path)
    params["experiment"]["results_dir"] = results_path

    log_dir = f"{log_path}/{run_name}"
    tb_logger = TBLogger(log_dir)

    # Save params to log dir
    with open(f"{log_dir}/settings.json", "w") as f:
        json.dump(params, f)
    copy2(params["experiment"]["config_path"], f"{log_dir}/neat_config")

    return [tb_logger, neat.reporting.StdOutReporter(True), neat.statistics.StatisticsReporter()]
