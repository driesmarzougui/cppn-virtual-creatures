import logging
import argparse

import psutil
import os

from evolution.creature_handling.env_runner import EnvRunner
from evolution.evolutionary_algorithms.base_ea import base_ea
from evolution.evolutionary_algorithms.cvt_map_elites import cvt_map_elites

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from evolution.utils.config import create_config
from evolution.utils.loggers import setup_loggers_and_dirs

from evolution.utils.params import get_params

import ray
from ray.util import ActorPool


def main(args: argparse.Namespace) -> None:
    """Single Agent Evaluation
    Used for experiments with multiple small environments in which all agents are individually evaluated.
    """

    PARAMS_PATH = args.params
    params = get_params(PARAMS_PATH)
    config = create_config(params)

    ENV_PATH = params["experiment"]["env_path"]

    # Setup loggers and output directories
    logging.info("Setting up loggers and output directories...")
    loggers = setup_loggers_and_dirs(params, args.name)
    logging.info("\tSetting up loggers and output directories done!")

    num_slaves = params["execution"]["n_processes"]
    if num_slaves == -1:
        num_slaves = psutil.cpu_count()

    behavior_class = globals()[params["ns_params"]["behavior"]]

    if args.cluster:
        ray.init(address="auto")
    else:
        ray.init()

    # [SLAVES] Initialise env runners
    env_runners = [
        EnvRunner.remote(worker_id=i + 1, env_path=ENV_PATH, config=config, params=params, behavior=behavior_class) for
        i in range(num_slaves)]

    pool = ActorPool(env_runners)

    logging.info("-" * 20)
    if params["experiment"]["cvt_map_elites"]:
        logging.info("RUNNING CVT_MAP_ELITES")

        cvt_map_elites(params=params, config=config, behavior_class=behavior_class, pool=pool, num_slaves=num_slaves,
                       loggers=loggers)
    else:
        logging.info("RUNNIGN BASIC EA")
        base_ea(params=params, config=config, loggers=loggers, checkpoint_path=args.checkpoint_path, pool=pool)

    logging.info("-" * 20)
