import logging
import argparse
import ray
from evolution.creature_handling.creature_group import CreatureGroupHandler
from AESHN.shared.evolution_handler import create_evolution_handler
import psutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from evolution.utils.config import create_config
from evolution.utils.environment import behavior_name_to_genome, remove_from_env, genome_to_behavior_name, \
    initialise_environment
from evolution.utils.loggers import setup_loggers_and_dirs
from tqdm import tqdm

from evolution.utils.params import get_params

def main(args: argparse.Namespace) -> None:
    """Long Lasting Environment Evaluation
    Used for experiments with one big environment in which all agents co-exist.
    """

    ray.init()
    worker_id = args.worker_id
    PARAMS_PATH = args.params

    params = get_params(PARAMS_PATH)
    config = create_config(params)

    ENV_PATH = params["experiment"]["env_path"]

    # Setup loggers and output directories
    logging.info("Setting up loggers and output directories...")
    loggers = setup_loggers_and_dirs(params, args.name)
    logging.info("\tSetting up loggers and output directories done!")

    # Setup evolution
    logging.info("Setting up evolution handler...")
    evolution_handler = create_evolution_handler(config=config, params=params, loggers=loggers,
                                                 checkpoint_path=args.checkpoint_path)

    logging.info("\tSetting up evolution handler done!")
    logging.info("Creating initial population...")
    initial_population = evolution_handler.query_new_sub_population(config.pop_size)
    logging.info("\tCreating initial population done!")

    num_slaves = params["execution"]["n_processes"]
    if num_slaves == -1:
        num_slaves = psutil.cpu_count()

    behavior_class = globals()[params["ns_params"]["behavior"]]

    ####################################################################################################################
    #                                                      ASYNC                                                       #
    ####################################################################################################################
    creature_group_handler = CreatureGroupHandler(num_slaves, config, params, behavior_class)

    # [SLAVES] Partition genomes over creature groups and create phenotype mappings
    logging.info("[SLAVES] Creating and partitioning creatures over creature groups...")
    creature_group_handler.add_genomes(initial_population)
    invalid_genomes, complexities = creature_group_handler.finish_add_genomes()
    logging.info("\t[SLAVES] Creating and partitioning creatures over creature groups done!")

    assert len(invalid_genomes) < len(initial_population), "All initial genomes were invalid!"
    evolution_handler.add_invalid_genomes(invalid_genomes)
    evolution_handler.add_brain_complexities(complexities)

    # [MASTER] Create environment
    logging.info("[MASTER] Initializing unity environment...")
    env, side_channels = initialise_environment(ENV_PATH, list(set(initial_population.keys()) - invalid_genomes),
                                                tb_logger=loggers[0], worker_id=worker_id)
    logging.info("\t[MASTER] Initializing unity environment done!")

    # ---------------------------------------------------- ASYNC END ---------------------------------------------------

    logging.info("Starting evolution process!")
    evolution_handler.start()

    # Progress bar
    pbar = tqdm(total=len(evolution_handler.population) - len(evolution_handler.invalid_genomes),
                desc=f"Generation {evolution_handler.generation}")

    while True:
        generation = evolution_handler.generation

        finished_genomes = list()
        step_genomes = dict()
        bad_agents = list()

        # Loop over all active agents and split up into ones that require an action and ones that have died
        for agent_id, behavior_spec in env.behavior_specs.items():
            try:
                genome_id = behavior_name_to_genome(agent_id)
                decision_steps, terminal_steps = env.get_steps(agent_id)

                if list(terminal_steps):
                    # Agent has died
                    finished_genomes.append(genome_id)
                elif list(decision_steps):
                    # Create inference task for genome network
                    obs = np.concatenate([ob.flatten() for ob in decision_steps.obs])
                    reward = decision_steps.reward[0]
                    step_genomes[genome_id] = (obs, reward)
                    evolution_handler.population_steps[genome_id] += 1

            except ValueError:
                bad_agents.append(agent_id)

        for agent_id in bad_agents:
            remove_from_env(env, agent_id)

        finished_genomes = creature_group_handler.terminate_genomes(generation, finished_genomes,
                                                                    evolution_handler.current_best)

        ################################################################################################################
        #                                                    ASYNC                                                     #
        ################################################################################################################
        # [SLAVES] Infer action for every alive agent
        grouped_action_refs = creature_group_handler.get_actions_set_rewards(step_genomes)

        num_alive = creature_group_handler.get_num_creatures()
        if finished_genomes:
            pbar.set_postfix(
                {"Evaluated": len(evolution_handler.evaluated),
                 "Evaluating": len(evolution_handler.under_evaluation),
                 "ToEvaluate": len(evolution_handler.get_unevaluated_genomes())})
            pbar.update(len(finished_genomes))

        # [MASTER] Handle dead agents & generation stepping
        generation_finished = evolution_handler.add_finished_genomes(finished_genomes)

        if generation_finished:
            logging.info(f"Generation {generation} finished! Creating new population and creatures...")

        if num_alive < params["experiment"]["min_population_size"] \
                and not side_channels["agent-creator"].creation_requested:

            # Need to add a new set of agents
            num_to_add = params["experiment"]["max_population_size"] - num_alive

            new_sub_population = evolution_handler.query_new_sub_population(num_to_add,
                                                                            disallowed=creature_group_handler.get_active_genomes())

            if new_sub_population:
                # [SLAVES] Partition new genomes over creature groups and create phenotype mappings
                creature_group_handler.add_genomes(new_sub_population)
                invalid_genomes, complexities = creature_group_handler.finish_add_genomes()
                evolution_handler.add_invalid_genomes(invalid_genomes)
                evolution_handler.add_brain_complexities(complexities)

                # Add to environment
                side_channels["agent-creator"].send_request(list(set(new_sub_population.keys()) - invalid_genomes))

                if num_alive == 0:
                    env.reset()
                    while not side_channels["agent-creator"].creation_done:
                        pass

        if generation_finished:
            pbar.close()
            pbar = tqdm(total=len(evolution_handler.population) - len(evolution_handler.invalid_genomes),
                        desc=f"Generation {evolution_handler.generation}")

        # [SLAVES] Get results of action inferences and set actions
        grouped_genome_actions = ray.get(grouped_action_refs)
        for group_genome_actions in grouped_genome_actions:
            for genome_id, actions in group_genome_actions:
                agent_id = genome_to_behavior_name(genome_id)
                env.set_actions(agent_id, actions)

        # -------------------------------------------------- ASYNC END -------------------------------------------------

        env.step()

    pbar.close()
    env.close()


