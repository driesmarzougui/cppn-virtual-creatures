import logging
from typing import List, Dict

from AESHN.shared.tensorboard_logger import TBLogger
from neat import Config, DefaultGenome, DefaultReproduction
from neat.reporting import BaseReporter
from ray.util import ActorPool
import random
import copy
from evolution.evolutionary_algorithms.cvt_me.cvt import CVTArchive
from evolution.evolutionary_algorithms.ns import Behavior
from itertools import count
from tqdm import tqdm

from evolution.evolutionary_algorithms.cma_es_cvt_map_elites.cma_archive import CMAArchive


class GenerationLoggingInfo(object):
    def __init__(self):
        self.fitnesses = None
        self.complexities = None
        self.oscillatory_used = None
        self.phenotype_validity = None
        self.archive_occupancy = None


def generate_offspring(archive: CVTArchive, amount: int, curiosity_selection: bool, to_reevaluate: Dict,
                       num_sexual, genome_indexer, config: Config, genome_generator: DefaultReproduction):
    logging.info(f"\tSelecting parent genomes from archive...")
    num_parents = min(amount, 4 * archive.num_occupied())
    selected_archive_genomes = archive.genome_selection(num_parents, curiosity=curiosity_selection)

    # Generate offspring
    population: Dict[int, DefaultGenome] = to_reevaluate.copy()
    couples = set()
    offspring_id_to_parent_centroids = dict()
    i = 0
    while len(population) < amount and i < len(selected_archive_genomes):
        centroid, genome = selected_archive_genomes[i]
        # Sexual - crossover
        if len(couples) < num_sexual:
            partner_centroid, partner_genome = random.choice(selected_archive_genomes)
            if centroid != partner_centroid:
                couple_key = (min(centroid, partner_centroid), max(centroid, partner_centroid))
                if couple_key not in couples:
                    partner = copy.deepcopy(partner_genome)
                    couples.add(couple_key)
                    child_id = next(genome_indexer)
                    child: DefaultGenome = config.genome_type(child_id)
                    child.configure_crossover(genome, partner, config.genome_config)
                    population[child_id] = child
                    offspring_id_to_parent_centroids[child_id] = [centroid, partner_centroid]

        # Asexual - mutate
        genome.mutate(config.genome_config)
        child_id = next(genome_indexer)
        population[child_id] = genome
        offspring_id_to_parent_centroids[child_id] = [centroid]

        i += 1

    logging.info(f"\t\tTotal offspring:      {len(population)}")
    logging.info(f"\t\t  Sexual offspring:   {len(couples)}")
    logging.info(f"\t\t  Asexual offspring:  {len(population) - len(couples)}")
    logging.info(f"\t\tRe-evaluated genomes: {len(to_reevaluate)}")

    # Push compute to it's limit by adding random new genomes as well
    num_extra_genomes = amount - len(population)
    if num_extra_genomes > 0:
        random_genomes = genome_generator.create_new(config.genome_type, config.genome_config, num_extra_genomes)
        for _, genome in random_genomes.items():
            genome.mutate(config.genome_config)
            population[next(genome_indexer)] = genome
        logging.info(f"\t\tRandom genomes:      {num_extra_genomes}")

    return population, offspring_id_to_parent_centroids


def cvt_map_elites(params: Dict, config: Config, behavior_class: Behavior.__class__, pool: ActorPool, num_slaves: int,
                   loggers: List[BaseReporter]):
    """
    CVT-MAP-Elites evolutionary algorithm: https://arxiv.org/abs/1610.05729
    """
    save_reward_threshold = params["experiment"]["save_n_best"]["min_fitness_threshold"]
    cvt_map_elites_params = params["cvt_me_params"]
    cma_enabled = cvt_map_elites_params["cma_es_weight_optim"]["enabled"]
    curiosity_selection = cvt_map_elites_params["curiosity"]
    tb_logger: TBLogger = loggers[0]

    assert not cma_enabled or not cvt_map_elites_params[
        "noisy_fitness"], "CMA-ES based weight optimization and noisy fitness can't currently be both enabled!"

    # Initialise CVT object with param k
    logging.info("Initializing CVTArchive...")
    archive = CVTArchive(behavior_class, params)
    cma_archive = CMAArchive(params=params, config=config)
    logging.info("\tCVTArchive initialization done!")

    genome_generator = config.reproduction_type(config.reproduction_config, None, None)
    genome_indexer = count(1)

    # Reproduction numbers
    max_num_genomes = num_slaves if cma_enabled else 4 * num_slaves
    min_num_genomes = 0 if cma_enabled else int(0.75 * max_num_genomes)
    current_num_genomes = 0

    sexual_fraction = cvt_map_elites_params["sexual_reproduction_fraction"]
    num_sexual = 0 if cma_enabled else int(sexual_fraction * max_num_genomes)

    # Main loop
    generation = 0
    to_reevaluate = dict()
    offspring_id_to_parent_centroids = dict()

    while True:
        logging.info(f"Generation {generation}")

        if cma_enabled:
            curiosity_selection = not curiosity_selection
        # Generate offspring
        population, tmp_offspring_id_to_parent_centroids = generate_offspring(archive,
                                                                              max_num_genomes - current_num_genomes,
                                                                              curiosity_selection,
                                                                              to_reevaluate, num_sexual, genome_indexer,
                                                                              config, genome_generator)
        offspring_id_to_parent_centroids.update(tmp_offspring_id_to_parent_centroids)

        # CMA-es
        if cma_enabled:
            logging.info(f"\tCMA-ES population conversion...")
            population, offspring_id_to_parent_centroids = cma_archive.convert_population(population,
                                                                                          offspring_id_to_parent_centroids,
                                                                                          genome_indexer)
            logging.info(f"\t\tCMA-ES population conversion done!")

        # Evaluate the selected genomes - submit tasks to workers
        logging.info(f"\tEvaluating {len(population)} genomes...")
        for genome in population.items():
            pool.submit(
                lambda env_runner, genome: env_runner.eval_genome.remote(generation, genome, save_reward_threshold),
                genome)
            current_num_genomes += 1

        # Gather results until a decent amount is ready
        logging.info(f"\tWaiting for {current_num_genomes - min_num_genomes} results...")
        evaluation_results = list()
        with tqdm(total=current_num_genomes - min_num_genomes) as pbar:
            while current_num_genomes > min_num_genomes:
                evaluation_results.append(pool.get_next())
                current_num_genomes -= 1
                pbar.update(1)

        cma_archive.add_to_archive(evaluation_results)
        num_novel, num_improved, to_reevaluate = archive.add_to_archive(evaluation_results,
                                                                        offspring_id_to_parent_centroids)
        logging.info(f"\tGeneration results:")
        logging.info(f"\t\tArchive occupancy:        {archive.num_occupied()} : {archive.occupancy()}%")
        logging.info(f"\t\tNew archive cells filled: {num_novel}")
        logging.info(f"\t\tArchive cells improved:   {num_improved}")

        if cma_enabled or generation % 5 == 0:
            tb_logger.log_generation_results(generation=generation,
                                             archive_occupancy=archive.occupancy(),
                                             archive_novel=num_novel,
                                             archive_replaced=num_improved,
                                             archive_rebalance_progress=archive.get_rebalance_progress(),
                                             archive_times_rebalanced=archive.times_rebalanced,
                                             archive_MAP_state=archive.get_MAP_visualization_data(),
                                             archive_fitnesses=archive.current_performances(),
                                             archive_complexities=archive.current_complexities(),
                                             archive_curiosity=archive.get_curiosity_mean_max(),
                                             archive_steps=archive.current_steps(),
                                             cma_archive=cma_archive,
                                             archive_unique_behaviors=archive.get_amount_of_unique_behaviors(),
                                             evaluation_results=evaluation_results)

        assert not cma_enabled or len(
            offspring_id_to_parent_centroids) == 0, "CMA-ES based weight optimization is enabled, offspring dict should be empty after a generation!"

        generation += 1
