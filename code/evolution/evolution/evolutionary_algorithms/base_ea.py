import logging
from typing import Dict, List, Optional

from neat import Config
from neat.reporting import BaseReporter
from ray.util import ActorPool

from AESHN.shared.evolution_handler import create_evolution_handler
from evolution.creature_handling.creature import CreatureEvaluationResult

def base_ea(params: Dict, config: Config, loggers: List[BaseReporter], checkpoint_path: str, pool: ActorPool):
    """
    Basic evolutionary algorithm.
    """
    successful_start = False
    while not successful_start:
        # Setup evolution
        logging.info("Setting up evolution handler...")
        evolution_handler = create_evolution_handler(config=config, params=params, loggers=loggers,
                                                     checkpoint_path=checkpoint_path)
        logging.info("\tSetting up evolution handler done!")

        # Start evolution
        logging.info("Starting evolution process!")
        evolution_handler.start()
        while True:
            generation = evolution_handler.generation
            population = evolution_handler.population

            save_reward_threshold = evolution_handler.current_best

            evaluation_results: List[Optional[CreatureEvaluationResult]] = \
                pool.map_unordered(
                    lambda env_runner, genome: env_runner.eval_genome.remote(generation, genome, save_reward_threshold),
                    list(population.items())
                )

            contains_valid_genome = False
            for evaluation_result in evaluation_results:
                genome_id = evaluation_result.genome_id

                evolution_handler.population_behaviors[genome_id] = evaluation_result.behavior
                evolution_handler.population_rewards[genome_id] = evaluation_result.total_reward

                if all(evaluation_result.valid):
                    evolution_handler.population_steps[genome_id] = evaluation_result.steps
                    complexity = evaluation_result.complexity
                    evolution_handler.brain_complexities.append((complexity.genotype, complexity.phenotype))

                    evolution_handler.oscillatory_used.append(evaluation_result.oscillatory_used)

                    successful_start = True
                    contains_valid_genome = True
                else:
                    if evaluation_result.valid[0]:
                        # A valid morphology is also enough to keep going in this direction
                        successful_start = True
                        evolution_handler.invalid_genomes_brain.add(evaluation_result.genome_id)
                    else:
                        evolution_handler.invalid_genomes_morph.add(evaluation_result.genome_id)

            if not contains_valid_genome and not successful_start:
                break

            evolution_handler.calculate_and_set_scores()
            evolution_handler.generation_step()
