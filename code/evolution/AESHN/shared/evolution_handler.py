"""Implements the core evolution algorithm."""
from __future__ import print_function

from collections import defaultdict
from typing import List, Tuple, Dict, Set, Union
import logging

import sortednp
from neat import DefaultGenome, DefaultSpeciesSet
from neat.config import Config
from neat.reporting import ReporterSet, BaseReporter
from neat.math_util import mean
import numpy as np
from sortedcontainers import SortedList

from AESHN.shared import TBLogger
from evolution.evolutionary_algorithms.ns.archive import Archive


class CompleteExtinctionException(Exception):
    pass


class EvolutionHandler(object):
    """
    This class implements the core evolution algorithm:
        1. Evaluate fitness of all genomes.
        2. Check to see if the termination criterion is satisfied; exit if it is.
        3. Generate the next generation from the current population.
        4. Partition the new generation into species based on genetic similarity.
        5. Go to 1.
    """

    def __init__(self, config: Config, params: Dict, tb_logger: TBLogger, checkpointer: object,
                 checkpoint: Tuple[
                     Dict[int, DefaultGenome], DefaultSpeciesSet, ReporterSet, int, Dict, SortedList] = None) -> None:
        self.reporters = ReporterSet()
        self.config = config
        self.params = params
        self.tb_logger = tb_logger
        self.invalid_behavior = [params["ns_params"]["behavior_dummy"]] * params["ns_params"]["behavior_size"]
        self.invalid_fitness = params["fitness_params"]["dummy"]
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = config.reproduction_type(config.reproduction_config,
                                                     self.reporters,
                                                     stagnation)
        self.checkpointer = checkpointer

        self.current_best = params["experiment"]["save_n_best"]["min_fitness_threshold"]

        self.novelty_search_mode = self.params["experiment"]["novelty_search"]
        if config.fitness_criterion == 'max':
            self.fitness_criterion = max
        elif config.fitness_criterion == 'min':
            self.fitness_criterion = min
        elif config.fitness_criterion == 'mean':
            self.fitness_criterion = mean
        elif not config.no_fitness_termination:
            raise RuntimeError(
                "Unexpected fitness_criterion: {0!r}".format(config.fitness_criterion))

        if checkpoint is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.population_behaviors = dict()
            self.population_rewards = dict()
            self.population_steps = defaultdict(int)
            self.invalid_genomes_morph = set()
            self.invalid_genomes_brain = set()
            self.brain_complexities = list()
            self.oscillatory_used = list()
            self.species = config.species_set_type(config.species_set_config, self.reporters)
            self.generation = 0
            self.species.speciate(config, self.population, self.generation)
            self.evaluated = set()
            self.under_evaluation = set()
            self.archive = Archive(self.params["ns_params"])
        else:
            logging.info("Starting from checkpoint!")
            self.population, self.species, self.reporters, self.generation, archive_restoration, self.current_best = checkpoint
            self.archive = Archive(self.params["ns_params"], archive_restoration)

        self.best_genome = None

    def add_reporter(self, reporter: BaseReporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter: BaseReporter) -> None:
        self.reporters.remove(reporter)

    def start(self):
        self.reporters.start_generation(self.generation)

    def generation_step(self) -> None:
        # Gather and report statistics.
        best = None
        best_idx = None
        for idx, g in self.population.items():
            if best is None or g.fitness > best.fitness:
                best = g
                best_idx = idx

        self.reporters.post_evaluate(self.config, self.population, self.species, best)
        self.tb_logger.log_brain_complexities(self.brain_complexities)
        self.tb_logger.log_oscillatory_used(self.oscillatory_used)
        try:
            self.tb_logger.log_invalid(len(self.invalid_genomes_morph) / len(self.population),
                                       len(self.invalid_genomes_brain) / (
                                               len(self.population) - len(self.invalid_genomes_morph)))
        except ZeroDivisionError:
            self.tb_logger.log_invalid(len(self.invalid_genomes_morph) / len(self.population),
                                       1.0)
        self.tb_logger.log_steps(list(self.population_steps.values()), self.population_steps[best_idx])
        self.tb_logger.log_fitnesses(list(self.population_rewards.values()))
        self.tb_logger.log_archive(self.archive)

        # Track the best genome ever seen.
        if self.best_genome is None or best.fitness > self.best_genome.fitness:
            self.best_genome = best

        # Create the next generation from the current generation.
        self.population = self.reproduction.reproduce(self.config, self.species,
                                                      self.config.pop_size, self.generation)

        # Check for complete extinction.
        if not self.species.species:
            self.reporters.complete_extinction()

            # If requested by the user, create a completely new population,
            # otherwise raise an exception.
            if self.config.reset_on_extinction:
                self.population = self.reproduction.create_new(self.config.genome_type,
                                                               self.config.genome_config,
                                                               self.config.pop_size)
            else:
                raise CompleteExtinctionException()

        # Divide the new population into species.
        self.species.speciate(self.config, self.population, self.generation)

        self.reporters.end_generation(self.config, self.population, self.species)

        self.generation += 1
        self.reporters.start_generation(self.generation)

        self.evaluated = set()
        self.under_evaluation = set()
        self.population_behaviors = dict()
        self.population_rewards = dict()
        self.population_steps = defaultdict(int)
        self.invalid_genomes_morph = set()
        self.invalid_genomes_brain = set()
        self.brain_complexities = list()
        self.oscillatory_used = list()

        if self.checkpointer.check_checkpoint_due(self.generation):
            # Checkpoint!
            self.checkpointer.checkpoint(self.config, self.population, self.species, self.archive.get_restoration(),
                                         self.current_best)

    def add_finished_genomes(self, finished_genomes: Dict[int, Tuple[List[float], float]]) -> bool:
        # Set fitnesses of newly finished
        for idx, (behavior, reward) in finished_genomes.items():
            if idx in self.under_evaluation:
                self.population_behaviors[idx] = behavior
                self.population_rewards[idx] = reward
                self.calculate_and_set_intermediate_score(idx)
                self.under_evaluation.remove(idx)
                self.evaluated.add(idx)
            # else: finished genome was reused and its fitness has already been set

        if len(self.evaluated) == len(self.population):
            # All genome fitnesses have been set -> next generation!
            self.calculate_and_set_scores()
            self.generation_step()
            return True

        return False

    def add_brain_complexities(self, complexities: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> None:
        self.brain_complexities.extend(complexities)

    def add_invalid_genomes(self, invalid_genomes: Set[int]) -> None:
        for idx in invalid_genomes:
            if idx in self.under_evaluation:
                self.population[idx].fitness = -1
                self.under_evaluation.remove(idx)
                self.evaluated.add(idx)
                self.invalid_genomes.add(idx)
            else:
                assert False, "shouldn't happen"

    def calculate_and_set_intermediate_score(self, idx: int) -> None:
        if self.novelty_search_mode:
            behavior = np.array(self.population_behaviors[idx])
            archive_distance, _ = Archive.query(self.archive.archive, behavior)
            archive_distance = [x for x in archive_distance if x != -1]
            if len(archive_distance) > 0:
                score = float(np.mean(archive_distance))
            else:
                score = 1
        else:
            score = self.population_rewards[idx]
        self.population[idx].fitness = score

    def calculate_and_set_scores(self) -> None:
        if self.novelty_search_mode:
            # Create behavior archive of current population
            genome_ids, behaviors = list(zip(*self.population_behaviors.items()))
            behaviors = np.array(behaviors)
            pop_archive = Archive.build_archive(data=behaviors,
                                                ns_params=self.params["ns_params"])

            pop_distances, _ = Archive.query(pop_archive, behaviors)
            archive_distances, _ = Archive.query(self.archive.archive, behaviors)

            current_archive = self.archive.data
            num_added = 0

            for genome_id, pop_ds, archive_ds in zip(genome_ids, pop_distances, archive_distances):
                try:
                    # Filter out bad values
                    pop_ds = pop_ds[:np.where(pop_ds == -1)[0][0]]
                except IndexError:
                    pass
                try:
                    # Filter out bad values
                    archive_ds = archive_ds[:np.where(archive_ds == -1)[0][0]]
                except IndexError:
                    pass

                # Set agent novelty (fitness) as the mean distance of the agent's behavior to the most similar behaviors of other agents
                ds = sortednp.merge(pop_ds, archive_ds)[:self.params["ns_params"]["k"]]
                novelty = float(np.mean(ds))
                self.population[genome_id].fitness = novelty

                if novelty > self.archive.threshold:
                    current_archive.append(self.population_behaviors[genome_id])
                    num_added += 1

            # Update archive
            self.archive.update(current_archive, num_added, len(self.population))
        else:
            for genome_id, reward in self.population_rewards.items():
                if reward > self.current_best:
                    self.current_best = reward
                self.population[genome_id].fitness = reward

    def query_new_sub_population(self, amount: int, disallowed: Set = None) -> Dict[int, DefaultGenome]:
        if disallowed is None:
            disallowed = set()

        genomes_added = 0
        new_sub_population = dict()

        # Give priority to genomes that haven't been evaluated yet
        unevaluated_genomes = self.get_unevaluated_genomes()

        while genomes_added < amount and unevaluated_genomes:
            genome_id = unevaluated_genomes.pop()
            if genome_id not in disallowed:
                new_sub_population[genome_id] = self.population[genome_id]
                self.under_evaluation.add(genome_id)
                genomes_added += 1

        if genomes_added < amount and self.params["experiment"]["reinsert_evaluated"]:
            # Fill up remaining space with the current best (most novel behavior) genomes of the population
            evaluated_genomes = sorted(self.evaluated, key=lambda idx: self.population[idx].fitness)
            while genomes_added < amount and evaluated_genomes:
                genome_id = evaluated_genomes.pop()
                if genome_id not in disallowed and genome_id not in self.invalid_genomes:
                    new_sub_population[genome_id] = self.population[genome_id]
                    genomes_added += 1

        return new_sub_population

    def get_unevaluated_genomes(self) -> Set[int]:
        return set(self.population.keys()) - self.evaluated - self.under_evaluation

    def new_genome_created(self):
        # todo: when a new genome has been naturally created by sexual reproduction of agents, replace the current lowest genome of the population with this one -> always favor natural creations
        raise NotImplementedError()


def create_evolution_handler(config: Config, params: Dict,
                             loggers: List[Union[TBLogger, BaseReporter]],
                             checkpoint_path: str = None,
                             ) -> EvolutionHandler:
    from evolution.utils.checkpointer import Checkpointer
    checkpointer = Checkpointer(generation_interval=params["experiment"]["checkpoint_interval"],
                                time_interval_seconds=None,
                                filename_prefix=f"{params['experiment']['results_dir']}/checkpoints/")

    if checkpoint_path is not None:
        eh = Checkpointer.restore_checkpoint(checkpoint_path, params, tb_logger=loggers[0], checkpointer=checkpointer,
                                             loggers=loggers)
    else:
        eh = EvolutionHandler(config=config, params=params, tb_logger=loggers[0], checkpointer=checkpointer,
                              checkpoint=None)
        for logger in loggers:
            eh.add_reporter(logger)

    return eh
