from collections import defaultdict
from typing import List, Tuple

import torch
from tensorboardX import SummaryWriter
from os import listdir

from neat.reporting import BaseReporter

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys
import numpy as np

from evolution.creature_handling.creature import CreatureEvaluationResult
from evolution.evolutionary_algorithms.cma_es_cvt_map_elites.cma_archive import CMAArchive


class TBLogger(BaseReporter):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        self.writer = SummaryWriter(logdir=log_dir)
        self.generation = None
        self.world_state = defaultdict(float)
        self.archive_prev_times_rebalanced = -1
        self.archive_prev_occupancy = -1

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.
        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        self.writer.add_scalar(tag, value, step)

    def start_generation(self, generation):
        self.generation = generation
        self.log_world_state()

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)

        self.log_scalar(f"Novelty/Best_novelty", best_genome.fitness, self.generation)
        self.log_scalar(f"Novelty/Mean_novelty", fit_mean, self.generation)
        self.log_scalar(f"Novelty/Standard_deviation", fit_std, self.generation)

        self.log_world_state()
        self.writer.flush()

    def log_fitnesses(self, fitnesses):
        self.log_scalar(f"Fitness/Best_fitness", max(fitnesses), self.generation)
        self.log_scalar(f"Fitness/Mean_fitness", np.mean(fitnesses), self.generation)
        self.log_scalar(f"Fitness/Standard_deviation", np.std(fitnesses), self.generation)

    def log_archive(self, archive):
        self.log_scalar(f"Archive/Size", len(archive.data), self.generation)
        self.log_scalar(f"Archive/Threshold", archive.threshold, self.generation)

    def log_archive_updates(self, occupancy: float, novel: float, curiosity: Tuple[float, float], replaced: float,
                            times_rebalanced: int,
                            rebalance_progress: float, archive_unique_behaviors: int,
                            archive_MAP_state: Tuple[np.ndarray, np.ndarray]):
        self.log_scalar(f"Archive/Occupancy", occupancy, self.generation)
        self.log_scalar(f"Archive/Novel", novel, self.generation)
        self.log_scalar(f"Archive/Replacements", replaced, self.generation)
        self.log_scalar(f"Archive/Times_Rebalanced", times_rebalanced, self.generation)
        self.log_scalar(f"Archive/Rebalance_Progress", rebalance_progress, self.generation)
        self.log_scalar(f"Archive/Unique_Behaviors", archive_unique_behaviors, self.generation)
        self.log_scalar(f"Archive/Curiosity_Mean", curiosity[0], self.generation)
        self.log_scalar(f"Archive/Curiosity_Max", curiosity[1], self.generation)

        if occupancy > self.archive_prev_occupancy or times_rebalanced > self.archive_prev_times_rebalanced:
            # Only create the embedding if something significant changed
            self.writer.add_embedding(
                torch.tensor(archive_MAP_state[0]),
                metadata=torch.tensor(archive_MAP_state[1]),
                global_step=self.generation,
            )
            self.archive_prev_occupancy = occupancy
            self.archive_prev_times_rebalanced = times_rebalanced

    def log_world_state(self) -> None:
        for obj, state in self.world_state.items():
            self.log_scalar(f"WorldState/{obj}", state, self.generation)

    def log_steps(self, steps: List[int], best_genome: int) -> None:
        steps_mean = np.mean(steps)
        steps_std = np.std(steps)
        self.log_scalar(f"Agents/Num_steps_mean", steps_mean, self.generation)
        self.log_scalar(f"Agents/Num_steps_std", steps_std, self.generation)
        self.log_scalar(f"Agents/Highest_fitness_steps", best_genome, self.generation)

    def log_invalid(self, invalid_ratio_morph: float, invalid_ratio_brain: float) -> None:
        self.log_scalar(f"Agents/Invalid_ratio_morphology", invalid_ratio_morph, self.generation)
        self.log_scalar(f"Agents/Invalid_ratio_brain", invalid_ratio_brain, self.generation)

    def log_phenotype_validity(self, phenotype_validity: List[Tuple[bool, bool]]) -> None:
        morph_validity, brain_validity = zip(*phenotype_validity)
        pop_size = len(phenotype_validity)

        num_valid_morphs = sum(morph_validity) / pop_size
        num_valid_brains = (sum(brain_validity) / sum(morph_validity)) if num_valid_morphs > 0 else 0
        self.log_scalar(f"Agents/Valid_morph_ratio", num_valid_morphs, self.generation)
        self.log_scalar(f"Agents/Valid_brain_ratio", num_valid_brains, self.generation)

    def log_brain_complexities(self, type: str, complexities: List[Tuple[int, int]]) -> None:
        if len(complexities) > 0:
            nodes, connections = zip(*complexities)
            mean_nodes = np.mean(nodes)
            std_nodes = np.std(nodes)

            mean_connections = np.mean(connections)
            std_connections = np.std(connections)

            self.log_scalar(f"Complexity/{type}/Num_nodes_mean", mean_nodes, self.generation)
            self.log_scalar(f"Complexity/{type}/Num_nodes_std", std_nodes, self.generation)
            self.log_scalar(f"Complexity/{type}/Num_connections_mean", mean_connections, self.generation)
            self.log_scalar(f"Complexity/{type}/Num_connections_std", std_connections, self.generation)

    def log_oscillatory_used(self, oscillatory_used: List[Tuple[bool, bool]]) -> None:
        if len(oscillatory_used) > 0:
            sin_used, cos_used = list(zip(*oscillatory_used))
            pop_size = len(oscillatory_used)

            self.log_scalar(f"Oscillatory_used/sin", sum(sin_used) / pop_size, self.generation)
            self.log_scalar(f"Oscillatory_used/cos", sum(cos_used) / pop_size, self.generation)

    def log_cma_archive(self, cma_archive: CMAArchive) -> None:
        self.log_scalar("CMA_Archive/Diversity", len(cma_archive.cma_instance_archive), self.generation)
        self.log_scalar("CMA_Archive/Depth_max", max(cma_archive.cmp_str_to_num_iterations.values()), self.generation)
        self.log_scalar("CMA_Archive/Depth_mean", float(np.mean(list(cma_archive.cmp_str_to_num_iterations.values()))),
                        self.generation)
        self.log_scalar("CMA_Archive/Evals_max", max(cma_archive.cmp_str_to_num_evals.values()), self.generation)
        self.log_scalar("CMA_Archive/Evals_mean", float(np.mean(list(cma_archive.cmp_str_to_num_evals.values()))),
                        self.generation)

    def log_generation_results(self, generation: int, archive_occupancy: float, archive_novel: float,
                               archive_replaced: float, archive_times_rebalanced: int,
                               archive_curiosity: Tuple[float, float],
                               archive_unique_behaviors: int,
                               archive_rebalance_progress: float,
                               archive_MAP_state: Tuple[np.ndarray, np.ndarray],
                               archive_fitnesses: List[float],
                               archive_complexities: Tuple[List[Tuple[int, int]], List[Tuple[int, int]]],
                               archive_steps: List[int],
                               cma_archive: CMAArchive,
                               evaluation_results: List[CreatureEvaluationResult]):
        phenotype_validity = [eval_result.valid for eval_result in evaluation_results]

        self.generation = generation
        self.log_fitnesses(archive_fitnesses)
        self.log_brain_complexities("genotype", archive_complexities[0])
        self.log_brain_complexities("phenotype", archive_complexities[1])
        self.log_phenotype_validity(phenotype_validity)
        self.log_steps(archive_steps, 0)
        self.log_archive_updates(archive_occupancy, archive_novel, archive_curiosity, archive_replaced,
                                 archive_times_rebalanced,
                                 archive_rebalance_progress, archive_unique_behaviors, archive_MAP_state)
        self.log_cma_archive(cma_archive)

    @staticmethod
    def get_unique_experiment_name(directory):
        entries = listdir(directory)
        return sum([x.startswith("run") for x in entries])
