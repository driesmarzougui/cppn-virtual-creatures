import logging
from typing import Dict, List, Tuple
import numpy as np
from neat import DefaultGenome
from sklearn.cluster import KMeans
import random
import copy

from evolution.evolutionary_algorithms.ns.behavior import Behavior
from evolution.creature_handling.creature import CreatureEvaluationResult, CreatureComplexity


class CVTArchive(object):
    """Centroidal Voronoi Tessellation
    Class that represents the CVT-archive used in CVT-MAP-Elites.
    """

    def __init__(self, behavior_class: Behavior.__class__, params: Dict):
        self.behavior_class = behavior_class
        self.params = params

        self.automatic_rebalancing_enabled = self.params["cvt_me_params"]["automatic_rebalancing"]
        self.cvt = None

        self.behavior_samples = set()
        self.times_rebalanced = 0

        self.rebalance_threshold_delta = self.params["cvt_me_params"]["rebalance_threshold_scale"] * \
                                         self.params["cvt_me_params"]["k"]
        self.rebalance_threshold = self.rebalance_threshold_delta
        self.random_init()

        self.behavior_archive = dict()  # centroid to behavior
        self.genome_archive = dict()  # centroid to genome
        self.performance_archive = dict()  # centroid to performance
        self.curiosity_archive = dict()  # centroid to curiosity
        self.complexity_archive = dict()  # centroid to complexity
        self.steps_archive = dict()  # centroid to num of steps
        self.evaluated = set()

    def random_init(self):
        behavior_generator = self.behavior_class(self.params["ns_params"]["behavior_dummy"],
                                                 self.params["ns_params"]["behavior_size"],
                                                 self.params["ns_params"][
                                                     "behavior_sample_frequency"])

        random_behaviors = [behavior_generator.random_sample() for _ in
                            range(self.params["cvt_me_params"]["init_samples"])]

        self.cvt = KMeans(n_clusters=self.params["cvt_me_params"]["k"], random_state=42).fit(random_behaviors)

    def rebalance(self):
        if self.automatic_rebalancing_enabled and \
                len(self.behavior_samples) > self.rebalance_threshold:
            logging.info("Automatically rebalancing CVTArchive...")

            # Update CVT
            behavior_samples = [list(sample) for sample in self.behavior_samples]
            self.cvt = KMeans(n_clusters=self.params["cvt_me_params"]["k"], random_state=42).fit(behavior_samples)

            # Update threshold
            self.rebalance_threshold = len(self.behavior_samples) + self.rebalance_threshold_delta
            self.times_rebalanced += 1

            # Remap old centroids to new centroids
            prev_centroids, behaviors = zip(*list(self.behavior_archive.items()))
            new_centroids = self.get_centroids_for_samples(behaviors)

            behavior_archive = dict()
            genome_archive = dict()
            performance_archive = dict()
            curiosity_archive = dict()
            complexity_archive = dict()
            steps_archive = dict()
            for prev_centroid, new_centroid in zip(prev_centroids, new_centroids):
                if new_centroid not in performance_archive or \
                        performance_archive[new_centroid] < self.performance_archive[prev_centroid]:
                    behavior_archive[new_centroid] = self.behavior_archive[prev_centroid]
                    genome_archive[new_centroid] = self.genome_archive[prev_centroid]
                    performance_archive[new_centroid] = self.performance_archive[prev_centroid]
                    curiosity_archive[new_centroid] = self.curiosity_archive[prev_centroid]
                    complexity_archive[new_centroid] = self.complexity_archive[prev_centroid]
                    steps_archive[new_centroid] = self.steps_archive[prev_centroid]

            self.behavior_archive = behavior_archive
            self.genome_archive = genome_archive
            self.performance_archive = performance_archive
            self.curiosity_archive = curiosity_archive
            self.complexity_archive = complexity_archive
            self.steps_archive = steps_archive

            logging.info("\tRebalancing CVTArchive done!")

    def get_rebalance_progress(self) -> float:
        return 1 - (self.rebalance_threshold - len(self.behavior_samples)) / self.rebalance_threshold_delta

    def get_amount_of_unique_behaviors(self) -> int:
        return len(self.behavior_samples)

    def genome_selection(self, amount: int, curiosity: bool = False) -> List[Tuple[int, DefaultGenome]]:
        """
        Evolutionary selection operator.
        Optionally uses curiosity scores to select parents.
        """
        centroids = list(self.genome_archive.keys())
        if curiosity and len(centroids) > 0:
            sample_weights = np.asarray([self.curiosity_archive[centroid] for centroid in centroids])
            sample_weights = sample_weights / np.sum(sample_weights)
        else:
            sample_weights = None
        selected_centroids = random.choices(centroids, k=amount, weights=sample_weights)
        return [(centroid, copy.deepcopy(self.genome_archive[centroid])) for centroid in selected_centroids]

    def add_behavior_sample(self, sample: List) -> None:
        if self.automatic_rebalancing_enabled:
            self.behavior_samples.add(tuple(sample))

    def get_MAP_visualization_data(self) -> Tuple[np.ndarray, np.ndarray]:
        centroids = self.cvt.cluster_centers_

        labels = np.zeros(self.params["cvt_me_params"]["k"])
        for centroid in self.performance_archive:
            labels[centroid] = 1

        return centroids, labels

    def get_centroids_for_samples(self, behavior_samples: List) -> np.ndarray:
        """
        Returns the archive cells for the given behavior samples.
        """
        return self.cvt.predict(behavior_samples)

    def get_curiosity_mean_max(self) -> Tuple[float, float]:
        curiosities = list(self.curiosity_archive.values())
        return float(np.mean(curiosities)), float(np.max(curiosities))

    def num_occupied(self) -> int:
        """
        Returns the amount of occupied archive cells.
        """
        return len(self.genome_archive)

    def occupancy(self) -> float:
        """
        Returns the percentage of archive cells that are occupied.
        """
        return self.num_occupied() / self.params["cvt_me_params"]["k"]

    def current_performances(self) -> List[float]:
        """
        Returns a list of all stored creature performances.
        """
        return list(self.performance_archive.values())

    def current_complexities(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        complexities: List[CreatureComplexity] = list(self.complexity_archive.values())
        genotype = list()
        phenotype = list()
        for complexity in complexities:
            genotype.append(complexity.genotype)
            phenotype.append(complexity.phenotype)
        return genotype, phenotype

    def current_steps(self) -> List[int]:
        return list(self.steps_archive.values())

    def add_to_archive(self,
                       evaluation_results: List[CreatureEvaluationResult],
                       offspring_to_parent_centroids: Dict[int, List[int]]) \
            -> Tuple[int, int, Dict[int, DefaultGenome]]:
        """
        Adds new genomes to the archive based on their evaluation.
        Also updates the curiosity scores of parents stored in the archive.

        If the 'noisy_fitness' parameter is enabled, this function also makes sure that genomes are evaluated twice
        before they replace an archived genome.
        """
        num_novel = 0
        num_improved = 0
        to_reevaluate = dict()

        behaviors = [eval_result.behavior for eval_result in evaluation_results]

        centroids = self.get_centroids_for_samples(behaviors)

        for centroid, eval_result in zip(centroids, evaluation_results):
            self.add_behavior_sample(eval_result.behavior)

            # Add to archive if cell empty or genome has better performance
            add_to_archive = False
            if centroid not in self.genome_archive:
                add_to_archive = True
                num_novel += 1

            if not add_to_archive and self.performance_archive[centroid] < eval_result.total_reward:
                # Possible replacement of genome in archive
                if not self.params["cvt_me_params"]["noisy_fitness"] or eval_result.genome_id in self.evaluated:
                    self.evaluated.discard(eval_result.genome_id)
                    add_to_archive = True
                    num_improved += 1
                else:
                    # Re-evaluate genome to mitigate noisy fitness issues
                    to_reevaluate[eval_result.genome_id] = eval_result.genome
                    self.evaluated.add(eval_result.genome_id)

            if add_to_archive:
                genome = eval_result.genome
                genome.fitness = eval_result.total_reward
                self.genome_archive[centroid] = genome
                self.performance_archive[centroid] = eval_result.total_reward
                self.curiosity_archive[centroid] = 4  # Slightly boost selection for new genomes
                self.complexity_archive[centroid] = eval_result.complexity
                self.steps_archive[centroid] = eval_result.steps
                self.behavior_archive[centroid] = eval_result.behavior

                # Increase parent curiosity score
                try:
                    parent_centroids = offspring_to_parent_centroids.pop(eval_result.genome_id)

                    if len(parent_centroids) > 0:
                        curiosity_reward = 4 / len(parent_centroids)
                        for parent in parent_centroids:
                            if parent != centroid:
                                # Parent centroid might be equal to offspring centroid; but as offspring is likely to generate
                                #   good offspring as well --> always reward curiosity to this centroid
                                self.curiosity_archive[parent] += curiosity_reward
                except KeyError as ex:
                    # Parent was randomly generated
                    pass
            else:
                # Decrease parent curiosity score
                try:
                    parent_centroids = offspring_to_parent_centroids.pop(eval_result.genome_id)
                    if len(parent_centroids) > 0:
                        curiosity_penalty = 1 / len(parent_centroids)
                        for parent in parent_centroids:
                            # Decrease parent curiosity but keep some selection probability
                            self.curiosity_archive[parent] = max(self.curiosity_archive[parent] - curiosity_penalty, 1)
                except KeyError as ex:
                    # Parent was randomly generated
                    pass

        self.rebalance()

        return num_novel, num_improved, to_reevaluate
