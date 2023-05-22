import copy
from collections import defaultdict
from itertools import count
from typing import List, Dict, Tuple, Iterator

from tqdm import tqdm
import cma
import numpy as np
from neat import DefaultGenome, Config

from evolution.creature_handling.creature import CreatureEvaluationResult
from pytorch_neat.activations import bss
from pytorch_neat.cppn import create_cppn, Node


class CMAArchive(object):
    def __init__(self, params: Dict, config: Config) -> None:
        self.cma_params = params["cvt_me_params"]["cma_es_weight_optim"]
        self.config = config
        self.cma_instance_archive: Dict[
            str, cma.CMAEvolutionStrategy] = dict()  # maps genome comparison strings to their CMA instance
        self.cmp_str_to_genome = dict()

        self.innovation_protection_num_genomes = self.cma_params["innovation_protection_num_genomes"]

        # caches
        self.genome_id_to_cmp_str_cache = dict()
        self.genome_id_to_params_cache = dict()
        self.in_use = set()

        # Logging purposes
        self.cmp_str_to_num_evals = defaultdict(int)
        self.cmp_str_to_num_iterations = defaultdict(int)

    def get_genome_structure_identifier(self, genome: DefaultGenome) -> str:
        cppn: List[Node] = create_cppn(genome, self.config,
                                       ["x1", "y1", "z1", "x2", "y2", "z2", "b_x", "b_y", "b_z", "cl", "bl", "bias"],
                                       ["w", "lr", "A", "B", "C", "D", "M", "FB", "CJ", "LAX",
                                        "HAX", "AYL", "AZL"], output_activation=bss)
        cppn_cmp_strs = [node.get_cmp_str() for node in cppn]
        return "#".join(cppn_cmp_strs)

    def create_cma_instance(self, genome: DefaultGenome, genome_structure: str) -> cma.CMAEvolutionStrategy:
        params = self.parameterize_genome(genome)
        mutate_power = 0.1
        cma_instance = cma.CMAEvolutionStrategy(params, mutate_power,
                                                {'popsize': 4, 'maxiter': 1e15, 'seed': 42, 'bounds': [-30, 30]})
        self.cma_instance_archive[genome_structure] = cma_instance
        self.cmp_str_to_genome[genome_structure] = genome
        return cma_instance

    def convert_population(self, population: Dict[int, DefaultGenome],
                           offspring_to_parent_centroids: Dict[int, List[int]],
                           genome_indexer: Iterator[int]) -> Tuple[Dict[int, DefaultGenome], Dict[int, List[int]]]:
        # Get identifier string for genomes and
        #   Count number of occurences of each genome structure in the given population
        genome_structure_to_genome_ids = defaultdict(list)
        for genome_id, genome in population.items():
            structure_identifier = self.get_genome_structure_identifier(genome)
            genome_structure_to_genome_ids[structure_identifier].append(genome_id)

        # Create the new population
        new_population = dict()
        new_offspring_to_parent_centroids = dict()

        for genome_structure, genome_ids in genome_structure_to_genome_ids.items():
            if genome_structure in self.in_use:
                continue
            # Retrieve CMA instance (creating a new one if it doesn't exist yet for this genome structure)
            if genome_structure in self.cma_instance_archive:
                # Existing genome structure
                number_to_create_factor = 4
                cma_instance = self.cma_instance_archive[genome_structure]
            else:
                # New genome structure
                example_genome = population[genome_ids[0]]
                number_to_create_factor = 8
                cma_instance = self.create_cma_instance(example_genome, genome_structure)

            parents = [offspring_to_parent_centroids.get(genome_id, []) for genome_id in genome_ids]
            parents = [x for y in parents for x in y]

            number_to_create = number_to_create_factor * len(genome_ids)
            new_genome_params = cma_instance.ask(number=number_to_create)
            for new_genome_param in new_genome_params:
                new_genome_id = next(genome_indexer)
                self.genome_id_to_cmp_str_cache[new_genome_id] = genome_structure
                self.genome_id_to_params_cache[new_genome_id] = new_genome_param
                new_population[new_genome_id] = self.reparameterize_genome(genome_structure, new_genome_param)
                new_offspring_to_parent_centroids[new_genome_id] = parents

            self.cmp_str_to_num_evals[genome_structure] += number_to_create
            self.cmp_str_to_num_iterations[genome_structure] += 1
            self.in_use.add(genome_structure)

        # Innovation protection - add the 50 least optimized genomes
        for _ in tqdm(range(self.innovation_protection_num_genomes), desc="Innovation Protection"):
            genome_structure = min(self.cmp_str_to_num_evals, key=self.cmp_str_to_num_evals.get)
            if genome_structure in self.in_use:
                break
            new_genome_params = self.cma_instance_archive[genome_structure].ask(number=4)

            for new_genome_param in new_genome_params:
                new_genome_id = next(genome_indexer)
                self.genome_id_to_cmp_str_cache[new_genome_id] = genome_structure
                self.genome_id_to_params_cache[new_genome_id] = new_genome_param
                new_population[new_genome_id] = self.reparameterize_genome(genome_structure, new_genome_param)
                new_offspring_to_parent_centroids[new_genome_id] = []

            self.cmp_str_to_num_evals[genome_structure] += 4
            self.cmp_str_to_num_iterations[genome_structure] += 1
            self.in_use.add(genome_structure)

        return new_population, new_offspring_to_parent_centroids

    def parameterize_genome(self, genome: DefaultGenome) -> List[float]:
        con_params = [connection.weight for connection in genome.connections.values()]
        node_bias_params = [node.bias for node in genome.nodes.values()]
        node_response_params = [node.response for node in genome.nodes.values()]

        return con_params + node_bias_params + node_response_params

    def reparameterize_genome(self, genome_structure: str, target_parameters: np.ndarray) -> DefaultGenome:
        genome = copy.deepcopy(self.cmp_str_to_genome[genome_structure])

        params = iter(target_parameters)
        for connection in genome.connections.values():
            connection.weight = next(params)

        for node in genome.nodes.values():
            node.bias = next(params)

        for node in genome.nodes.values():
            node.response = next(params)

        return genome

    def add_to_archive(self, evaluation_results: List[CreatureEvaluationResult]) -> None:
        genome_structure_to_params_and_scores = dict()
        for evaluation_result in evaluation_results:
            genome_structure = self.genome_id_to_cmp_str_cache.pop(evaluation_result.genome_id)
            if genome_structure not in genome_structure_to_params_and_scores:
                genome_structure_to_params_and_scores[genome_structure] = ([], [])

            genome_structure_to_params_and_scores[genome_structure][0].append(
                self.genome_id_to_params_cache.pop(evaluation_result.genome_id)
            )
            genome_structure_to_params_and_scores[genome_structure][1].append(
                -evaluation_result.total_reward)  # negate, CMA-ES minimizes

        for genome_structure, (solutions, scores) in tqdm(genome_structure_to_params_and_scores.items(),
                                                          desc="Updating CMA-ES instances..."):
            self.in_use.discard(genome_structure)
            self.cma_instance_archive[genome_structure].tell(solutions, scores)
