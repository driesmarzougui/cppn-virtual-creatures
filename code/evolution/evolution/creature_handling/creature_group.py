from collections import defaultdict

import ray
from neat import Config, DefaultGenome
from typing import Dict, List, Tuple, Set
import numpy as np
from sortedcontainers import SortedList

from evolution.evolutionary_algorithms.ns import Behavior
from evolution.creature_handling.creature import Creature


@ray.remote
class CreatureGroup(object):
    """Deprecated
    Helper class to encapsulate a group of creatures for simultaneous evaluation.
    """
    def __init__(self, config: Config, params: Dict,
                 behavior_class: Behavior.__class__) -> None:
        self.creatures: Dict[int, Creature] = dict()
        self.config = config
        self.params = params
        self.behavior_class = behavior_class

    def add_creatures(self, genomes: List[Tuple[int, DefaultGenome]]) -> Tuple[
        Set[int], List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        invalid = set()
        complexities = list()
        for genome_id, genome in genomes:
            creature = Creature(genome_id, genome, self.config, self.params, self.behavior_class)
            if creature.net is None:
                invalid.add(genome_id)
            else:
                self.creatures[genome_id] = creature
                complexities.append((creature.genotype_complexity, creature.phenotype_complexity))

        return invalid, complexities

    def terminate_creatures(self, generation: int, genome_ids: List[int]) -> List[Tuple[List[float], float]]:
        return [self.creatures[genome_id].get_behavior_reward(generation) for genome_id in genome_ids]

    def finish_termination(self, generation: int, genome_ids: List[int], genomes_to_save: List[int]) -> None:
        for genome_id in genomes_to_save:
            self.creatures[genome_id].save(generation, overal_best=True)

        for genome_id in genome_ids:
            del self.creatures[genome_id]

    def get_actions_and_set_rewards(self, genomes_obs_rewards: List[Tuple[int, np.ndarray, float]]) -> \
            List[Tuple[int, np.ndarray]]:
        genome_id_actions = list()
        for genome_id, observations, reward in genomes_obs_rewards:
            creature = self.creatures[genome_id]
            creature.add_reward(reward)
            actions = creature.get_actions(observations)
            if actions is not None:
                genome_id_actions.append((genome_id, actions))

        return genome_id_actions


class CreatureGroupHandler(object):
    """Deprecated
    Helper class to load balance creatures over different CreatureGroups.
    This class forms the interface between genomes and creatures.
    """
    
    def __init__(self, num_slaves: int, config: Config, params: Dict, behavior_class: Behavior.__class__) -> None:
        self.num_slaves = num_slaves
        self.config = config
        self.params = params
        self.behavior_class = behavior_class
        self.creature_groups: List[CreatureGroup] = [CreatureGroup.remote(self.config, self.params, self.behavior_class)
                                                     for _ in
                                                     range(self.num_slaves)]
        self.creature_group_workload = np.array([0] * num_slaves)

        self.genome_to_creature_group = dict()

        self._add_genomes_ref: List[ray.ObjectRef] = list()
        self._save_genomes_ref: List[ray.ObjectRef] = list()

    def get_num_creatures(self) -> int:
        return len(self.genome_to_creature_group)

    def get_active_genomes(self) -> Set:
        return set(self.genome_to_creature_group.keys())

    def add_genomes(self, genomes: Dict[int, DefaultGenome]) -> None:
        creature_group_to_genomes = defaultdict(list)

        # Load balance the given genomes over creature groups
        for genome_id, genome in genomes.items():
            target_group_idx = np.argmin(self.creature_group_workload)  # todo: optimise this by keeping a sorted list
            creature_group_to_genomes[target_group_idx].append((genome_id, genome))
            self.genome_to_creature_group[genome_id] = target_group_idx
            self.creature_group_workload[target_group_idx] += 1

        # Add genomes to creature groups based on load balancing result
        self._add_genomes_ref = [self.creature_groups[creature_group_idx].add_creatures.remote(w_genome_ids) for
                                 creature_group_idx, w_genome_ids in creature_group_to_genomes.items()]

    def finish_add_genomes(self) -> Tuple[set, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
        """
        Stalls until all creatures have been made (genotype -> phenotype mappings).
        :return: (set of genome ids with invalid phenotypes, tuples of valid genomes (#nodes, #connections))
        """
        invalids, complexities = zip(*(ray.get(self._add_genomes_ref)))
        invalid_genomes = set().union(*invalids)
        complexities = [c for cg_comp in complexities for c in cg_comp]  # flatten complexities

        # Remove invalid genomes from workgroups state
        for genome_id in invalid_genomes:
            target_group_idx = self.genome_to_creature_group.pop(genome_id)
            self.creature_group_workload[target_group_idx] -= 1

        return invalid_genomes, complexities

    def terminate_genomes(self, generation: int, genome_ids: List[int], current_best: SortedList) -> Dict[
        int, Tuple[List[float], float]]:
        creature_group_to_genomes = defaultdict(list)

        # Set genomes to be removed for every workgroup
        for genome_id in genome_ids:
            target_group_idx = self.genome_to_creature_group.pop(genome_id)
            creature_group_to_genomes[target_group_idx].append(genome_id)
            self.creature_group_workload[target_group_idx] -= 1

        # Send creature termination tasks to every workgroup
        behaviors_rewards_per_creature_group = ray.get(
            [self.creature_groups[group_idx].terminate_creatures.remote(generation, w_genome_ids) for
             group_idx, w_genome_ids in creature_group_to_genomes.items()])

        # Finish termination per creature group
        finish_termination_refs = list()

        # Convert to dictionary mapping genome id to (behavior, reward)
        genome_to_behavior_reward = dict()
        for w_behaviors_rewards, (w_id, w_genome_ids) in zip(behaviors_rewards_per_creature_group,
                                                             creature_group_to_genomes.items()):
            genomes_to_save = list()
            for genome_id, (behavior, reward) in zip(w_genome_ids, w_behaviors_rewards):
                genome_to_behavior_reward[genome_id] = (behavior, reward)

                # Get position in list of current best
                rank = current_best.bisect_left(reward)

                if rank < len(current_best):
                    # Replace value at that position with new (higher) best
                    current_best.pop(rank)
                    current_best.add(reward)
                    genomes_to_save.append(genome_id)

            finish_termination_refs.append(
                self.creature_groups[w_id].finish_termination.remote(generation, w_genome_ids, genomes_to_save))

        # Make sure terminations are all finished
        ray.get(finish_termination_refs)

        return genome_to_behavior_reward

    def get_actions_set_rewards(self, genome_obs_reward: Dict[int, Tuple[np.ndarray, float]]) -> List[ray.ObjectRef]:
        creature_groups_to_genome_obs_reward = defaultdict(list)

        # Set genomes for every workgroup
        for genome_id, (obs, reward) in genome_obs_reward.items():
            target_group_idx = self.genome_to_creature_group[genome_id]
            creature_groups_to_genome_obs_reward[target_group_idx].append((genome_id, obs, reward))

        # Send action inference and reward setting tasks to every workgroup
        return [self.creature_groups[group_idx].get_actions_and_set_rewards.remote(genomes_obs_rewards) for
                group_idx, (genomes_obs_rewards) in creature_groups_to_genome_obs_reward.items()]
