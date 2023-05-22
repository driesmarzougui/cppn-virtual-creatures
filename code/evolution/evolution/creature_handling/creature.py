import pickle
from pathlib import Path
from typing import Dict, Optional

from neat import DefaultGenome, Config

from AESHN.shared.cppn.cppn import CPPN
# from AESHN.shared.ctrnn.ctrnn_handler import CTRNNHandler
from AESHN.shared.visualize import draw_net, draw_trajectory
from AESHN.es_hyperneat import ESNetwork
import numpy as np

from MORPH.CPPNMorph import CPPNMorph


class CreatureComplexity(object):
    """
    Helper class to hold creature phenotype and genotype complexities (amount of nodes and complexities).
    """

    def __init__(self, genotype_complexity: Tuple[int, int], phenotype_complexity: Tuple[int, int]) -> None:
        self.genotype = genotype_complexity
        self.phenotype = phenotype_complexity


class CreatureEvaluationResult(object):
    """
    Helper class to hold creature evaluation results.
    """

    def __init__(self, valid: Tuple[bool, bool], genome_id: int, genome: DefaultGenome, total_reward: int, steps: int, behavior: List[float],
                 complexity: CreatureComplexity, oscillatory_used: Tuple[bool, bool] = (False, False)) -> None:
        """
        :param valid: (valid morphology, valid brain)
        :param genome_id:
        :param total_reward:
        :param steps:
        :param behavior:
        :param complexity:
        """
        self.valid = valid
        self.genome_id = genome_id
        self.genome = genome
        self.total_reward = total_reward
        self.steps = steps
        self.behavior = behavior
        self.complexity = complexity
        self.oscillatory_used = oscillatory_used


class Creature(object):
    """
    Class that encapsulates everything a creature needs to be created and has to be able to do during it's life.
    """

    def __init__(self, idx: int, genome: DefaultGenome, config: Config, params: Dict,
                 behavior: Behavior.__class__, cppn: Optional[CPPN] = None) -> None:
        self.idx = idx
        self.genome = genome

        self.config = config
        self.params = params

        if cppn is None:
            self.cppn = CPPN.create(genome=genome, config=self.config, params=params)
        else:
            self.cppn = cppn
        self.morph = CPPNMorph(idx, params, self.cppn)

        # self.ctrnn_handler = CTRNNHandler(params, self.cppn)
        self.network = None
        self.net = None
        self.complexity = CreatureComplexity(genotype_complexity=(self.cppn.num_nodes, self.cppn.num_connections),
                                             phenotype_complexity=(0, 0))
        self.build_reward = self.params["fitness_params"]["dummy"]
        self.reward = 0
        self.steps = 0
        self.behavior = behavior(self.params["ns_params"]["behavior_dummy"],
                                 self.params["ns_params"]["behavior_size"],
                                 self.params["ns_params"][
                                     "behavior_sample_frequency"])
        self.oscillatory_used = (False, False)
        self.valid = self.create_phenotype()

        # Trajectory drawing
        self.trajectory_samples = list()
        self.target_positions = list()

    def build_morphology(self) -> bool:
        """
        Build the creature's morphology.
        :return: bool indicating whether the morphology is valid
        """
        valid_morph = self.morph.build_morphology()
        return valid_morph

    def build_ctrnns(self):
        """
        Add all configurable joints to the CTRNNHandler, which in turn creates the CTRNNs.
        """
        # self.ctrnn_handler.add_configurable_joints(self.morph.configurable_joint_locations)
        pass

    def add_general_input_nodes(self) -> None:
        """
        Add the general input (e.g. energy status) and output (e.g. eat action) node locations to the substrate.
        """
        prune_params = self.params["es_params"]["prune_not_fully_connected"]
        substrate_top_plane = 1.15
        substrate_bottom_plane = -1.15  # a little behind the middle depth of the outer brain substrate region

        # FOOD ENERGY
        #   EAT state
        self.morph.substrate.add_general_input_node((0.1, substrate_top_plane, 0.0), prune_params["inputs"])
        #   EAT action
        self.morph.substrate.add_general_output_node((-0.1, substrate_top_plane, 0.0), prune_params["outputs"])

        # Two input nodes that will be given a sin and cosin signal (from unity)
        self.morph.substrate.add_general_input_node((-0.5, substrate_bottom_plane, 0.0), prune_params["inputs"])
        self.morph.substrate.add_general_input_node((0.5, substrate_bottom_plane, 0.0), prune_params["inputs"])

        # Bias input node (always last)
        self.morph.substrate.add_general_input_node((0.0, substrate_bottom_plane, 0.0), prune_params["inputs"])



    def build_brain(self) -> bool:
        """
        Build the creature's brain.
        :return: bool indicating whether the brain is valid
        """
        self.network = ESNetwork(self.morph.substrate, self.cppn, self.params["es_params"])
        self.net = self.network.create_phenotype_network()
        # self.build_reward += self.network.build_reward
        if self.net:
            genotype_complexity = self.cppn.num_nodes, self.cppn.num_connections
            phenotype_complexity = self.net.num_nodes, self.net.num_connections
            self.complexity = CreatureComplexity(genotype_complexity=genotype_complexity,
                                                 phenotype_complexity=phenotype_complexity)
            self.oscillatory_used = tuple(self.network.oscillatory_used)
            return True
        return False

    def create_phenotype(self) -> Tuple[bool, bool]:
        """
        Build creature morphology and it's brain.
        :return: tuple of bools indicating respectively if the created morphology and brain are valid
        """
        valid_morph = self.build_morphology()
        if valid_morph:
            self.add_general_input_nodes()
            valid_brain = self.build_brain()

            if valid_brain:
                self.behavior.set(self.morph)
                if self.params["experiment"]["dgcb"]:
                    self.build_ctrnns()

            return valid_morph, valid_brain
        return valid_morph, False

    def get_actions(self, observations: np.ndarray) -> Optional[np.ndarray]:
        """
        Given observations, retrieve
        :param observations:
        :return:
        """
        self.steps += 1
        o = self.net.activate(observations)
        if self.params["experiment"]["dgcb"]:
            # Feed BJO's to CTRNNs and use their outputs for joint outputs
            brain_joint_outputs = o[:self.morph.substrate.output_general_start_idx]
            # joint_outputs = self.ctrnn_handler.activate(brain_joint_outputs)

            general_outputs = o[self.morph.substrate.output_general_start_idx:]
            # o = joint_outputs + general_outputs

        actions = np.pad(o, (
            0, 49 - len(o)))  # todo: change padding amount this dynamically based on # outputs (also in unity)
        actions = np.array(actions, np.float32)

        return actions[None, :]

    def add_trajectory_and_target_sample(self, current_loc: List[float],
                                         current_target_loc: List[float]) -> None:
        self.trajectory_samples.append(tuple(current_loc))
        self.target_positions.append(tuple(current_target_loc))

    def add_reward(self, reward: float):
        self.reward += reward

    def set_reward(self, reward: float):
        self.reward = reward

    def add_behavior(self, value):
        self.behavior.add(value)

    def set_behavior(self, value: float):
        self.behavior.set(value)

    def save(self, generation: int, reward_threshold: float, forced: bool = False):
        overal_best = self.reward >= reward_threshold
        solved = self.reward >= self.params["fitness_params"]["solved"]

        if forced or solved or overal_best:
            name = f"gen_{generation}_genome_{self.idx}_reward_{round(self.reward, 2)}_"
            base_path = Path(self.params["experiment"]["results_dir"])

            if not solved:
                base_path = base_path / "overal_best"

            output_path = base_path / name
            output_path.mkdir(exist_ok=True, parents=True)

            # Save cppn
            draw_net(self.cppn, filename=str(output_path / "cppn"))
            with open(str(output_path / "cppn.pkl"), "wb") as output:
                pickle.dump(self.cppn, output, pickle.HIGHEST_PROTOCOL)

            # Save genome
            with open(str(output_path / "genome.pkl"), "wb") as output:
                pickle.dump(self.genome, output, pickle.HIGHEST_PROTOCOL)

            # Save (adapted) ANN: pickle currently not possible due to numba class type restrictions (Connection class is numba)
            # with open(str(output_path / "ANN.pkl"), "wb") as output:
            #    pickle.dump(self.net, output, pickle.HIGHEST_PROTOCOL)
            self.network.draw(str(output_path / "ANN"), self.morph)

            # Save trajectory
            if not forced:
                draw_trajectory(self.trajectory_samples, self.target_positions, str(output_path / "trajectory"))

    def reset(self) -> None:
        self.steps = 0
        self.reward = 0
        self.behavior.reset()
        self.net.reset()

    def reset_brain(self) -> None:
        self.net.reset()

    def finish_evaluation(self, generation: int, save_reward_threshold: float) -> CreatureEvaluationResult:
        if all(self.valid):
            self.save(generation, save_reward_threshold)
        if self.params["experiment"]["build_rewards"]:
            self.build_reward += sum(self.valid)
        else:
            self.build_reward = 0

        return CreatureEvaluationResult(valid=self.valid, genome_id=self.idx, genome=self.genome,
                                        total_reward=self.build_reward + self.reward,
                                        steps=self.steps, behavior=self.behavior.end(),
                                        complexity=self.complexity, oscillatory_used=self.oscillatory_used)
