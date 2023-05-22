from typing import Dict, Optional, Tuple

import ray
from mlagents_envs.environment import UnityEnvironment
from neat import Config, DefaultGenome
import numpy as np
from AESHN.shared import EnvFactory
from evolution.evolutionary_algorithms.ns import Behavior
from evolution.creature_handling.creature import Creature, CreatureEvaluationResult
from evolution.side_channels.AgentCreatorSC import AgentCreatorSC
from evolution.side_channels.MorphCreatorSC import MorphCreatorSC
from evolution.utils.environment import behavior_name_to_genome


@ray.remote(num_cpus=1)
class EnvRunner(object):
    """
    Helper class to run a Unity environment for single agent evaluation.
    """

    def __init__(self, worker_id: int, env_path: str, config: Config, params: Dict, behavior: Behavior.__class__):
        self.worker_id = worker_id
        self.env_path = env_path
        self.config = config
        self.params = params
        self.behavior = behavior
        self.agent_creator_sc = AgentCreatorSC()
        self.morph_creator_sc = MorphCreatorSC()
        self.env = self.initialise_env()

    def initialise_env(self) -> UnityEnvironment:
        # env = EnvFactory(self.env_path)(worker_id=self.worker_id, no_graphics=True, gym=False,
        #                                side_channels=[self.agent_creator_sc, self.morph_creator_sc])
        env = EnvFactory(self.env_path)(worker_id=self.worker_id, no_graphics=self.params["experiment"]["no_graphics"],
                                        gym=False,
                                        side_channels=[self.agent_creator_sc, self.morph_creator_sc])
        env.reset()
        return env

    def add_agent(self, morph_str: str) -> None:
        self.env.reset()
        self.morph_creator_sc.send_request(morph_str)
        self.env.step()
        while not self.morph_creator_sc.creation_done:
            pass

    def eval_genome(self, generation: int, genome_t: Tuple[int, DefaultGenome], save_reward_threshold: float) -> \
            Optional[CreatureEvaluationResult]:
        genome_id, genome = genome_t

        # Create creature
        creature = Creature(genome_id, genome, self.config, self.params, self.behavior)

        if all(creature.valid):
            print(f"VALID: {genome_id}")

            # Evaluate creature
            rewards = []

            for _ in range(self.params["experiment"]["trials"]):
                # Add creature to env
                self.add_agent(str(creature.morph))

                reward = 0.00001  # avoid 0 reward
                steps = 0
                done = False
                while not done:
                    self.env.step()
                    for agent_id, behavior_spec in self.env.behavior_specs.items():
                        try:
                            g_id = behavior_name_to_genome(agent_id)
                            if g_id == genome_id:
                                steps += 1
                                decision_steps, terminal_steps = self.env.get_steps(agent_id)
                                if list(terminal_steps):
                                    # Agent has died
                                    done = True

                                    reward += terminal_steps.reward[0]
                                    # creature.set_reward(reward)
                                    # creature.set_behavior(reward)
                                    break
                                elif list(decision_steps):
                                    obs = np.concatenate([ob.flatten() for ob in decision_steps.obs])

                                    reward += decision_steps.reward[0]

                                    location_xz = obs[creature.net.inputs_end:][:2]
                                    # target_xz = obs[creature.net.inputs_end:][2:4]
                                    target_xz = [0.0, 0.0]

                                    # Add trajectory behavior info
                                    creature.add_behavior(location_xz)

                                    # Add trajectory and target samples
                                    creature.add_trajectory_and_target_sample(location_xz, target_xz)

                                    actions = creature.get_actions(obs)
                                    self.env.set_actions(agent_id, actions)
                        except ValueError:
                            pass
                self.env.reset()
                creature.reset_brain()

                rewards.append(reward)
            creature.add_reward(min(rewards))

        # Return results: (valid, total_reward, steps, behavior)
        return creature.finish_evaluation(generation, save_reward_threshold)
