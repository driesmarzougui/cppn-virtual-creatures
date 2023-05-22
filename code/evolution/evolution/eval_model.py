import argparse

import os

from AESHN.shared import EnvFactory
from evolution.creature_handling.creature import Creature
from evolution.side_channels.AgentCreatorSC import AgentCreatorSC
from evolution.side_channels.MorphCreatorSC import MorphCreatorSC
from evolution.utils.config import create_config
from evolution.utils.params import get_params

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from pathlib import Path
import tqdm


def initialise_environment(path: str, morph_string: str, env=None, morph_creator_sc=None):
    if env is None:
        agent_creator_sc = AgentCreatorSC()
        morph_creator_sc = MorphCreatorSC()
        env = EnvFactory("meta/builds/ive-v1-1/ive-v1-1")(0, no_graphics=False, gym=False, side_channels=[agent_creator_sc, morph_creator_sc])
    env.reset()

    morph_creator_sc.send_request(morph_string)
    env.step()

    while not morph_creator_sc.creation_done:
        pass

    return env, morph_creator_sc


def main(args: argparse.Namespace):
    PARAMS_PATH = args.params

    params = get_params(PARAMS_PATH)
    ENV_PATH = params["experiment"]["env_path"]

    config = create_config(params)
    behavior_class = globals()[params["ns_params"]["behavior"]]

    base_path = Path(args.eval_path)

    index = -2  # highest reward
    #index = 1   # latest gen

    original_rewards = [float(x[0].split('/')[-1].split('_')[index]) for x in os.walk(base_path) if
                        x[0].split('/')[-1].startswith("gen")]
    sorted_indices = np.argsort(original_rewards)[::-1]
    cppn_paths = [Path(x[0]) / "genome.pkl" for x in os.walk(base_path) if x[0].split('/')[-1].startswith("gen")]
    cppn_paths = [cppn_paths[i] for i in sorted_indices]

    pbar = tqdm.tqdm(total=len(cppn_paths))
    rewards = list()
    for cppn_path in cppn_paths:
        print(f"EVALUATING {cppn_path}")
        with open(cppn_path, 'rb') as cppn_f:
            genome_id = 0
            agent_id = f"{genome_id}?team=0"
            genome = pickle.load(cppn_f)
            creature = Creature(idx=genome_id, genome=genome, config=config, params=params, behavior=behavior_class,
                                )
            print(len(creature.net.modulatory_node_evals))
            print(len(creature.net.modulatory_nodes))
            if all(creature.valid):
                env, msc = None, None
                for _ in range(1):
                    env, msc = initialise_environment(str(ENV_PATH), str(creature.morph), env, msc)
                    done = False
                    steps = 0
                    while not done:
                        env.step()
                        for a_id, behavior_spec in env.behavior_specs.items():
                            if agent_id == a_id:
                                steps += 1
                                decision_steps, terminal_steps = env.get_steps(agent_id)
                                if list(terminal_steps):
                                    # Agent has died
                                    #print(f"TERMINAL REWARDS :: {terminal_steps.reward[0]}")
                                    done = True
                                elif list(decision_steps):
                                    obs = np.concatenate([ob.flatten() for ob in decision_steps.obs])
                                    reward = decision_steps.reward[0]
                                    #print(f"DECISION REWARDS :: {reward}")
                                    creature.add_reward(reward)
                                    actions = creature.get_actions(obs)
                                    env.set_actions(agent_id, actions)
                                #print(steps)
                        if done:
                            rewards.append(creature.reward)
                    creature.reset_brain()
                env.close()
            else:
                print("\tINVALID")

            pbar.update(1)

    print(list(zip(rewards, cppn_path)))
    print(max(rewards))
    print(min(rewards))


if __name__ == '__main__':
    eval()
