from typing import List, Tuple, Dict, Union

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel import SideChannel

from AESHN.shared import TBLogger, EnvFactory
from evolution.side_channels.AgentCreatorSC import AgentCreatorSC
from evolution.side_channels.WorldStateLoggingSC import WorldStateLoggingSC


def initialise_environment(path: str, initial_genomes: List[int], tb_logger: TBLogger, worker_id: int = 0) -> Tuple[
    UnityEnvironment, Dict[str, Union[AgentCreatorSC, SideChannel]]]:
    """
    Creates a UnityEnvironment with all side-channels attached.
    """
    side_channels = create_side_channels(tb_logger)
    env = EnvFactory(path)(worker_id, no_graphics=True, gym=False, side_channels=list(side_channels.values()))
    env.reset()

    agent_creator_sc: AgentCreatorSC = side_channels["agent-creator"]

    agent_creator_sc.send_request(initial_genomes)

    env.reset()

    while not agent_creator_sc.creation_done:
        pass

    return env, side_channels


def create_side_channels(tb_logger: TBLogger) -> Dict[str, SideChannel]:
    """
    Initialises all Unity-Python side-channels.
    """
    agent_creator_sc = AgentCreatorSC()
    logging_sc = WorldStateLoggingSC(tb_logger)
    return {"agent-creator": agent_creator_sc, "world-state-logging": logging_sc}


def genome_to_behavior_name(genome_id: int) -> str:
    """
    Transform the given genome id to an Unity ML-Agents behavior name.
    """
    return f"{genome_id}?team=0"


def behavior_name_to_genome(behavior_name: str) -> int:
    """
    Transform the given Unity ML-Agents behavior name to a genome id.
    """
    return int(behavior_name.split("?")[0])


def remove_from_env(env: UnityEnvironment, agent: str) -> None:
    """
    Remove the given agent (specified by behavior name) from the UnityEnvironment cache.
    """
    env._env_state.pop(agent, None)
    env._env_specs.pop(agent, None)
    env._env_actions.pop(agent, None)
