from typing import List, Union

from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym


class EnvFactory:
    """
    Helper class to spawn unity environments.
    """

    def __init__(self, path: str):
        self.path = path

    def __call__(self, worker_id, no_graphics=True, gym=True, side_channels=None) -> Union[UnityEnvironment, gym.Env]:
        if side_channels is None:
            side_channels = list()
        conf_channel = EngineConfigurationChannel()
        channels = [conf_channel] + side_channels
        unity_env = UnityEnvironment(self.path, no_graphics=no_graphics, side_channels=channels, seed=42,
                                     worker_id=worker_id)
        conf_channel.set_configuration_parameters(capture_frame_rate=60, time_scale=3.0)
        if gym:
            return UnityToGymWrapper(unity_env, allow_multiple_obs=True)
        else:
            return unity_env
