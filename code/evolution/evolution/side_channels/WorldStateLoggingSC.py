import logging
from typing import List

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


# Create the StringLogChannel class
from AESHN.shared import TBLogger


class WorldStateLoggingSC(SideChannel):
    """
    Unity-Python side-channel communication.
    Receives virtual ecosystem state information from unity for tensorboard logging.
    """

    def __init__(self, tensorboard_logger: TBLogger) -> None:
        super().__init__(uuid.UUID("f5a93c6c-3806-11eb-ab0a-d8cb8a18a539"))
        self.tensorboard_logger = tensorboard_logger

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        message = msg.read_string()
        obj, state = message.split(':')

        self.tensorboard_logger.world_state[obj] = float(state)
