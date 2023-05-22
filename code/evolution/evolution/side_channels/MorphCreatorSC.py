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
class MorphCreatorSC(SideChannel):
    """
    Unity-Python side-channel communication.
    Sends agent creation requests to unity for agents with dynamically built morphologies.
    """

    def __init__(self) -> None:
        super().__init__(uuid.UUID('e910421e-4dd9-11eb-a9e8-e3175c87fdbf'))
        self.creation_requested = False
        self.creation_done = False

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        self.creation_done = True
        self.creation_requested = False

        message = msg.read_string()
        logging.info(message)

    def send_request(self, morph_str: str) -> None:
        logging.info(f"Requesting morphology creation of agent!")
        self.creation_done = False
        self.creation_requested = True

        msg = OutgoingMessage()
        msg.write_string(morph_str)
        super().queue_message_to_send(msg)
