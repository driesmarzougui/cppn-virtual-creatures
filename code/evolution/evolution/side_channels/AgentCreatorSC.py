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
class AgentCreatorSC(SideChannel):
    """
    Unity-Python side-channel communication.
    Sends agent creation requests to unity for fixed-morphology agents.
    """

    def __init__(self) -> None:
        super().__init__(uuid.UUID("cb5d3762-3322-11eb-9dc2-417b92aeb2c2"))
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

    def send_request(self, data: List[int]) -> None:
        logging.info(f"Requesting creation of {len(data)} agents!")
        self.creation_done = False
        self.creation_requested = True

        send_str = ";".join([str(x) for x in data])
        msg = OutgoingMessage()
        msg.write_string(send_str)
        super().queue_message_to_send(msg)
