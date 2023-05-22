from typing import Dict, List, Tuple

from AESHN.shared.cppn.cppn import CPPN
from AESHN.shared.ctrnn.ctrnn import CTRNN


class CTRNNHandler(object):
    """
    Helper class to handle the CTRNN for every configurable joint of the agent.
    """

    def __init__(self, params: Dict, cppn: CPPN) -> None:
        self.params = params
        self.cppn = cppn
        self.ctrnns: List[CTRNN] = list()

    def add_configurable_joints(self, configurable_joint_locations: List[Tuple[float, float, float]]) -> None:
        """
        Add the list of configurable joint(s) (locations) by creating and storing the corresponding CTRNNs.
        """
        self.ctrnns.extend([CTRNN.create(self.cppn, cj_loc) for cj_loc in configurable_joint_locations])

    def activate(self, brain_joint_outputs: List[float]) -> List[float]:
        """
        Feed the given list of brain joint outputs to each corresponding CTRNN and return a flat list of all
            CTRNN outputs.
        """
        output = list()
        for ctrnn, bjo in zip(self.ctrnns, brain_joint_outputs):
            output.extend(ctrnn.activate(bjo))

        return output
