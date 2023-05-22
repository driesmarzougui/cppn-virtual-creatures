from typing import List, Tuple, Dict
import numpy as np


class LazyCPPNBrainOutput:
    def __init__(self, output):
        self.output = output
        self.w = next(self.output)

    def get_connection_params(self) -> Tuple[float, float, float, float, float, float, float]:
        return (self.w, next(self.output), next(self.output), next(self.output), next(self.output), next(self.output),
                next(self.output))


class CPPNOutput:
    """
    Helper class to represent CPPN output based on the given experiment settings.
    """

    def __init__(self, output: list, params: Dict):
        self.SB = -100  # dummy
        self.SBD_X, self.SBD_Y, self.SBD_Z = -1, -10, -100  # dummy
        self.BB = -100  # dummy

        if params["experiment"]["dgcb"]:
            if params["morph_params"]["fix_brain"]:
                self.w, self.bias, self.tau, self.lr, self.A, self.B, self.C, self.D, self.M, self.NB, self.FB, \
                self.SB, self.SBD, self.FJ, self.CJ, self.LAX, self.HAX, self.AYL, self.AZL = output
            else:
                self.w, self.bias, self.tau, self.lr, self.A, self.B, self.C, self.D, self.M, self.NB, self.BB, \
                self.FB, self.SB, self.SBD, self.FJ, self.CJ, self.LAX, self.HAX, self.AYL, self.AZL = output
        else:
            if params["morph_params"]["fix_brain"]:
                self.w, self.lr, self.A, self.B, self.C, self.D, self.M, self.FB, \
                self.CJ, self.LAX, self.HAX, self.AYL, self.AZL = output
            else:
                self.w, self.lr, self.A, self.B, self.C, self.D, self.M, self.NB, self.BB, self.FB, self.SB, self.SBD, \
                self.FJ, self.CJ, self.LAX, self.HAX, self.AYL, self.AZL = output

        # CPPN has bipolar steepened sigmoid as final activation, so out.w will lie in [-1, 1]
        if -0.1 < self.w < 0.1:
            self.w = 0
        else:
            # Scale output weight to [-max_weight, max_weight]
            self.w *= params["es_params"]["max_weight"]

    def get_block_type(self) -> int:
        """
        Get the block type.
        """
        if self.FB > 0:
            return 1
        else:
            return 0

    def get_block_types_and_activations(self) -> List[Tuple[int, float]]:
        """
        Return a list of (block type, activation) tuples, sorted by descending activation.
        """
        activations = [self.NB, self.FB, self.SB, self.BB]
        types = np.argsort(activations)
        return [(block_type, activation) for block_type, activation in zip(types, activations)]

    def get_joint_type(self, low_limit: float, high_limit: float) -> Tuple[int, int, int, int, int]:
        """
        Get joint type at queried position together with CJ joint parameters
        """
        if self.CJ > 0:
            # Configurable joint; scale output parameters to predetermined ranges
            return 1, self.scale_to_int(self.LAX, -high_limit, -low_limit), self.scale_to_int(self.HAX, low_limit,
                                                                                              high_limit), \
                   self.scale_to_int(self.AYL, low_limit, high_limit), self.scale_to_int(self.AZL, low_limit,
                                                                                         high_limit)
        else:
            # Fixed joint
            return 0, 0, 0, 0, 0

    def scale_to_int(self, a: float, minimum: float, maximum: float) -> int:
        return int(round(self.scale_to_float(a, minimum, maximum)))

    def scale_to_float(self, a: float, minimum: float, maximum: float) -> float:
        # BSS output activation function so [-1, 1] to given [min, max] interval
        return (a + 1) / 2 * (maximum - minimum) + minimum

    def get_connection_params(self) -> Tuple[float, float, float, float, float, float, float]:
        return self.w, self.A, self.B, self.C, self.D, self.lr, self.M

    def get_sensor_block_direction(self) -> int:
        # Returns the sensor block direction in terms of an index from the DIRECTIONS variable in CPPNMorph.py
        cppn_values = [self.SBD_X, self.SBD_Y, self.SBD_Z]
        axis = int(np.argmax(np.abs(cppn_values)))
        direction = axis * 2
        if cppn_values[axis] < 0:
            direction += 1
        return direction
