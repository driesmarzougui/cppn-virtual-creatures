from typing import List, Tuple, Dict
import numpy as np
from AESHN.shared.cppn.activation_functions import sigmoid,bss
from AESHN.shared.cppn.cppn import CPPN
import numba as nb

# Fixed node locations within [-1, 1] x [-1, 1] x [-1, 1] substrate
INPUT_NODES = [0]
HIDDEN_NODES = [1, 2]
OUTPUT_NODES = [3, 4, 5]

NODE_TO_LOCATION = {
    0: (0, 0, -1),
    1: (-1, 0, 0),
    2: (1, 0, 0),
    3: (-1, 0, 1),
    4: (0, 0, 1),
    5: (1, 0, 1)
}

# Build node to input node dictionary (this can be prebuilt as connections are fixed although some might have 0 weight)
NUM_NODES = 6
NODE_TO_INODES = {
    1: np.array([0, 2], dtype=np.uint8),
    2: np.array([0, 1], dtype=np.uint8),
    3: np.array([0, 1, 2, 4, 5], dtype=np.uint8),
    4: np.array([0, 1, 2, 3, 5], dtype=np.uint8),
    5: np.array([0, 1, 2, 3, 4], dtype=np.uint8)
}
NODE_TO_INODES_NB = nb.typed.Dict.empty(key_type=nb.types.uint8, value_type=nb.types.uint8[:])
for node, inodes in NODE_TO_INODES.items():
    NODE_TO_INODES_NB[node] = inodes

# Defaults
TIME_SLICE = 0.02
OUTPUT_NODE_BIAS = 3.0
OUTPUT_NODE_TAU = 0.1
TAU_RANGE_MIN, TAU_RANGE_MAX = 0.1, 2.0
BIAS_RANGE_MIN, BIAS_RANGE_MAX = -3.0, 3.0


class CTRNN(object):
    """
    Class that represents a Continous-Time Recurrent Neural Network (CTRNN).

    Note: As this is a really small network and as there are is a maximum of 16 configurable joints and thus CTRNNs,
            not a lot of performance optimisations for CTRNN creation have been carried out.
            For activation (which happens more) numba was used to boost performance.
    """

    def __init__(self, con_weights: np.ndarray, node_to_time_factor: np.ndarray, node_to_bias: np.ndarray) -> None:
        self.con_weights = con_weights
        self.node_to_time_factor = node_to_time_factor
        self.node_to_bias = node_to_bias

        self.values = np.zeros(6)

    def activate(self, brain_joint_output: float) -> List[float]:
        return self.calc_nodes(
            node_to_inodes=NODE_TO_INODES_NB,
            bjo=brain_joint_output,
            values=self.values,
            node_to_bias=self.node_to_bias,
            node_to_tf=self.node_to_time_factor,
            con_weights=self.con_weights
        )

    @staticmethod
    @nb.njit()
    def calc_nodes(node_to_inodes: nb.typed.Dict, bjo: float, values: np.ndarray,
                   node_to_bias: np.ndarray,
                   node_to_tf: np.ndarray,
                   con_weights: np.ndarray) -> List[float]:
        # Follows "Confronting the Challenge of Learning a Flexible Neural Controller for a Diversity of Morphologies"
        values[0] = bjo
        prev_values = values.copy()

        for node in range(1, 6):
            bias = node_to_bias[node]
            time_factor = node_to_tf[node]
            prev_val = prev_values[node]

            inodes = node_to_inodes[node]
            input_sum = 0.0
            for inode in inodes:
                input_sum += con_weights[node, inode] * sigmoid(prev_values[inode] + bias)

            values[node] = prev_val + time_factor * (-prev_val + input_sum)

        return [bss(val) for val in values[-3:]]

    @staticmethod
    def create(cppn: CPPN, cj_loc: Tuple[float, float, float]):
        """
        Applies HyperNEAT to create a CTRNN in a local substrate with fixed node locations, positioned at the given
            configurable joint location.
        """

        # Set connection weights
        connection_weights = np.zeros((NUM_NODES, NUM_NODES))
        for node, inodes in NODE_TO_INODES.items():
            loc = NODE_TO_LOCATION[node]
            for inode in inodes:
                iloc = NODE_TO_LOCATION[inode]
                cppn_output = cppn.query_cppn_ctrnn_con(coord1=iloc, coord2=loc, coord_b=cj_loc)
                connection_weights[node, inode] = cppn_output.w

        # Set node tau and bias parameters
        node_to_time_factor = np.zeros(NUM_NODES)
        node_to_bias = np.zeros(NUM_NODES)
        for node in OUTPUT_NODES:
            node_to_time_factor[node] = TIME_SLICE / OUTPUT_NODE_TAU
            node_to_bias[node] = OUTPUT_NODE_BIAS

        #   Hidden nodes
        for node in HIDDEN_NODES:
            loc = NODE_TO_LOCATION[node]
            cppn_output = cppn.query_cppn_ctrnn_node(coord=loc, coord_b=cj_loc)

            bias = cppn_output.scale_to_float(cppn_output.bias, BIAS_RANGE_MIN, BIAS_RANGE_MAX)
            tau = cppn_output.scale_to_float(cppn_output.tau, TAU_RANGE_MIN, TAU_RANGE_MAX)

            node_to_time_factor[node] = TIME_SLICE / tau
            node_to_bias[node] = bias

        return CTRNN(connection_weights, node_to_time_factor, node_to_bias)
