from typing import Tuple, List, Dict

from AESHN.shared.visualize import draw_net
from neat.graphs import feed_forward_layers
from neat.six_util import itervalues
import numba as nb
import numpy as np

from AESHN.shared.cppn.activation_functions import get_activation_function
from AESHN.shared.cppn.cppn_output import CPPNOutput, LazyCPPNBrainOutput
from evolution.utils.nb_math_utils import distance_from_origin, distance_between_coords
import torch

from pytorch_neat.activations import bss
from pytorch_neat.cppn import create_cppn


class CPPN(object):
    """
    Class that represents a Compositional Pattern Producing Network (CPPN).
    """

    def __init__(self, inputs, outputs, node_evals, params, pt_cppn):
        self.input_nodes = inputs
        self.inputs_end = len(self.input_nodes)
        self.output_nodes = outputs
        self.outputs_end = self.inputs_end + len(self.output_nodes)

        input_output_nodes = set(inputs + outputs)
        self.hidden_nodes_and_act_f = [(node, act_f.__name__) for node, act_f, _, _, _, _ in node_evals if
                                       node not in input_output_nodes]
        self.num_nodes = len(self.hidden_nodes_and_act_f)
        self.node_evals = node_evals
        self.params = params
        self.bias = self.params["morph_params"]["subspace_size"][0]

        # Complexity
        self.num_connections = sum([len(inodes) for _, _, _, _, inodes, _ in self.node_evals])

        self.pt_cppn = pt_cppn

    def query_cppn_morph(self, coord_b: Tuple[float, float, float]) -> CPPNOutput:
        """
        Helper function to query the CPPN for morphology components (blocks and joints).
        :param coord_b: block or joint coordinate tuple
        """
        coord_b = (coord_b[0] / self.bias, coord_b[1] / self.bias, coord_b[2] / self.bias)
        bl = distance_from_origin(coord_b) / self.bias
        cppn_input = self.prepare_cppn_inputs(coord_b=coord_b, bl=bl)
        cppn_output = self.activate(cppn_input)
        return CPPNOutput(cppn_output, self.params)

    def query_cppn_brain_joint_sensor_input_con(self, coord_b: Tuple[float, float, float],
                                                coord: Tuple[float, float, float],
                                                outgoing: bool) -> LazyCPPNBrainOutput:
        """
        Helper function to query the CPPN for joint/sensor to brain input connections.
        :param coord_b: source sensor block or joint node coordinate tuple
        :param coord: target brain node coordinate tuple
        :param outgoing: not used here but kept to remain uniform with other query functions
        """
        coord_b = (coord_b[0] / self.bias, coord_b[1] / self.bias, coord_b[2] / self.bias)
        bl = distance_from_origin(coord_b) / self.bias
        cppn_input = self.prepare_cppn_inputs(coord2=coord, coord_b=coord_b, bl=bl)
        cppn_output = self.activate(cppn_input)
        return LazyCPPNBrainOutput(cppn_output)

    def query_cppn_brain_general_con(self, coord1: Tuple[float, float, float],
                                     coord2: Tuple[float, float, float], outgoing: bool = True) -> LazyCPPNBrainOutput:
        """
        Helper function to query the CPPN for general brain connections.
        :param coord1: node1 coordinate tuple
        :param coord2: node2 coordinate tuple
        :param outgoing: direction of the connection (True: coord1 -> coord2, False: coord2 -> coord1)
        """
        cl = distance_between_coords(coord1, coord2)
        cppn_input = self.prepare_cppn_inputs(coord1=coord1, coord2=coord2, cl=cl, outgoing=outgoing)
        cppn_output = self.activate(cppn_input)
        return LazyCPPNBrainOutput(cppn_output)

    def query_cppn_brain_joint_output_con(self, coord_b: Tuple[float, float, float],
                                          coord: Tuple[float, float, float], outgoing: bool) -> LazyCPPNBrainOutput:
        """
        Helper function to query the CPPN for brain to joint output connections.
        :param coord_b: target joint node coordinate tuple
        :param coord: source brain node coordinate tuple
        :param outgoing: not used here but kept to remain uniform with other query functions
        :return:
        """
        coord_b = (coord_b[0] / self.bias, coord_b[1] / self.bias, coord_b[2] / self.bias)
        bl = distance_from_origin(coord_b) / self.bias
        cppn_input = self.prepare_cppn_inputs(coord1=coord, coord_b=coord_b, bl=bl)
        cppn_output = self.activate(cppn_input)
        return LazyCPPNBrainOutput(cppn_output)

    def prepare_cppn_inputs(self, coord1: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                            coord2: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                            coord_b: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                            cl: float = 0.0, bl: float = 0.0,
                            outgoing: bool = True) -> Dict[str, float]:
        """
        Transforms the given coordinate tuples to a flat list representation with
            the given distance between points (l) and constant bias (1.O) inputs added.
        """
        x1, y1, z1 = coord1
        x2, y2, z2 = coord2
        b_x, b_y, b_z = coord_b

        x1, y1, z1, x2, y2, z2, b_x, b_y, b_z, cl, bl, bias = torch.tensor(
            [x1, y1, z1, x2, y2, z2, b_x, b_y, b_z, cl, bl, self.bias])
        if outgoing:
            return dict(x1=x1, y1=y1, z1=z1, x2=x2, y2=y2, z2=z2, b_x=b_x, b_y=b_y, b_z=b_z, cl=cl, bl=bl,
                        bias=bias)
        else:
            return dict(x1=x2, y1=y2, z1=z2, x2=x1, y2=y1, z2=z1, b_x=b_x, b_y=b_y, b_z=b_z, cl=cl, bl=bl,
                        bias=bias)

    def activate(self, x):
        return (node(x).item() for node in self.pt_cppn)

    @staticmethod
    def create(genome, config, params):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """
        # Gather expressed connections.
        connections = [cg.key for cg in itervalues(genome.connections) if cg.enabled]

        # map strange neat-python node indices (e.g. negatives) to normal positive values
        node_mapper = {key: i for i, key in
                       enumerate(config.genome_config.input_keys + config.genome_config.output_keys)}
        counter = len(node_mapper)

        input_keys = [node_mapper[node] for node in config.genome_config.input_keys]
        output_keys = [node_mapper[node] for node in config.genome_config.output_keys]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                # Force bipolar steepened sigmoid activation function if output node
                if node in config.genome_config.output_keys:
                    genome.nodes[node].activation = "bss"  # Bipolar Steepened Sigmoid
                ng = genome.nodes[node]
                activation_function = get_activation_function(ng.activation)

                if node not in node_mapper:
                    node_mapper[node] = counter
                    counter += 1

                input_nodes = []
                input_weights = []
                node_expr = []  # currently unused
                for conn_key in connections:
                    inode, onode = conn_key

                    if inode not in node_mapper:
                        node_mapper[inode] = counter
                        counter += 1
                    if onode not in node_mapper:
                        node_mapper[onode] = counter
                        counter += 1

                    if onode == node:
                        cg = genome.connections[(inode, onode)]
                        input_nodes.append(node_mapper[inode])
                        input_weights.append(cg.weight)
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                node_evals.append((node_mapper[node], activation_function, ng.bias, ng.response,
                                   np.array(input_nodes, dtype=int), np.array(input_weights, dtype=float)))

        pt_cppn = create_cppn(genome, config,
                              ["x1", "y1", "z1", "x2", "y2", "z2", "b_x", "b_y", "b_z", "cl", "bl", "bias"],
                              ["w", "lr", "A", "B", "C", "D", "M", "FB", "CJ", "LAX",
                               "HAX", "AYL", "AZL"], output_activation=bss)
        #pt_cppn = create_cppn(genome, config,
        #                      ["x1", "y1", "z1", "x2", "y2", "z2", "cl", "bias"],
        #                      ["w", "lr", "A", "B", "C", "D", "M", "FB", "CJ", "LAX",
        #                       "HAX", "AYL", "AZL"], output_activation=bss)
        return CPPN(input_keys, output_keys, node_evals, params, pt_cppn)
