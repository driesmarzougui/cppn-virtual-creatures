import math
from typing import List

import numpy as np
import numba as nb
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaTypeSafetyWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)


class ModularNetwork(object):
    """
    Numba optimized Neural Network implementation.
    Used for creature brains created by AESHN.
    """
    def __init__(self, inputs, outputs, node_evals, activation, max_weight=5.0, update_interval=1):
        self.input_nodes = inputs
        self.inputs_end = len(self.input_nodes)
        self.output_nodes = outputs
        self.outputs_end = self.inputs_end + len(self.output_nodes)
        self.hidden_nodes = [ne[0] for ne in node_evals if ne[0] not in self.output_nodes]
        self.max_weight = max_weight
        self.update_interval = update_interval
        self.activation = activation
        self.py_regular_node_evals = dict()
        self.regular_node_evals = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.Tuple((nb.types.float64, nb.types.float64,
                                       nb.types.List(nb.types.Tuple((nb.int64, Connection.class_type.instance_type)))))
        )
        self.py_modulatory_node_evals = dict()
        self.modulatory_node_evals = nb.typed.Dict.empty(
            key_type=nb.types.int64,
            value_type=nb.types.Tuple((nb.types.List(nb.int64), nb.types.List(Connection.class_type.instance_type)))
        )
        self.modulatory_nodes = {node for node in
                                 self.hidden_nodes}  # will hold the nodes that don't contribute to the output but only to the connection modulated learning

        self.node_evals = node_evals
        self.node_evaluation_order = list()

        self.nodes = self.input_nodes + self.output_nodes + self.hidden_nodes

        self.acts = None
        self.neuron_signals = None
        self.reset()

        # Brain complexity
        self.num_nodes = len(self.hidden_nodes)
        self.num_connections = 0

        # required for iterative inference
        self.activated = {id: False for id in self.nodes}
        self.in_activation = {id: False for id in self.nodes}
        self.initialise_iterative_inference(node_evals)

    def initialise_iterative_inference(self, node_evals):
        """
        Helper function to enable iterative inference (instead of recursive).
        """
        # Split given connections into modulatory and regular
        for node, bias, response, links in node_evals:
            self.num_connections += len(links)
            regular_links = list()
            mod_inodes = list()
            mod_cons = list()
            for (input_node, connection) in links:
                if connection.M > 0.0:
                    # modulatory connection
                    mod_inodes.append(input_node)
                    mod_cons.append(connection)
                else:
                    # regular connection
                    regular_links.append((input_node, connection))
                    self.modulatory_nodes.discard(input_node)

            self.py_regular_node_evals[node] = (bias, response, regular_links)
            self.py_modulatory_node_evals[node] = (mod_inodes, mod_cons)

        # Set iterative inference order by activating the network once recursively
        self.get_iterative_node_order()

        for node, val in self.py_regular_node_evals.items():
            self.regular_node_evals[node] = val

        for node, val in self.py_modulatory_node_evals.items():
            if len(val[0]) > 0:
                self.modulatory_node_evals[node] = val

    def reset(self):
        self.acts = 0
        self.neuron_signals = np.zeros(len(self.nodes))

    def get_iterative_node_order(self):
        """
        Carries out a single network activation recursively, starting from the output nodes.
        Used to set the activation paths, enabling iterative activation.
        """
        # Set input nodes as activated
        for input_node in self.input_nodes:
            self.activated[input_node] = True

        # Get each output node activation recursively
        for output_node in self.output_nodes:
            try:
                self.get_iterative_node_order_rec(output_node)
            except KeyError as ex:
                # Output node has no incoming connections
                pass

        for mod_node in self.modulatory_nodes:
            assert not self.activated[mod_node], f"Modulatory node {mod_node} shouldn't have been activated already!"
            self.get_iterative_node_order_rec(mod_node)

    def activate(self, observations) -> List[float]:
        """
        Carries out a single network activation.
        Sets activation signal per neuron.
        """
        # Set input nodes
        self.neuron_signals[:self.inputs_end] = observations[:self.inputs_end]  # Throw away surplus of observations

        self.calc_nodes(self.node_evaluation_order, self.regular_node_evals, self.neuron_signals, self.activation)

        self.step()

        return list(self.neuron_signals[self.inputs_end:self.outputs_end])

    @staticmethod
    @nb.njit()
    def calc_nodes(node_eval_order, regular_node_evals, neuron_signals, activation):
        for node in node_eval_order:
            bias, response, links = regular_node_evals[node]

            input_sum = 0.0
            for inode, con in links:
                input_sum += con.weight * neuron_signals[inode]
            neuron_signals[node] = activation(input_sum)    # node specific bias and response currently not used

    def step(self):
        """
        Update model weights.
        """
        self.acts += 1
        self.calc_step(self.modulatory_node_evals, self.neuron_signals, self.regular_node_evals, self.update_interval,
                       self.acts, self.max_weight)

        if self.acts == self.update_interval:
            self.acts = 0

    @staticmethod
    @nb.njit()
    def calc_step(modulatory_node_evals, neuron_signals, regular_node_evals, update_interval, acts, max_weight):
        for node, (inodes, cons) in modulatory_node_evals.items():
            # Calculate modulatory signal
            mod_signal = 0.0
            for inode, con in zip(inodes, cons):
                mod_signal += con.weight * neuron_signals[inode]

            if mod_signal != 0:
                post = neuron_signals[node]

                # Tanh
                mod_signal = math.tanh(mod_signal / 2)

                # Update weights
                _, _, links = regular_node_evals[node]
                for (input_node, connection) in links:
                    A = connection.A
                    B = connection.B
                    C = connection.C
                    lr = connection.lr
                    D = connection.D

                    pre = neuron_signals[input_node]

                    connection.delta_weight += mod_signal * lr * (A * pre * post + B * pre + C * post + D)
                    if acts == update_interval:
                        connection.weight += (connection.delta_weight / update_interval)
                        if connection.weight > max_weight:
                            connection.weight = max_weight
                        elif connection.weight < -max_weight:
                            connection.weight = -max_weight
                        connection.delta_weight = 0

    def get_iterative_node_order_rec(self, node):
        # If we've reached an already activated node we return since the signal is already set
        if self.activated[node]:
            return

        # Mark that the node is currently being calculated
        self.in_activation[node] = True

        _, _, links = self.py_regular_node_evals[node]

        # Go through each incoming connection and activate it
        for input_node, connection in links:
            # If this node is currently in activation, then we have reached a cycle or a recurrent connection --> use the previous activation and stop this traversal
            if not self.in_activation[input_node] and not self.activated[input_node]:
                self.get_iterative_node_order_rec(input_node)

        # Mark this neuron as completed
        self.activated[node] = True

        # Node is not longer being processed
        self.in_activation[node] = False

        self.node_evaluation_order.append(node)


# Class representing a connection from one point to another with a certain weight.
connection_spec = [
    ('x1', nb.float64),
    ('y1', nb.float64),
    ('z1', nb.float64),
    ('x2', nb.float64),
    ('y2', nb.float64),
    ('z2', nb.float64),
    ('weight', nb.float64),
    ('A', nb.float64),
    ('B', nb.float64),
    ('C', nb.float64),
    ('D', nb.float64),
    ('lr', nb.float64),
    ('M', nb.float64),
    ('modulatory', nb.boolean),
    ('delta_weight', nb.float64),
    ('qt_width', nb.float64),
]


@nb.experimental.jitclass(connection_spec)
class Connection:
    """
    Helper class to represent a brain connection.
    """
    def __init__(self, x1, y1, z1, x2, y2, z2, cppn_output, qt_width):
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.weight, self.A, self.B, self.C, self.D, self.lr, self.M = cppn_output
        self.modulatory = self.M > 0.0
        self.delta_weight = 0.0  # connection local storage for updates
        self.qt_width = qt_width

        if self.modulatory > 0 and x1 == x2 and y1 == y2 and z1 == z2:
            # Recurrent connection to itself, force non-modulatory
            self.M = 0
            self.modulatory = False

    # Below is needed for use in set.
    def __eq__(self, other):
        return self.x1, self.y1, self.z1, self.x2, self.y2, self.z2 == other.x1, other.y1, other.z1, other.x2, other.y2, other.z2

    def __hash__(self):
        return hash((self.x1, self.y1, self.z1, self.x2, self.y2, self.z2, self.weight, self.modulatory))
