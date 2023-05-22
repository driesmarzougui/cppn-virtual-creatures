from collections import defaultdict
from typing import Optional, Union, List

import numpy as np

from AESHN.shared.substrate_factory import DGCBSubstrate
from AESHN.shared.visualize import draw_es
from AESHN.shared.modular_network_it import ModularNetwork, Connection
import numba as nb
from AESHN.shared.cppn.activation_functions import get_activation_function
import itertools
import functools

from MORPH.CPPNMorph import CPPNMorph


@functools.lru_cache(maxsize=512)
def get_division_point_offsets(d):
    return list(itertools.product((-d, d), (-d, d), (-d, d)))


class ESNetwork:
    """Adaptive Enhanced Substrate HyperNEAT
    Creates agent brains by placing and connecting nodes in the given substrate using the given CPPN.
    """

    def __init__(self, substrate: DGCBSubstrate, cppn, params):
        self.substrate = substrate
        self.cppn = cppn
        self.build_reward = 0
        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.update_interval = params["update_interval"]
        self.prune_not_fully_connected = params["prune_not_fully_connected"]
        self.activations = 2 ** params["max_depth"] + 1  # Number of layers in the network.
        self.activation = get_activation_function(params["activation"])
        self.coords_to_id = None
        self.connections = None

        self.oscillatory_used = [False, False]

    def draw(self, filename: str, morph: Optional[CPPNMorph] = None) -> None:
        draw_es(self.coords_to_id, self.connections, self.substrate, filename, morph)

    # Create a RecurrentNetwork using the ES-HyperNEAT approach.
    def create_phenotype_network(self, filename=None):
        # Where the magic happens.
        hidden_node_coords, self.connections, valid = self.es_hyperneat()
        if not valid or len(hidden_node_coords) == 0:
            return

        input_coordinates = self.substrate.input_coordinates
        output_coordinates = self.substrate.output_coordinates

        # Assign id's to input and output nodes
        input_nodes = list(range(len(input_coordinates)))
        output_nodes = list(range(len(input_nodes), len(input_nodes) + len(output_coordinates)))

        coordinates = input_coordinates + output_coordinates + hidden_node_coords
        indices = [i for i in range(len(coordinates))]

        # Map input and output coordinates to their IDs.
        self.coords_to_id = dict(zip(coordinates, indices))

        node_evals = list()
        node_links = defaultdict(list)

        for c in self.connections:
            source = self.coords_to_id[(c.x1, c.y1, c.z1)]
            target = self.coords_to_id[(c.x2, c.y2, c.z2)]
            node_links[target].append((source, c))

        # Combine the indices with the connections/links forming node_evals used by the ModularNetwork.
        for idx, links in node_links.items():
            node_evals.append((idx, 1.0, 1.0, links))

        # Visualize the network?
        if filename is not None:
            self.draw(filename)

        try:
            network = ModularNetwork(input_nodes, output_nodes, node_evals, self.activation, self.max_weight,
                                     self.update_interval)

            return network
        except RecursionError as ex:
            # Treat too complex brain phenotypes as invalid
            # print(ex)
            return

    # Recursively collect all weights for a given QuadPoint.
    @staticmethod
    def get_weights(p):
        temp = []

        def loop(pp):
            if pp is not None and len(pp.cs) > 0:
                for i in range(len(pp.cs)):
                    loop(pp.cs[i])
            else:
                if pp is not None:
                    temp.append(pp.w)

        loop(p)
        return np.array(temp)

    # Find the variance of a given QuadPoint.
    def variance(self, p):
        if not p:
            return 0.0
        weights = self.get_weights(p)
        if len(weights) == 0:
            return 0.0

        c2 = self.calc_var(weights)
        return c2

    @staticmethod
    @nb.njit()
    def calc_var(weights):
        return np.var(weights)

    # Initialize the quadtree by dividing it in appropriate quads.
    def division_initialization(self, coord, outgoing, query_cppn):
        o_x, o_y, o_z = self.substrate.origin  # todo: won't always be origin
        root = QuadPoint(o_x, o_y, o_z, self.substrate.width, self.substrate.height, self.substrate.depth,
                         1)  # todo: wont be original substrate dimensions
        q = [root]

        while q:
            p = q.pop(0)

            if p is None:
                continue

            # todo: can't use width for every dimension
            new_width = p.width / 2
            offsets = get_division_point_offsets(new_width)
            p.cs = [QuadPoint(p.x + dx, p.y + dy, p.z + dz, new_width, new_width,
                              new_width, p.lvl + 1) for dx, dy, dz in offsets]

            children_weights = list()
            append = children_weights.append
            for c in p.cs:
                cppn_ouput = query_cppn(coord, (c.x, c.y, c.z), outgoing)
                append(cppn_ouput.w)
                c.set_cppn_output(cppn_ouput)

            if (p.lvl < self.initial_depth) or \
                    (p.lvl < self.max_depth and self.calc_var(
                        np.array(children_weights, copy=False)) > self.division_threshold):
                q.extend(p.cs)

        return root

    # Determines which connections to express - high variance = more connections.
    def pruning_extraction(self, coord, p, outgoing, add_new_connection, query_cppn):
        for c in p.cs:
            # todo: don't recursively go through whole tree every time
            var = self.variance(c)
            # print(f"PRUNING EXTRACTION VAR: {var}")
            if var > self.variance_threshold:
                self.pruning_extraction(coord, c, outgoing, add_new_connection, query_cppn)
            else:
                # c's child nodes are removed (only when drawing)
                # c.cs = [None] * len(c.cs)

                coord2 = (c.x, c.y, c.z)

                # Only consider connection if
                #   weight non zero
                if c.w != 0.0 and self.check_banding_threshold(coord, coord2, c.w, p.width, outgoing, query_cppn):
                    if outgoing:
                        source = coord
                        target = coord2
                    else:
                        source = coord2
                        target = coord

                    con = Connection(source[0], source[1], source[2], target[0], target[1], target[2],
                                     c.cppn_output.get_connection_params(), p.width)
                    add_new_connection(con)

    def check_banding_threshold(self, source, target, weight, width, outgoing, query_cppn):
        # Band pruning
        d_left = abs(
            weight - query_cppn(source, (target[0] - width, target[1], target[2]), outgoing).w)
        d_right = abs(
            weight - query_cppn(source, (target[0] + width, target[1], target[2]), outgoing).w)
        d_h = min(d_left, d_right)
        # print(f"BANDING DH : {d_h}")
        create = d_h > self.band_threshold

        if not create:
            d_top = abs(
                weight - query_cppn(source, (target[0], target[1] + width, target[2]), outgoing).w)
            d_bot = abs(
                weight - query_cppn(source, (target[0], target[1] - width, target[2]), outgoing).w)
            d_v = min(d_top, d_bot)
            # print(f"BANDING DV : {d_v}")
            create = d_v > self.band_threshold

            if not create:
                d_front = abs(
                    weight - query_cppn(source, (target[0], target[1], target[2] - width), outgoing).w)
                d_back = abs(
                    weight - query_cppn(source, (target[0], target[1], target[2] + width), outgoing).w)
                d_d = min(d_front, d_back)
                # print(f"BANDING DD : {d_d}")
                create = d_d > self.band_threshold

        return create

    def es_hyperneat(self):
        """
        Explore hidden nodes and their connections.

        Note: The explore from input loops are unfolded (into joint/sensor specific and general) in order to avoid
            "if" statements in this performance critical code.
        :return:
        """
        inputs = self.substrate.input_coordinates
        outputs = self.substrate.output_coordinates
        hidden_nodes = set()
        connnections = list()

        # Performance boost by avoiding dots in loops
        append_connection = connnections.append
        append_connections = connnections.extend
        hidden_nodes_add = hidden_nodes.add
        division_initialization = self.division_initialization
        pruning_extraction = self.pruning_extraction
        check_banding_threshold = self.check_banding_threshold

        query_cppn = self.cppn.query_cppn_brain_joint_sensor_input_con
        for input_node_coord in inputs[:self.substrate.input_general_start_idx]:  # Explore from joint / sensor inputs.
            root = division_initialization(input_node_coord, True, query_cppn)
            new_connections = list()
            pruning_extraction(input_node_coord, root, True, new_connections.append, query_cppn)

            if input_node_coord in self.substrate.required_nodes and len(new_connections) == 0:
                return None, None, False

            append_connections(new_connections)
            for c in new_connections:
                hidden_nodes_add(((c.x2, c.y2, c.z2), c.qt_width))

        query_cppn = self.cppn.query_cppn_brain_general_con
        for input_node_coord in inputs[self.substrate.input_general_start_idx:]:  # Explore from general inputs.
            root = division_initialization(input_node_coord, True, query_cppn)
            new_connections = list()
            pruning_extraction(input_node_coord, root, True, new_connections.append, query_cppn)

            if input_node_coord in self.substrate.required_nodes and len(new_connections) == 0:
                return None, None, False

            append_connections(new_connections)
            for c in new_connections:
                hidden_nodes_add(((c.x2, c.y2, c.z2), c.qt_width))

        self.build_reward += 0.1

        query_cppn = self.cppn.query_cppn_brain_joint_output_con
        for output_node_coord in outputs[:self.substrate.output_general_start_idx]:  # Explore from joint outputs.
            root = division_initialization(output_node_coord, False, query_cppn)
            new_connections = list()
            pruning_extraction(output_node_coord, root, False, new_connections.append, query_cppn)

            if output_node_coord in self.substrate.required_nodes and len(new_connections) == 0:
                return None, None, False

            append_connections(new_connections)
            for c in new_connections:
                hidden_nodes_add(((c.x1, c.y1, c.z1), c.qt_width))

        query_cppn = self.cppn.query_cppn_brain_general_con
        for output_node_coord in outputs[self.substrate.output_general_start_idx:]:  # Explore from general outputs.
            root = division_initialization(output_node_coord, False, query_cppn)
            new_connections = list()
            pruning_extraction(output_node_coord, root, False, new_connections.append, query_cppn)

            if output_node_coord in self.substrate.required_nodes and len(new_connections) == 0:
                return None, None, False

            append_connections(new_connections)
            for c in new_connections:
                hidden_nodes_add(((c.x1, c.y1, c.z1), c.qt_width))

        self.build_reward += 0.1

        # Hidden nodes can be connected to other hidden nodes
        query_cppn = self.cppn.query_cppn_brain_general_con
        for source, _ in hidden_nodes:
            for target, width in hidden_nodes:
                cppn_output = query_cppn(source, target)
                # Check banding threshold
                if cppn_output.w != 0.0 and check_banding_threshold(source, target, cppn_output.w, width, True,
                                                                    query_cppn):
                    con = Connection(source[0], source[1], source[2], target[0], target[1], target[2],
                                     cppn_output.get_connection_params(),
                                     width)
                    append_connection(con)

        hidden_node_coords, connections, valid = self.clean_net(connnections)
        if valid:
            self.build_reward += 0.3

        return hidden_node_coords, connections, valid

    # Clean a net for dangling connections by intersecting paths from input nodes with paths to output.
    def clean_net(self, connections):
        if not connections:
            return None, None, False

        connected_to_inputs = set(tuple(i) for i in self.substrate.input_coordinates)
        connected_to_outputs = set(tuple(i) for i in self.substrate.output_coordinates)

        # split connections into regular and modulatory
        #   Regular connections are kept if they are on a path from input to output nodes
        #   A Modulatory connection is kept if its startpoint is somehow connnected to inputs and its endpoint is on a regular connection path from input to output

        regular_connections = set()
        modulatory_connections = set()
        mod_nodes = set()
        mod_start_to_end = defaultdict(set)

        start_nodes, end_nodes = set(), set()
        start_to_end, end_to_start = defaultdict(set), defaultdict(set)

        for c in connections:
            s = (c.x1, c.y1, c.z1)
            e = (c.x2, c.y2, c.z2)
            if c.M > 0.0:
                modulatory_connections.add(c)
                mod_nodes.add(s)
                mod_start_to_end[s].add(e)
            else:
                regular_connections.add(c)

                start_nodes.add(s)
                start_to_end[s].add(e)

                end_nodes.add(e)
                end_to_start[e].add(s)

        end_nodes_copy = end_nodes.copy()  # used for modulatory paths later

        add_happened = True
        new_nodes = connected_to_inputs.copy()
        while add_happened:  # The path from inputs.
            new_node_starts = new_nodes.intersection(start_nodes)
            start_nodes -= new_node_starts

            add_happened = bool(new_node_starts)

            new_nodes = set()
            for start_node in new_node_starts:
                connected_to_inputs.update(start_to_end[start_node])
                new_nodes.update(start_to_end[start_node])

        add_happened = True
        new_nodes = connected_to_outputs.copy()
        while add_happened:  # The path to outputs.
            new_node_ends = new_nodes.intersection(end_nodes)
            end_nodes -= new_node_ends

            add_happened = bool(new_node_ends)

            new_nodes = set()
            for end_node in new_node_ends:
                connected_to_outputs.update(end_to_start[end_node])
                new_nodes.update(end_to_start[end_node])

        true_path_nodes = connected_to_inputs.intersection(
            connected_to_outputs)  # nodes on real regularly connected paths from input to output

        # filter out non pure modulatory nodes (nodes on a real path)
        mod_nodes -= true_path_nodes
        # only keep mod nodes from which a mod connection ends in a node on a true path
        mod_nodes = set(
            [mod_node for mod_node in mod_nodes if bool(mod_start_to_end[mod_node].intersection(true_path_nodes))])

        connected_to_mod_nodes = mod_nodes.copy()

        add_happened = True
        new_nodes = mod_nodes.copy()
        while add_happened:  # The path to modulatory nodes.
            new_node_ends = new_nodes.intersection(end_nodes_copy)
            end_nodes_copy -= new_node_ends

            add_happened = bool(new_node_ends)

            new_nodes = set()
            for end_node in new_node_ends:
                connected_to_mod_nodes.update(end_to_start[end_node])
                new_nodes.update(end_to_start[end_node])

        mod_path_nodes = connected_to_inputs.intersection(
            connected_to_mod_nodes)  # nodes on real paths from input to modulatory nodes

        output_coords = set(self.substrate.output_coordinates)
        input_coords = set(self.substrate.input_coordinates)
        valid = False
        true_connections = set()
        true_nodes = true_path_nodes.copy()
        for c in regular_connections:
            # Only include regular connection if both source and target node resides in the real path from input to output
            #   or if it is on a path to a modulatory node
            start = (c.x1, c.y1, c.z1)
            end = (c.x2, c.y2, c.z2)
            if start in true_path_nodes and end in true_path_nodes:
                true_connections.add(c)
                input_coords.discard(start)
                output_coords.discard(end)
                valid = True
            elif start in mod_path_nodes and end in mod_path_nodes:
                true_nodes.add(start)
                true_connections.add(c)

        if valid:
            for c in modulatory_connections:
                # Only include modulatory connection if source node is connected in a regular way to the input and the target node is on the path from input to output
                start = (c.x1, c.y1, c.z1)
                end = (c.x2, c.y2, c.z2)
                if (start in true_path_nodes or start in mod_path_nodes) and end in true_path_nodes:
                    true_connections.add(c)
                    true_nodes.add(start)

        # Check if all required nodes are included
        valid = valid and len(self.substrate.required_nodes - true_nodes) == 0

        true_nodes -= (set(self.substrate.input_coordinates).union(set(self.substrate.output_coordinates)))

        return list(true_nodes), list(true_connections), valid


class QuadPoint:
    def __init__(self, x, y, z, width, height, depth, lvl):
        self.x = x
        self.y = y
        self.z = z
        self.w = 0.0
        self.cppn_output = None
        self.width = width
        self.height = height
        self.depth = depth
        self.cs: List[Union[Optional, QuadPoint]] = list()
        self.lvl = lvl

    def set_cppn_output(self, cppn_output):
        self.cppn_output = cppn_output
        self.w = cppn_output.w
