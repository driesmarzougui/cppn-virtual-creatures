from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Tuple, Set

import numpy as np
from AESHN.shared.cppn.cppn import CPPN
from AESHN.shared.substrate_factory import Substrate, DGCBSubstrate
import json
import sys

from evolution.utils.nb_math_utils import distance_from_origin

DIRECTIONS = [np.array([1, 0, 0]), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, -1, 0]),
              np.array([0, 0, 1]), np.array([0, 0, -1])]

RAYS_PER_DIRECTION = 0
NUM_DETECTABLE_TAGS = 4


class CPPNMorph(object):
    """
    Helper class that encapsulates the CPPN-based creation of creature morphologies.
    This class also takes care of creating the initial brain substrate and morphology-dependent input and output nodes.
    """

    def __init__(self, genome_id: int, params: Dict, cppn: CPPN):
        self.genome_id = genome_id
        self.params = params
        self.cppn = cppn

        self.m_brain_loc = None
        self.brain_location = None
        self.input_node_locations = list()
        self.output_node_locations = list()
        self.substrate: DGCBSubstrate = None

        self.build_reward = 0

        # Create matrix morphology representation
        self.ws, self.hs, self.ds = self.params["morph_params"]["space_size"]
        self.sub_s = self.params["morph_params"]["subspace_size"]
        self.sub_ws, self.sub_hs, self.sub_ds = self.sub_s
        self.m_w, self.m_h, self.m_d = (
            int(self.ws / self.sub_ws), int(self.hs / self.sub_hs), int(self.ds / self.sub_ds))
        self.x_start, self.y_start, self.z_start = -self.ws / 2 + self.sub_ws / 2, -self.hs / 2 + self.sub_hs / 2, -self.ds / 2 + self.sub_ds / 2

        self.m_morph_tmp = np.zeros((self.m_w, self.m_h, self.m_d))  # 0 : NB, 1: FB, 2: SB, 3: BB
        self.m_joint_loc_to_params = dict()

        self.m_morph = np.zeros((self.m_w, self.m_h, self.m_d))
        self.morph: List[Dict] = list()

        self.block_locations: List[Tuple[float, float, float]] = list()
        self.configurable_joint_locations = list()
        self.m_joints = np.zeros((self.m_w, self.m_h, self.m_d))
        self.lowest_level = 1000

    def m_loc_to_agent_space_loc(self, m_x, m_y, m_z) -> Tuple[float, float, float]:
        return self.x_start + self.sub_ws * m_x, self.y_start + self.sub_hs * m_y, self.z_start + self.sub_ds * m_z

    def recursive_query_body_blocks(self, m_loc, direction, m_block_locations, m_sb_to_act_sbd: Dict) -> None:
        """
        Recursively build the morphology matrix by starting from the given location and exploring the given direction.
        This should only be used when the brain has a static location.

        This function places the resulting morphology matrix representation in self.m_morph, while using
        m_morph_tmp to check if a location has been visited yet

        :param loc: current matrix location
        :param direction: direction to explore
        :param m_block_locations: current list of matrix positions of connected blocks
        :param m_sb_to_act_sbd: Dict mapping matrix sensor block location to activation and sensor block direction (SBD)
        :return:
        """
        try:
            m_new_loc = m_loc + direction
            m_x, m_y, m_z = m_new_loc
            # Check if location is within bounds and hasn't been visited yet
            if all(m_new_loc >= 0) and self.m_morph_tmp[m_x, m_y, m_z] == 0 and m_y < self.m_brain_loc[1]:
                self.m_morph_tmp[m_x, m_y, m_z] = 1

                # Get block type from CPPN
                coord_b = self.m_loc_to_agent_space_loc(m_x, m_y, m_z)
                cppn_output = self.cppn.query_cppn_morph(coord_b)
                block_type = cppn_output.get_block_type()

                if block_type != 0:
                    if m_y < self.lowest_level:
                        self.lowest_level = m_y
                    self.m_morph[m_x, m_y, m_z] = block_type
                    self.block_locations.append(coord_b)
                    m_block_locations.append(m_new_loc)

                    if block_type == 2:
                        m_sb_to_act_sbd[(m_x, m_y, m_z)] = (cppn_output.SB, cppn_output.get_sensor_block_direction())
                    # Continue search
                    for new_direction in DIRECTIONS:
                        self.recursive_query_body_blocks(m_new_loc, new_direction, m_block_locations,
                                                         m_sb_to_act_sbd)
        except IndexError:
            # position out of range
            pass

    def query_body_blocks(self, m_sb_to_sbd: Dict) -> bool:
        """
        Query block for every subspace in agent space.
        This also find out which subspace is the brain (highest activation) and sets its location for later use.

        :param m_sb_to_sbd: Dict mapping matrix sensor block location to sensor block direction (SBD)
        :return: bool indicating if a brain was found (no brain --> invalid morphology)
        """
        max_brain_act = -20
        for m_x in range(self.m_w):
            for m_y in range(self.m_h):
                for m_z in range(self.m_d):
                    coord_b = self.m_loc_to_agent_space_loc(m_x, m_y, m_z)

                    cppn_output = self.cppn.query_cppn_morph(coord_b)

                    blocks_and_activation = cppn_output.get_block_types_and_activations()
                    block_type, activation = blocks_and_activation[0]

                    if block_type == 3:
                        # Brain block
                        if activation > max_brain_act:
                            max_brain_act = activation
                            self.brain_location = coord_b
                            self.m_brain_loc = m_x, m_y, m_z

                        # Temporarily set block with second highest activation todo: maybe set filler block instead?
                        block_type = blocks_and_activation[1][0]

                    if block_type == 2:
                        m_sb_to_sbd[(m_x, m_y, m_z)] = (cppn_output.SB, cppn_output.get_sensor_block_direction())

                    self.m_morph_tmp[m_x, m_y, m_z] = block_type

        if self.brain_location is not None:
            m_bx, m_by, m_bz = self.m_brain_loc
            self.m_morph_tmp[m_bx, m_by, m_bz] = 0
            self.m_morph[m_bx, m_by, m_bz] = 3
            self.m_brain_loc = np.asarray(self.m_brain_loc)
            m_sb_to_sbd.pop((m_bx, m_by, m_bz), None)
            return True

        # No brain -> invalid morphology
        return False

    def recursive_connected_blocks_search(self, m_loc, direction, m_block_locations, m_sb_locs: List) -> None:
        """
        Recursively search through the matrix morphology representation to find blocks that are connected to the
        brain.

        :param m_loc: previous matrix location
        :param direction: direction to start searching into
        :param m_block_locations: current list of connected block matrix locations
        :param m_sb_locs: current list of sensor block matrix locations
        :return:
        """
        try:
            m_new_loc = m_loc + direction
            m_x, m_y, m_z = m_new_loc
            # Check if block isn't empty
            if all(m_new_loc >= 0) and self.m_morph_tmp[m_x, m_y, m_z] != 0:
                m_block_locations.append(m_new_loc)
                self.m_morph[m_x, m_y, m_z] = self.m_morph_tmp[m_x, m_y, m_z]
                self.m_morph_tmp[m_x, m_y, m_z] = 0

                self.block_locations.append(self.m_loc_to_agent_space_loc(m_x, m_y, m_z))
                if self.m_morph[m_x, m_y, m_z] == 2:
                    m_sb_locs.append((m_x, m_y, m_z))

                # Continue search
                for new_direction in DIRECTIONS:
                    self.recursive_connected_blocks_search(m_new_loc, new_direction, m_block_locations,
                                                           m_sb_locs)
        except IndexError:
            # position out of range
            pass

    def filter_sensor_blocks(self, m_sb_to_act_sbd: Dict[Tuple[int, int, int], Tuple[float, int]]) \
            -> Dict[Tuple[int, int, int], int]:
        """
        Filters sensor blocks based on valid sensor direction.
        If a sensor block's view direction is blocked by another block, this function replaces it with a filler block;
            otherwise it's agent space location is added to the list of input node locations

        This function also makes sure that maximum 16 sensor blocks are used.
        This function also adds the input observation nodes for the brain.

        :param m_sb_to_act_sbd: dict mapping sensor block matrix location to it's cppn activation and view direction
        :return: dict mapping every matrix sensor block location to its target face direction
        """
        m_sb_to_sbd_clean = dict()

        # Sort according to activation
        sensor_blocks = sorted([(m_loc, act, sbd) for m_loc, (act, sbd) in m_sb_to_act_sbd.items()], key=lambda x: x[1],
                               reverse=True)

        for m_loc, _, sbd in sensor_blocks:
            m_x, m_y, m_z = m_loc

            if len(m_sb_to_sbd_clean) < 16:
                sensor_direction = DIRECTIONS[sbd]
                m_neighbour_loc = np.asarray(m_loc) + sensor_direction
                m_nx, m_ny, m_nz = m_neighbour_loc

                try:
                    if all(m_neighbour_loc >= 0) and self.m_morph[m_nx, m_ny, m_nz] != 0:
                        # Something blocks the sensor view direction -> convert to FB
                        self.m_morph[m_x, m_y, m_z] = 1
                    else:
                        m_sb_to_sbd_clean[m_loc] = sbd
                        self.add_sensor_block_input_nodes(m_loc, sensor_direction)
                except IndexError:
                    m_sb_to_sbd_clean[m_loc] = sbd
                    self.add_sensor_block_input_nodes(m_loc, sensor_direction)
            else:
                self.m_morph[m_x, m_y, m_z] = 1

        return m_sb_to_sbd_clean

    def add_sensor_block_input_nodes(self, m_sb_loc: Tuple[int, int, int], direction: np.ndarray) -> None:
        """
        Add input nodes for the given ray sensor block.
        Number of nodes: (1 + 2 * RAYS_PER_DIRECTION) * (NUM_DETECTABLE_TAGS + 2)
        cfr https://github.com/Unity-Technologies/ml-agents/blob/main/com.unity.ml-agents/Runtime/Sensors/RayPerceptionSensor.cs#L170

        This makes sure that nodes are placed perpendicular to the sensor view direction.

        :param m_sb_loc: matrix position of sensor block
        :param direction: sensor direction vector
        :return:
        """
        m_x, m_y, m_z = m_sb_loc
        sb_loc = self.m_loc_to_agent_space_loc(m_x, m_y, m_z)

        num_rays = 1 + 2 * RAYS_PER_DIRECTION
        num_nodes_per_ray = 2 + NUM_DETECTABLE_TAGS

        view_axis = int(np.argmax(np.abs(direction)))
        # Get ray dimension positions (perpendicular to sensor view direction)
        ray_dimension_index = 2 if view_axis == 0 else 0
        ray_dim_loc = sb_loc[ray_dimension_index]
        ray_dim_bound = 0.1 * num_rays / 2

        # Get node dimension positions (perpendicular to sensor view direction and ray dimension)
        node_dimension_index = 2 if view_axis == 1 else 1
        node_dim_loc = sb_loc[node_dimension_index]
        node_dim_bound = 0.2 * num_nodes_per_ray / 2

        ray_locs = np.linspace(ray_dim_loc - ray_dim_bound, ray_dim_loc + ray_dim_bound, num_rays)
        node_locs = np.linspace(node_dim_loc - node_dim_bound, node_dim_loc + node_dim_bound, num_nodes_per_ray)

        view_axis_loc_offset = 0.4
        if -1 in direction:
            # Reverse locations if view direction is negative
            ray_locs = ray_locs[::-1]
            node_locs = node_locs[::-1]
            view_axis_loc_offset *= -1

        base_loc = list(sb_loc)
        base_loc[view_axis] += view_axis_loc_offset
        for ray_loc in ray_locs:
            loc = base_loc.copy()
            [x + (s / 2) - 0.1 for s, x in zip(self.sub_s, sb_loc)]
            loc[ray_dimension_index] = ray_loc
            for node_loc in node_locs:
                loc[node_dimension_index] = node_loc

                self.input_node_locations.append(tuple(loc))

    def get_block_neighbours(self, m_block_locations) -> Dict:
        """
        Get neighbour matrix block locations for every matrix block location

        :return: dict mapping matrix block location to a set of neighbouring block locations
        """
        m_block_neighbours = defaultdict(set)
        for m_loc in m_block_locations:
            # Get neighbours
            for direction in DIRECTIONS:
                try:
                    m_neighbour_loc = m_loc + direction
                    m_nx, m_ny, m_nz = m_neighbour_loc
                    if all(m_neighbour_loc >= 0) and self.m_morph[m_nx, m_ny, m_nz] > 0:
                        m_block_neighbours[(m_loc[0], m_loc[1], m_loc[2])].add((m_nx, m_ny, m_nz))
                except IndexError:
                    pass

        return m_block_neighbours

    def get_mount_blocks(self, m_block_neighbours: Dict, mount_blocks: List) -> bool:
        """
        Identify and retrieve all 'mount blocks':
            blocks with type 1 (filler block) and 2 neighbours

        :param m_block_neighbours:
        :return: set of mount block matrix locations
        """
        for m_loc, neighbours in m_block_neighbours.items():
            if self.m_morph[m_loc[0], m_loc[1], m_loc[2]] == 1 and len(neighbours) == 2:
                n1, n2 = neighbours
                if len(m_block_neighbours[n1].intersection(m_block_neighbours[n2])) == 1:
                    if not self.params["morph_params"]["disallow_bottom_mount_blocks"] or m_loc[1] > self.lowest_level:
                        mount_blocks.append(m_loc)

        return len(mount_blocks) > 0

    def build_substrate(self) -> None:
        """
        Create the brain substrate space based on the acquired brain location.

        All joint / sensor block inputs should have been added by this stage, so the general input and output
            start indices can be set here as the lengths of the current input and output node coordinate lists.
        """
        self.substrate = DGCBSubstrate(input_coordinates=self.input_node_locations,
                                       input_general_start_idx=len(self.input_node_locations),
                                       output_coordinates=self.output_node_locations,
                                       output_general_start_idx=len(self.output_node_locations),
                                       hidden_dims=(1, 1, 1),  # Default brain substrate dims
                                       origin=self.brain_location,
                                       required=self.params["es_params"]["prune_not_fully_connected"]
                                       )

    def get_configurable_joints(self, m_block_neighbours, mount_blocks) -> Dict[Tuple, Dict]:
        """
        Get the blocks that should be connected using a configurable joint (CJ).

        Constraints:
            Only mount blocks can connect to another block using a CJ
            Maximum 6 CJs in total
            Max 1 CJ per block
            No bidirectional CJ between two blocks
            No brain block (input / output nodes might otherwise interfere with substrate)

        :return: Dict mapping tuples (block a, block b) to joint dicts meaning that block a connects to block b using a CJ with the parameters in the corresponding dict value
        """
        cj_blocks_to_activation = dict()
        block_to_cj_blocks_to_params = defaultdict(dict)

        for mount_block in mount_blocks:
            m_x, m_y, m_z = mount_block
            x, y, z = self.m_loc_to_agent_space_loc(m_x, m_y, m_z)

            for neighbour in m_block_neighbours[mount_block]:
                m_nx, m_ny, m_nz = neighbour
                xn, yn, zn = self.m_loc_to_agent_space_loc(m_nx, m_ny, m_nz)

                coord_b = (np.asarray((xn, yn, zn)) + np.asarray((x, y, z))) / 2
                cppn_output = self.cppn.query_cppn_morph(coord_b)

                joint_type, lax, hax, ayl, azl = cppn_output.get_joint_type(
                    self.params["morph_params"]["cjp_low_limit"],
                    self.params["morph_params"]["cjp_high_limit"])

                if joint_type == 1 and self.m_morph[m_nx, m_ny, m_nz] != 3:  # No CJ connection to brain
                    cj_blocks_to_activation[(mount_block, neighbour)] = cppn_output.CJ
                    dx, dy, dz = np.asarray(neighbour) - np.asarray(mount_block)
                    block_to_cj_blocks_to_params[mount_block][neighbour] = {"type": joint_type,
                                                                            "dx": int(dx), "dy": int(dy), "dz": int(dz),
                                                                            "lax": lax, "hax": hax, "ayl": ayl,
                                                                            "azl": azl}

        if len(cj_blocks_to_activation) > 0:
            # Only 1 CJ per block
            for mount_block, cj_blocks_to_params in block_to_cj_blocks_to_params.items():
                if len(cj_blocks_to_params) == 2:
                    # Pick the one with the highest activation or closest from brain if activation is equal
                    #   We use the closest to try to make the configurable joints flow in "an outward direction"
                    #   starting from the brain
                    target1, target2 = list(cj_blocks_to_params.keys())

                    dir1 = (mount_block, target1)
                    dir2 = (mount_block, target2)

                    act1 = cj_blocks_to_activation[dir1]
                    act2 = cj_blocks_to_activation[dir2]

                    t1_l = distance_from_origin(target1)
                    t2_l = distance_from_origin(target2)

                    if act1 > act2 or (act1 == act2 and t1_l <= t2_l):
                        cj_blocks_to_activation.pop(dir2)
                        block_to_cj_blocks_to_params[mount_block].pop(target2)
                    else:
                        cj_blocks_to_activation.pop(dir1)
                        block_to_cj_blocks_to_params[mount_block].pop(target1)

            # No bidirectional CJ between two blocks
            # todo: shouldn't be running over all mount blocks here, only the ones that currently still have a CJ
            for i in range(len(mount_blocks) - 1):
                for j in range(i + 1, len(mount_blocks)):
                    mount_block1 = mount_blocks[i]
                    mount_block2 = mount_blocks[j]

                    dir1 = (mount_block1, mount_block2)
                    dir2 = (mount_block2, mount_block1)

                    if dir1 in cj_blocks_to_activation and dir2 in cj_blocks_to_activation:
                        # Bidirectional CJ connection found
                        #   activation will be equal as queried joint location is equal
                        #       connect the mount block that is further from origin to the mount block that is closest
                        mb1_l = distance_from_origin(mount_block1)
                        mb2_l = distance_from_origin(mount_block2)

                        if mb1_l <= mb2_l:
                            # Mount block 1 is closer to origin --> attach mount block 2 to mount block 1
                            cj_blocks_to_activation.pop(dir1)
                            block_to_cj_blocks_to_params[mount_block1].pop(mount_block2)
                        else:
                            # Mount block 2 is closer to origin --> attach mount block 1 to mount block 2
                            cj_blocks_to_activation.pop(dir2)
                            block_to_cj_blocks_to_params[mount_block2].pop(mount_block1)

            if len(cj_blocks_to_activation) > 16:
                # Pick highest 16 activations
                final_cjs = sorted(cj_blocks_to_activation, key=cj_blocks_to_activation.get, reverse=True)[:16]
            else:
                final_cjs = list(cj_blocks_to_activation.keys())

            return {
                (mount_block, neighbour): block_to_cj_blocks_to_params[mount_block][neighbour] for
                (mount_block, neighbour) in final_cjs
            }
        else:
            return dict()

    def build_morphology(self) -> bool:
        sys.setrecursionlimit(1600)

        m_sb_to_act_sbd = dict()

        if self.params["morph_params"]["fix_brain"]:
            # Brain is always located at (0, 0, 0)
            m_bx, m_by, m_bz = self.m_w // 2, self.m_h // 2, self.m_d // 2
            self.m_brain_loc = np.asarray((m_bx, m_by, m_bz))
            self.brain_location = (0.0, 0.0, 0.0)
            self.block_locations.append(self.brain_location)

            self.m_morph[m_bx, m_by, m_bz] = 3
            self.m_morph_tmp[m_bx, m_by, m_bz] = 1

            m_block_locations = [self.m_brain_loc]  # list of matrix block locations

            self.recursive_query_body_blocks(self.m_brain_loc, np.array([0, -1, 0]), m_block_locations,
                                             m_sb_to_act_sbd)

        else:
            if not self.query_body_blocks(m_sb_to_act_sbd):
                # No brain location -> Invalid morphology
                return False

            m_block_locations = [self.m_brain_loc]  # list of matrix block locations
            m_sb_locs = list()
            for direction in DIRECTIONS:
                self.recursive_connected_blocks_search(self.m_brain_loc, direction, m_block_locations,
                                                       m_sb_locs)
            m_sb_to_act_sbd = {m_loc: m_sb_to_act_sbd[m_loc] for m_loc in m_sb_locs}


        # Add sensor input nodes
        edge_plane = 1.15
        #   North
        for crd1 in np.linspace(-0.3, 0.3, 3):
            for crd2 in np.linspace(-0.3, 0.3, 6):
                self.input_node_locations.append((crd1, crd2, edge_plane))
        #   East
        for crd1 in np.linspace(-0.3, 0.3, 3):
            for crd2 in np.linspace(-0.3, 0.3, 6):
                self.input_node_locations.append((edge_plane, crd2, crd1))
        #   South
        for crd1 in np.linspace(-0.3, 0.3, 3):
            for crd2 in np.linspace(-0.3, 0.3, 6):
                self.input_node_locations.append((crd1, crd2, -edge_plane))
        #   West
        for crd1 in np.linspace(-0.3, 0.3, 3):
            for crd2 in np.linspace(-0.3, 0.3, 6):
                self.input_node_locations.append((-edge_plane, crd2, crd1))

        m_sb_to_dir: Dict[Tuple[int, int, int], int] = dict() #self.filter_sensor_blocks(m_sb_to_act_sbd)

        m_block_neighbours = self.get_block_neighbours(m_block_locations)

        mount_blocks = list()
        if not self.get_mount_blocks(m_block_neighbours, mount_blocks):
            # Not movable morphology -> invalid morphology
            return False

        self.build_reward += 0.2

        blocks_to_cjs = self.get_configurable_joints(m_block_neighbours, mount_blocks)

        if len(blocks_to_cjs) == 0:
            # No CJs --> not movable morphology -> invalid morphology
            return False

        self.build_reward += 0.3

        # Create side channel list representation and set joint types and output nodes
        sensor_block_order = list()
        for m_loc, neighbours in m_block_neighbours.items():
            m_x, m_y, m_z = m_loc
            block_type = int(self.m_morph[m_x, m_y, m_z])
            joints = list()

            # Get joints
            for neighbour in neighbours:
                # Check if CJ
                if (m_loc, neighbour) in blocks_to_cjs:
                    joints.append(blocks_to_cjs[(m_loc, neighbour)])

                    # Joint input and output nodes are placed around the center of the CJ connected mount block
                    j_x, j_y, j_z = self.m_loc_to_agent_space_loc(m_x, m_y, m_z)

                    #   Configurable joints define observations -> add input nodes (3 position, 3 rotation)
                    self.input_node_locations.append((j_x + 0.1, j_y, j_z))  # x pos
                    self.input_node_locations.append((j_x, j_y + 0.1, j_z))  # y pos
                    self.input_node_locations.append((j_x, j_y, j_z + 0.1))  # z pos
                    self.input_node_locations.append((j_x + 0.2, j_y, j_z))  # x relative angle
                    self.input_node_locations.append((j_x, j_y + 0.2, j_z))  # y relative angle
                    self.input_node_locations.append((j_x, j_y, j_z + 0.2))  # z relative angle

                    #   Configurable joints define actions
                    if self.params["experiment"]["dgcb"]:
                        # 1 Brain Joint Output
                        self.output_node_locations.append((j_x, j_y, j_z))
                    else:
                        self.output_node_locations.append((j_x - 0.1, j_y, j_z))  # X joint output
                        self.output_node_locations.append((j_x, j_y - 0.1, j_z))  # Y joint output
                        self.output_node_locations.append((j_x, j_y, j_z - 0.1))  # Z joint output

                    # Add joint location to list of used configurable joints
                    self.configurable_joint_locations.append((j_x, j_y, j_z))
                    self.m_joints[m_x, m_y, m_z] = 1

                elif (neighbour, m_loc) not in blocks_to_cjs:
                    # Only add a FJ if the neighbour doesn't connect to this block with a CJ
                    dx, dy, dz = np.asarray(neighbour) - np.asarray(m_loc)
                    joints.append(
                        {"type": 0,
                         "dx": int(dx), "dy": int(dy), "dz": int(dz),
                         "lax": 0, "hax": 0, "ayl": 0, "azl": 0}
                    )

            if len(joints) == 0:
                # Add dummy joint
                joints.append(
                    {"type": -1,
                     "dx": 0, "dy": 0, "dz": 0,
                     "lax": 0, "hax": 0, "ayl": 0, "azl": 0}
                )

            if block_type == 2:
                sensor_block_order.append(m_loc)

            self.morph.append(
                {"type": block_type, "x": int(m_x), "y": int(m_y), "z": int(m_z), "sbd": m_sb_to_dir.get(m_loc, -1),
                 "joints": joints}
            )

        # Final (just for safety) check:
        if len(self.output_node_locations) == 0 or len(self.input_node_locations) == 0:
            return False

        # build substrate
        self.build_substrate()

        return True

    def __str__(self) -> str:
        json_repr = {"genomeId": int(self.genome_id),
                     "agentSpaceWidth": self.ws, "agentSpaceHeight": self.hs, "agentSpaceDepth": self.ds,
                     "agentSubSpaceWidth": self.sub_ws, "agentSubSpaceHeight": self.sub_hs,
                     "agentSubSpaceDepth": self.sub_ds,
                     "blocks": self.morph}

        return json.dumps(json_repr)
