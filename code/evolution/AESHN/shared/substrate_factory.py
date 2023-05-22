import math
from decimal import Decimal
from typing import Tuple

import numpy as np


class Substrate(object):
    """
    Base substrate class.
    """
    def __init__(self, input_coordinates, output_coordinates, hidden_dims, origin=(0.0, 0.0, 0.0)):
        self.input_coordinates = input_coordinates
        self.output_coordinates = output_coordinates
        self.width, self.height, self.depth = hidden_dims
        self.origin = origin


class DGCBSubstrate(Substrate):
    """
    Helper class to represent the agent substrate.

    The general_start_idx attributes hold the indices on which the node coordinate lists switch from
        joint / sensor inputs and outputs to general inputs and outputs.
    """

    def __init__(self, input_coordinates, input_general_start_idx, output_coordinates, output_general_start_idx,
                 hidden_dims, required, origin=(0.0, 0.0, 0.0)):
        super(DGCBSubstrate, self).__init__(input_coordinates, output_coordinates, hidden_dims, origin)
        self.input_general_start_idx = input_general_start_idx
        self.output_general_start_idx = output_general_start_idx
        self.required_nodes = set()
        if required["inputs"]:
            self.required_nodes = self.required_nodes.union(self.input_coordinates)
        if required["outputs"]:
            self.required_nodes = self.required_nodes.union(self.output_coordinates)

    def add_general_input_node(self, coord: Tuple[float, float, float], required=False) -> None:
        self.input_coordinates.append(coord)
        if required:
            self.required_nodes.add(coord)

    def add_general_output_node(self, coord: Tuple[float, float, float], required=False) -> None:
        self.output_coordinates.append(coord)
        if required:
            self.required_nodes.add(coord)


def create_substrate_2d() -> Substrate:
    def add_observations_2d(start_height, width, left_bound, rays, obs_per_ray, coords):
        for i in range(1, rays + 1):
            x = float(Decimal(left_bound + i * width / (rays + 1)).quantize(
                Decimal('1e-4')))  # ordinary python round doesn't do the job
            for j in range(1, obs_per_ray + 1):
                y = float(start_height - j * 0.2)
                coords.append((x, y))

        return math.floor(start_height - 0.2 * (obs_per_ray + 1))

    left_bound = -1
    right_bound = 1
    width = right_bound - left_bound

    # Network input coordinates.
    next_height = -1.5
    input_coordinates = list()
    #   Top left to right ray casts - y range ]-3.0, -1.5]
    next_height = add_observations_2d(next_height, width, left_bound, 5, 6, input_coordinates)
    #   Middle left to right ray casts - y range [-3.7, -3.0]
    next_height = add_observations_2d(next_height, width, left_bound, 3, 4, input_coordinates)
    #   Bottom left to right ray casts - y range [-4.4 , -3.7]
    add_observations_2d(next_height, width, left_bound, 3, 4, input_coordinates)

    #   Energy state - x range [-1, 0] - y range ]-1.5, -1.0]
    add_observations_2d(-1, 1, -1, 3, 1, input_coordinates)
    #   Motion state - x range [0, 1] - y range ]-1.5, -1.0]
    add_observations_2d(-1, 1, 0, 3, 1, input_coordinates)

    # Network output coordinates
    output_coordinates = list()
    #   Forward motion - [NOP, backward, forward]
    output_coordinates += [(0.0, 1.4), (0.0, 1.2), (0.0, 1.6)]
    #   Side motion    - [NOP, left, right]
    output_coordinates += [(-0.1, 1.4), (left_bound, 1.4), (right_bound, 1.4)]
    #   Rotation       - [NOP, left, right]
    output_coordinates += [(0.1, 1.4), (left_bound / 2, 1.4), (right_bound / 2, 1.4)]
    #   Action         - [NOP, jump, eat, drink]
    output_coordinates += [(0.0, 2.0), (0.7, 2.0), (-0.8, 2.0), (-0.6, 2.0)]

    input_coordinates = [(x, y, 0) for x, y in input_coordinates]
    output_coordinates = [(x, y, 0) for x, y in output_coordinates]

    return Substrate(input_coordinates, output_coordinates, (right_bound - left_bound, 0, 0))


def create_substrate_3d() -> Substrate:
    def add_observations_3d(start_height, width, left_bound, start_depth, max_extra_depth, rays, obs_per_ray, coords):
        if rays % 2 == 0:
            xp = [1, rays // 2 + 0.5, rays]
        else:
            xp = [1, rays // 2 + 1, rays]
        fp = [start_depth, start_depth + max_extra_depth, start_depth]

        for i in range(1, rays + 1):
            x = float(Decimal(left_bound + i * width / (rays + 1)).quantize(
                Decimal('1e-4')))  # ordinary python round doesn't do the job
            z = float(Decimal(np.interp(i, xp=xp, fp=fp)).quantize(Decimal('1e-4')))

            for j in range(1, obs_per_ray + 1):
                y = float(Decimal(start_height - j * 0.1).quantize(Decimal('1e-4')))
                coords.append((x, y, z))

        return float(Decimal(start_height - (obs_per_ray + 1) * 0.1).quantize(Decimal('1e-4')))

    front_bound = -1.0
    back_bound = 1.0
    left_bound = -1.0
    right_bound = 1.0
    top_bound = 1.0
    bottom_bound = -1.0

    width = right_bound - left_bound
    height = top_bound - bottom_bound
    depth = back_bound - front_bound

    # Network input coordinates.
    next_height = 0.8
    input_coordinates = list()
    #   Top left to right ray casts - y range ]0.1, 0.75]
    next_height = add_observations_3d(next_height, width, left_bound, -1.3, 0.3, 5, 6, input_coordinates)
    #   Middle left to right ray casts - y range ]-0.4, 0.1]
    next_height = add_observations_3d(next_height, width, left_bound, -1.3, 0.2, 3, 4, input_coordinates)
    #   Bottom left to right ray casts - y range [-0.7, -0.4]
    add_observations_3d(next_height, width, left_bound, -1.3, 0.1, 3, 4, input_coordinates)

    #   Energy state
    input_coordinates.append((-0.55, 1.0, -1.05))  # health
    input_coordinates.append((-0.5, 0.9, -1.05))  # food
    input_coordinates.append((-0.6, 0.9, -1.05))  # drink

    #   Motion state (velocities) - x range [0, 1] - y range ]-1.5, -1.0]
    input_coordinates.append((0.55, 0.9, -1.2))  # left
    input_coordinates.append((0.6, 0.9, -1.15))  # forward
    input_coordinates.append((0.6, 0.95, -1.2))  # up

    # Network output coordinates
    output_coordinates = list()
    #   Forward motion - [NOP, backward, forward]
    output_coordinates += [(0.0, 0.0, 1.3), (0.0, 0.0, 1.1), (0.0, 0.0, 1.4)]
    #   Side motion    - [NOP, left, right]
    output_coordinates += [(0.0, 0.0, 1.35), (left_bound, 0.0, 1.3), (right_bound, 0.0, 1.3)]
    #   Rotation       - [NOP, left, right]
    output_coordinates += [(0.0, 0.0, 1.25), (left_bound / 2, 0.0, 1.1), (right_bound / 2, 0.0, 1.1)]
    #   Action         - [NOP, jump, eat, drink]
    output_coordinates += [(0.0, 0.8, 1.3), (0.0, 0.5, 1.3), (-0.5, 0.9, 1.05), (-0.6, 0.9, 1.05)]

    return Substrate(input_coordinates, output_coordinates, (width / 2, height / 2, depth / 2))


def create_substrate(dimensions: int = 2) -> Substrate:
    if dimensions == 3:
        return create_substrate_3d()
    else:
        return create_substrate_2d()


if __name__ == '__main__':
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    substrate = create_substrate(True)
    inp = substrate.input_coordinates + substrate.output_coordinates
    print(len(inp))
    x, y, z = zip(*inp)
    ax.scatter3D(x, y, z)
    plt.show()
