from typing import Tuple, List

import numpy as np

from MORPH.CPPNMorph import CPPNMorph
import random
import math


class Behavior:
    def __init__(self, dummy_behavior, size):
        self.dummy_behavior = dummy_behavior
        self.size = size

    def dummy(self):
        return [self.dummy_behavior] * self.size

    def step(self, actions, obs=None):
        pass

    def add(self, behavior):
        pass

    def set(self, behavior):
        pass

    def random_sample(self):
        pass


class DistanceBehavior(Behavior):
    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, 1)
        self.behavior = self.dummy()

    def set(self, behavior):
        self.behavior = behavior

    def end(self):
        return self.behavior


class RewardMorphBehavior(Behavior):
    """ Length = 17
    Behavior characterisation that includes:
        0   : current reward
        1-9 : Relative amount of configurable joints per quadrant
        9-17: Relative amount of filler blocks per quadrant
    """

    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, size)
        self.behavior = self.dummy()

    def set(self, behavior):
        try:
            reward = float(behavior)
            # Reward was given
            self.behavior[0] = reward
        except TypeError:
            # Morph was given -> count number of cj's and number of blocks in each quadrant of agent space
            morph: CPPNMorph = behavior
            x_mid, y_mid, z_mid = morph.m_morph.shape
            x_mid, y_mid, z_mid = x_mid // 2, y_mid // 2, z_mid // 2

            # Relative amount of configurable joints per quadrant
            m_j_quadrants = [
                morph.m_joints[:x_mid, :y_mid, z_mid:],
                morph.m_joints[:x_mid, y_mid:, :z_mid],
                morph.m_joints[:x_mid, :y_mid, :z_mid],
                morph.m_joints[:x_mid, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, z_mid:],
                morph.m_joints[x_mid:, y_mid:, :z_mid],
                morph.m_joints[x_mid:, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, :z_mid]
            ]
            self.behavior[1:9] = [np.sum(j_quadrant != 0) / 16 for j_quadrant in m_j_quadrants]

            # Relative amount of blocks per quadrant
            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, y_mid:, :z_mid],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[:x_mid, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, y_mid:, :z_mid],
                morph.m_morph[x_mid:, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            self.behavior[9:] = [np.sum(b_quadrant != 0) / b_quadrant.size for b_quadrant in m_b_quadrants]

    def add(self, reward):
        reward = float(reward)
        self.behavior[0] += reward

    def end(self):
        # Scale reward
        self.behavior[0] *= 10
        return self.behavior


class MorphBehavior(Behavior):
    """
    Length: 16
        Normalized amount of filler blocks per quadrant (dimension size: 8)
        Normalized amount of configurable joints per quadrant (dimension size: 8)
    """

    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, size)
        self.behavior = self.dummy()

    def set(self, behavior: CPPNMorph):
        if isinstance(behavior, CPPNMorph):
            morph: CPPNMorph = behavior
            bc = list()

            w, h, d = morph.m_morph.shape
            x_mid, y_mid, z_mid = w // 2, h // 2, d // 2

            # Normalized amount of blocks per quadrant
            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, y_mid:, :z_mid],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[:x_mid, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, y_mid:, :z_mid],
                morph.m_morph[x_mid:, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(b_quadrant != 0) / b_quadrant.size for b_quadrant in m_b_quadrants])

            # Normalized amount of configurable joints per quadrant
            m_j_quadrants = [
                morph.m_joints[:x_mid, :y_mid, z_mid:],
                morph.m_joints[:x_mid, y_mid:, :z_mid],
                morph.m_joints[:x_mid, :y_mid, :z_mid],
                morph.m_joints[:x_mid, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, z_mid:],
                morph.m_joints[x_mid:, y_mid:, :z_mid],
                morph.m_joints[x_mid:, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(j_quadrant != 0) / 16 for j_quadrant in m_j_quadrants])

            self.behavior = bc
        else:
            raise ValueError("MorphBehavior expects to be set with a CPPNMorph")

    def end(self):
        return [x * 2 for x in self.behavior]

    def random_sample(self):
        return [random.random() ** 3 for _ in range(self.size)]


class CrawlerDirectionTrajectory2DMorphBehavior(Behavior):
    """
    Length: 28
        Normalized amount of filler blocks per quadrant (dimension size: 4)
        Normalized amount of configurable joints per quadrant (dimension size: 4)
        10 (direction, distance) samples (dimension size: 20 = 2 * 10)
            direction is the normalized angle on unit sphere
    """

    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, size)
        self.sample_interval = sample_interval
        self.step_count = 0
        self.behavior = self.dummy()
        self.last_location = (0, 0)
        self.trajectory = list()

    def set(self, behavior: CPPNMorph):
        if isinstance(behavior, CPPNMorph):
            morph: CPPNMorph = behavior
            bc = list()

            w, h, d = morph.m_morph.shape
            x_mid, y_mid, z_mid = w // 2, h // 2, d // 2

            # Normalized amount of blocks per quadrant
            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(b_quadrant == 1) / b_quadrant.size for b_quadrant in m_b_quadrants])

            # Normalized amount of configurable joints per quadrant
            m_j_quadrants = [
                morph.m_joints[:x_mid, :y_mid, z_mid:],
                morph.m_joints[:x_mid, :y_mid, :z_mid],
                morph.m_joints[x_mid:, :y_mid, z_mid:],
                morph.m_joints[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(j_quadrant != 0) / 4 for j_quadrant in m_j_quadrants])

            self.behavior = bc
        else:
            raise ValueError(f"{DirectionTrajectory2DMorphBehavior} expects to be set with a CPPNMorph")

    def add(self, behavior: Tuple[float, float]):
        self.step_count += 1
        if (self.step_count % self.sample_interval) == 0 and len(self.trajectory) < 20:
            dx = behavior[0] - self.last_location[0]
            dz = behavior[1] - self.last_location[1]

            direction_angle = math.atan2(dz, dx)

            # Normalize angle to [0, 1]
            if direction_angle < 0:
                direction_angle = 2 * math.pi + direction_angle
            direction_angle /= 2 * math.pi

            distance = min(math.sqrt(dx ** 2 + dz ** 2) / 20, 1)

            self.trajectory.append(direction_angle)
            self.trajectory.append(distance)

            self.last_location = behavior

    def end(self) -> List[float]:
        self.behavior.extend(self.trajectory)

        # Pad to required length
        while len(self.behavior) < self.size:
            self.behavior.extend(self.behavior[-2:])

        return self.behavior

    def random_sample(self):
        skewness = [4] * 8 + [1, 3] * 10

        return [random.random() ** skew for skew in skewness]


class CrawlerMorphBehavior(Behavior):
    """
    Length: 8
        Normalized amount of filler blocks per quadrant (dimension size: 4)
        Normalized amount of configurable joints per quadrant (dimension size: 4)
    """

    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, size)
        self.sample_interval = sample_interval
        self.step_count = 0
        self.behavior = self.dummy()

    def set(self, behavior: CPPNMorph):
        if isinstance(behavior, CPPNMorph):
            morph: CPPNMorph = behavior
            bc = list()

            w, h, d = morph.m_morph.shape
            x_mid, y_mid, z_mid = w // 2, h // 2, d // 2

            # Normalized amount of blocks per quadrant
            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(b_quadrant == 1) / b_quadrant.size for b_quadrant in m_b_quadrants])

            # Normalized amount of configurable joints per quadrant
            m_j_quadrants = [
                morph.m_joints[:x_mid, :y_mid, z_mid:],
                morph.m_joints[:x_mid, :y_mid, :z_mid],
                morph.m_joints[x_mid:, :y_mid, z_mid:],
                morph.m_joints[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(j_quadrant != 0) / 4 for j_quadrant in m_j_quadrants])

            self.behavior = bc
        else:
            raise ValueError(f"{CrawlerMorphBehavior} expects to be set with a CPPNMorph instance")

    def add(self, behavior: Tuple[float, float]):
        pass

    def end(self) -> List[float]:
        return self.behavior

    def random_sample(self):
        return [random.random() for _ in range(8)]


class DirectionTrajectory2DMorphBehavior(Behavior):
    """
    Length: 44
        Normalized amount of filler blocks per quadrant (dimension size: 8)
        Normalized amount of sensor blocks per quadrant (dimension size: 8)
        Normalized amount of configurable joints per quadrant (dimension size: 8)
        10 (direction, distance) samples (dimension size: 20 = 2 * 10)
            direction is the normalized angle on unit sphere
    """

    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, size)
        self.sample_interval = sample_interval
        self.step_count = 0
        self.behavior = self.dummy()
        self.last_location = (0, 0)
        self.trajectory = list()

    def set(self, behavior: CPPNMorph):
        if isinstance(behavior, CPPNMorph):
            morph: CPPNMorph = behavior
            bc = list()

            w, h, d = morph.m_morph.shape
            x_mid, y_mid, z_mid = w // 2, h // 2, d // 2

            # Normalized amount of blocks per quadrant
            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, y_mid:, :z_mid],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[:x_mid, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, y_mid:, :z_mid],
                morph.m_morph[x_mid:, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(b_quadrant == 1) / b_quadrant.size for b_quadrant in m_b_quadrants])

            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, y_mid:, :z_mid],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[:x_mid, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, y_mid:, :z_mid],
                morph.m_morph[x_mid:, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(b_quadrant == 2) / 16 for b_quadrant in m_b_quadrants])

            # Normalized amount of configurable joints per quadrant
            m_j_quadrants = [
                morph.m_joints[:x_mid, :y_mid, z_mid:],
                morph.m_joints[:x_mid, y_mid:, :z_mid],
                morph.m_joints[:x_mid, :y_mid, :z_mid],
                morph.m_joints[:x_mid, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, z_mid:],
                morph.m_joints[x_mid:, y_mid:, :z_mid],
                morph.m_joints[x_mid:, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, :z_mid]
            ]
            bc.extend([np.sum(j_quadrant != 0) / 16 for j_quadrant in m_j_quadrants])

            self.behavior = bc
        else:
            raise ValueError(f"{DirectionTrajectory2DMorphBehavior} expects to be set with a CPPNMorph")

    def add(self, behavior: Tuple[float, float]):
        self.step_count += 1
        if (self.step_count % self.sample_interval) == 0 and len(self.trajectory) < 20:
            dx = behavior[0] - self.last_location[0]
            dz = behavior[1] - self.last_location[1]

            direction_angle = math.atan2(dz, dx)

            # Normalize angle to [0, 1]
            if direction_angle < 0:
                direction_angle = 2 * math.pi + direction_angle
            direction_angle /= 2 * math.pi

            distance = min(math.sqrt(dx ** 2 + dz ** 2) / 10, 1)

            self.trajectory.append(direction_angle)
            self.trajectory.append(distance)

            self.last_location = behavior

    def end(self):
        self.behavior.extend(self.trajectory)

        # Pad to required length
        while len(self.behavior) < self.size:
            self.behavior.extend(self.behavior[-2:])

        return self.behavior

    def random_sample(self):
        skewness = [4] * 24 + [1, 3] * 10

        return [random.random() ** skew for skew in skewness]


class Trajectory2DMorphBehavior(Behavior):
    """ Length = 36
    Behavior characterisation that includes:
        0-20    : 10 trajectory samples (2D) averaged to x axis as in
                    "Novelty Search for Soft Robotic space exploration"; this makes the trajectories rotation invariant
        20-28   : Relative amount of configurable joints per quadrant
        28-36   : Relative amount of filler blocks per quadrant
    """

    def __init__(self, dummy_behavior, size, sample_interval=None):
        super().__init__(dummy_behavior, size)
        self.sample_interval = sample_interval
        self.behavior = self.dummy()
        self.trajectory = list()
        self.step_count = 0

    def set(self, behavior):
        if isinstance(behavior, CPPNMorph):
            # Morph was given -> count number of cj's and number of blocks in each quadrant of agent space
            morph: CPPNMorph = behavior
            x_mid, y_mid, z_mid = morph.m_morph.shape
            x_mid, y_mid, z_mid = x_mid // 2, y_mid // 2, z_mid // 2

            # Relative amount of configurable joints per quadrant
            m_j_quadrants = [
                morph.m_joints[:x_mid, :y_mid, z_mid:],
                morph.m_joints[:x_mid, y_mid:, :z_mid],
                morph.m_joints[:x_mid, :y_mid, :z_mid],
                morph.m_joints[:x_mid, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, z_mid:],
                morph.m_joints[x_mid:, y_mid:, :z_mid],
                morph.m_joints[x_mid:, y_mid:, z_mid:],
                morph.m_joints[x_mid:, :y_mid, :z_mid]
            ]
            self.behavior[20:28] = [np.sum(j_quadrant != 0) / 16 for j_quadrant in m_j_quadrants]

            # Relative amount of blocks per quadrant
            m_b_quadrants = [
                morph.m_morph[:x_mid, :y_mid, z_mid:],
                morph.m_morph[:x_mid, y_mid:, :z_mid],
                morph.m_morph[:x_mid, :y_mid, :z_mid],
                morph.m_morph[:x_mid, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, z_mid:],
                morph.m_morph[x_mid:, y_mid:, :z_mid],
                morph.m_morph[x_mid:, y_mid:, z_mid:],
                morph.m_morph[x_mid:, :y_mid, :z_mid]
            ]
            self.behavior[28:] = [np.sum(b_quadrant != 0) / b_quadrant.size for b_quadrant in m_b_quadrants]

    def add(self, trajectory_sample):
        self.step_count += 1
        if (self.step_count % self.sample_interval) == 0 and len(self.trajectory) < 10:
            self.trajectory.append(np.array(trajectory_sample))

    def end(self):
        if len(self.trajectory) > 0:
            # Rotate so that the average trajectory sample lies on the x axis (z == 0)
            #   Calculate rotation matrix
            average_point = np.mean(self.trajectory, axis=0)
            average_point_on_unit_circle = average_point / np.linalg.norm(average_point)
            theta = np.arccos(average_point_on_unit_circle[0])  # angle to rotate
            if average_point_on_unit_circle[1] > 0:
                theta = -theta

            c = np.cos(theta)
            s = np.sin(theta)
            rot_matrix = np.array([[c, -s], [s, c]])

            rotated_trajectory = [np.dot(rot_matrix, point) for point in self.trajectory]

            # Flatten
            self.trajectory = np.concatenate(rotated_trajectory).flatten()

        # Pad to required length
        self.trajectory = np.pad(self.trajectory, (0, 20 - len(self.trajectory)))
        self.behavior[:20] = self.trajectory

        return self.behavior


class ActionDistributionBehavior(Behavior):
    def __init__(self, dummy_behavior, size):
        super().__init__(dummy_behavior, size)
        self.behavior = self.dummy()

    def step(self, actions, obs=None):
        self.behavior[actions[0]] += 1
        self.behavior[actions[1] + 3] += 1
        self.behavior[actions[2] + 6] += 1
        self.behavior[actions[3] + 9] += 1

    def end(self):
        norm = sum(self.behavior)
        return [x / norm for x in self.behavior]


class TimeDividedActionDistributionBehavior(Behavior):
    def __init__(self, dummy_behavior, size, sample_interval=100):
        super().__init__(dummy_behavior, size)
        self.sample_interval = sample_interval
        self.time = None
        self.adb = None
        self.behavior = None
        self.samples = None
        self.reset()

    def step(self, actions, obs=None):
        if self.samples < (self.size / 13):
            self.time += 1
            self.adb.step(actions)
            if self.time % self.sample_interval == 0:
                self.behavior.extend(self.adb.end())
                self.adb = ActionDistributionBehavior(0, 13)
                self.samples += 1

    def end(self):
        # Pad if needed
        self.behavior += [self.dummy_behavior] * (self.size - len(self.behavior))
        return self.behavior

    def reset(self):
        self.behavior = list()
        self.adb = ActionDistributionBehavior(0, 13)
        self.samples = 0
        self.time = 0


class TimedMoveEatDrinkBehavior(Behavior):
    def __init__(self, dummy_behavior, size):
        super().__init__(dummy_behavior, size)
        self.behavior = None  # Array containing mean amount of time since (forward_motion, side_motion, rotation, jumping, eating, drinking)
        self.last_action_at_time = None  # Array containing time of last execution of action
        self.actions_done = None  # Array containing the number of times an action was done per item
        self.time = 0
        self.reset()

    def step_(self, i):
        self.actions_done[i] += 1
        self.behavior[i] = self.time - self.last_action_at_time[i]
        self.last_action_at_time[i] = self.time

    def step(self, actions, obs=None):
        self.time += 1

        # Forward, side motion and rotation
        for i in range(3):
            if actions[i] != 0:
                self.step_(i)

        # Eating, drinking and jumping
        act = actions[3]
        if act != 0:
            i = 2 + act
            self.step_(i)

    def end(self):
        if self.time > 0:
            return [x / n if n > 0 else 0 for x, n in zip(self.behavior, self.actions_done)]
        else:
            return self.behavior

    def reset(self):
        self.behavior = self.dummy()  # Array containing mean amount of time since (forward_motion, side_motion, rotation, jumping, eating, drinking)
        self.last_action_at_time = self.dummy()  # Array containing time of last execution of action
        self.actions_done = self.dummy()  # Array containing the number of times an action was done per item
        self.time = 0


class EnergyTimeSeriesBehavior(Behavior):
    def __init__(self, dummy_behavior, size, sample_interval=100):
        super().__init__(dummy_behavior, size)
        assert size % 3 == 0, f"{self.__class__} expects a size that is a multiple of 3!"
        self.behavior = None
        self.sample_interval = sample_interval
        self.time = 0
        self.samples = 0
        self.reset()

    def step(self, actions, obs=None):
        assert obs is not None, f"{self.__class__} requires observations!"
        self.time += 1
        if self.samples < (self.size / 3) and self.time % self.sample_interval == 0:
            energy_state = obs[-8:][:3]
            self.behavior[self.samples * 3: self.samples * 3 + 3] = energy_state
            self.samples += 1

    def end(self):
        return self.behavior

    def reset(self):
        self.behavior = self.dummy()
        self.time = 0
        self.samples = 0


class TimedDistanceEnergyAccumulationBehavior(Behavior):
    def __init__(self, dummy_behavior, size, sample_interval=100):
        super().__init__(dummy_behavior, size)
        self.sample_interval = sample_interval
        self.behavior = None
        self.time = None
        self.samples = None

        self.original_position = None
        self.food_accumulated = None
        self.drink_accumulated = None
        self.distance_changed = None
        self.last_food = None
        self.last_drink = None
        self.last_position = None
        self.reset()

    def step(self, action, obs=None):
        assert obs is not None, f"{self.__class__} requires observations!"
        health, food, drink = obs[-8:][:3]
        position = obs[-2:]

        if self.time == 0:
            self.original_position = position
        else:
            if food > self.last_food:
                self.food_accumulated += food - self.last_food
            if drink > self.last_drink:
                self.drink_accumulated += drink - self.last_drink

            self.distance_changed = self.distance_changed + abs(
                position - self.last_position)  # can't use += because it throws a pickle error when using MP

        self.last_drink = drink
        self.last_food = food
        self.last_position = position

        self.time += 1

        if self.samples < (self.size / 6) and self.time % self.sample_interval == 0:
            original_position_change = position - self.original_position
            self.distance_changed = self.distance_changed / self.sample_interval
            self.behavior.extend([self.distance_changed[0], self.distance_changed[1], original_position_change[0],
                                  original_position_change[1], self.food_accumulated, self.drink_accumulated])
            self.samples += 1
            self.food_accumulated = 0
            self.drink_accumulated = 0
            self.distance_changed = np.array((0, 0))
            self.original_position = position

    def end(self):
        # pad if necessary
        self.behavior += [0] * (self.size - len(self.behavior))
        return self.behavior

    def reset(self):
        self.behavior = list()
        self.time = 0
        self.samples = 0

        self.original_position = np.array((0, 0))
        self.food_accumulated = 0
        self.drink_accumulated = 0
        self.distance_changed = np.array((0, 0))
        self.last_food = 1
        self.last_drink = 1
        self.last_position = np.array((0, 0))
