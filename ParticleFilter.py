from copy import copy
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import bisect
from __future__ import absolute_import
from __future__ import division
from random import uniform, gauss
import math

V_VAR = 0.1
THETA_VAR = 0.01
RANGE_SENSOR_SIGMA = 0.005
BEARING_SENSOR_SIGMA = 0.2


def Closest(ID1, beacon1, ID2, beacon2):
    "calculates which beacon is closest"
    r1 = np.sqrt(beacon1[0] ** 2 + beacon1[1] ** 2)
    r2 = np.sqrt(beacon2[0] ** 2 + beacon2[1] ** 2)
    if (r1 > r2):
        return ID2, beacon2
    else:
        return ID1, beacon1


def angle_wrap(a):
    """ Wrap angle into range -180 <= a < 180 """
    if a < -np.pi:
        a += 2 * np.pi
    elif a > np.pi:
        a -= 2 * np.pi
    return a


def gauss_pdf(z, sigma):
    return np.exp(-0.5 * (z / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


class Robot(object):
    def __init__(self, z):
        """Initiliazing robot with a pose (x, y, theta)"""
        """Pose is input as a 3 element array"""

        self.x = z[0]
        self.y = z[1]
        self.theta = z[2]

    def state(self):
        """Return state of robot in an array"""
        return [self.x, self.y, self.theta]

    def velocity_motion_model(self, dt, v, omega):
        """Robot's new state through velocity motion model"""

        v += np.random.normal(0, V_VAR, 1)

        if (abs(omega) < 0.01):
            dx = v * dt * np.cos(self.theta)
            dy = v * dt * np.sin(self.theta)
            dtheta = np.random.normal(0, THETA_VAR, 1)
            # dtheta = 0

        else:
            omega += np.random.normal(0, THETA_VAR, 1)
            dx = -(v / omega) * np.sin(self.theta) + (v / omega) * np.sin(self.theta + omega * dt)
            dy = (v / omega) * np.cos(self.theta) - (v / omega) * np.cos(self.theta + omega * dt)
            dtheta = omega * dt

        self.x += dx
        self.y += dy
        self.theta += dtheta

    def odometry_motion_model(self, odom_prev, odom_current):
        diff = odom_current - odom_prev

        dx = np.random.normal(diff[0], RANGE_SENSOR_SIGMA, 1)
        dy = np.random.normal(diff[1], RANGE_SENSOR_SIGMA, 1)
        dtheta = np.random.normal(diff[2], BEARING_SENSOR_SIGMA, 1)

        self.x += dx
        self.y += dy
        self.theta += dtheta

    def pose(self, beacon_relative):

        self.range = np.sqrt(beacon_relative[0] ** 2 + beacon_relative[1] ** 2) + RANGE_SENSOR_SIGMA
        self.bear = self.theta - np.arctan2(beacon_relative[1], beacon_relative[0]) + BEARING_SENSOR_SIGMA

    def weight(self, robot_r, robot_b, beacon_pos):

        dx = abs(beacon_pos[0] - self.x)
        dy = abs(beacon_pos[1] - self.y)
        self.range = np.sqrt(dx ** 2 + dy ** 2)
        self.bear = self.theta - np.arctan2(dy, dx)

        range_w = gauss_pdf(robot_r - self.range, RANGE_SENSOR_SIGMA)
        bearing_w = gauss_pdf(angle_wrap(robot_b - self.bear), BEARING_SENSOR_SIGMA)
        return range_w * bearing_w


class Particles(object):
    def __init__(self, particles):
        """Initiliazing particles"""

        self.particles = particles
        self.weights = np.ones(len(self.particles))

    def predict(self, dt, v, w, odom_prev, odom, model):
        """Perform prediction step using state transition and control vector u"""
        if (model == 'velocity'):
            for p in self.particles:
                p.velocity_motion_model(dt, v, w)

        if (model == 'odom'):
            for p in self.particles:
                p.odometry_motion_model(odom_prev, odom)

    def update(self, robot_r, robot_b, beacon_pos):
        """Perform update step using weights and measurement vector z"""

        for m, p in enumerate(self.particles):
            self.weights[m] *= p.weight(robot_r, robot_b, beacon_pos)
            # self.weights = self.weights/self.weights.sum()

    def resample(self):
        """Resample particles in proportion to their weights.
        Particles and weights should be arrays, and will be updated in place."""

        cum_weights = np.cumsum(self.weights)
        cum_weights /= cum_weights[-1]

        new_particles = []
        for _ in self.particles:
            # Copy a particle into the list of new particles, choosing based
            # on weight
            m = bisect.bisect_left(cum_weights, np.random.uniform(0, 1))
            p = self.particles[m]
            new_particles.append(p)

        # Replace old particles with new particles
        for m, p in enumerate(new_particles):
            self.particles[m] = p

        # Reset weights
        self.weights[:] = 1

    def mean_and_confidence(self, robot):
        """ Return mean particle position and confidence """
        x = np.array([p.x for p in self.particles])
        y = np.array([p.y for p in self.particles])
        a = np.array([p.theta for p in self.particles])

        m_x = (x * self.weights).sum() / self.weights.sum()
        m_y = (y * self.weights).sum() / self.weights.sum()
        # Calculate weighted mean heading in Cartesian coords to avoid wrap
        m_h = np.arctan2((np.sin(a) * self.weights).sum(), (np.cos(a) * self.weights).sum())

        robot.x, robot.y, robot.theta = m_x, m_y, m_h

    def is_degenerate(self):
        w = self.weights
        w = w / sum(w)
        return 1 / sum(w ** 2) < 0.5 * len(w)


###############################################################################
# Load data

# data is a (many x 13) matrix. Its columns are:
# time_ns, velocity_command, rotation_command, map_x, map_y, map_theta, odom_x, odom_y, odom_theta,
# beacon_id, beacon_x, beacon_y, beacon_theta
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

t = data[:, 0]  # Time in ns
command = data[:, 1:3]  # Velocity command in m/s, rotation command in rad/s
map_pos = data[:, 3:6]  # Position in map frame, from SLAM
odom_pos = data[:, 6:9]  # Position in odometry frame, from wheel encoders and gyro
beacon_id = data[:, 9]  # Beacon id and position in camera frame
beacon_pos = data[:, 10:]
# map_data is a (many x 13) matrix. Its columns are:
# beacon_id, x, y, theta, (9 columns of covariance)
map_data = np.genfromtxt('beacon_map.csv', delimiter=',', skip_header=1)
# Mapping from beacon id to beacon position
beacons = {id_: (x, y, theta) for (id_, x, y, theta) in map_data[:, :4]}


###############################################################################

PARTICLE_COUNT = 4000
# row = x,y,0, column = (m)th particle
particle_pose = np.zeros((PARTICLE_COUNT, 3))
x_max = 2.0;
x_min = -14.0;
y_max = 6.0;
y_min = -18.0

robot = Robot(np.array([x_max, y_max, np.pi]))  # velocity model
robot2 = Robot(np.array([x_max, y_max, np.pi]))  # odom model

particle_pose[:, 0] = np.random.uniform(x_min, x_max, PARTICLE_COUNT)
particle_pose[:, 1] = np.random.uniform(y_min, y_max, PARTICLE_COUNT)
particle_pose[:, 2] = np.random.uniform(-np.pi, np.pi, PARTICLE_COUNT)

particles = []
for i in range(0, PARTICLE_COUNT):
    new_particle = Robot(particle_pose[i])

    particles.append(new_particle)

PF = Particles(particles)
PF2 = Particles(particles)

PF.mean_and_confidence(robot)  # random pose and heading for initialising robot
PF2.mean_and_confidence(robot2)  # same starting position for robot

t = [t[0] * 0.99] + t

# for i in range(20):
for i in range(len(t) - 2):
    dt = t[i + 1] - t[i]
    dt2 = t[i + 2] - t[i + 1]
    DT = dt * 1e-9  # change scale

    if (dt == 0):  # if is used for 2nd beacon, skip
        continue
    ID, beacon = beacon_id[i], beacon_pos[i]
    if (dt2 == 0):
        ID, beacon_relative = Closest(beacon_id[i], beacon_pos[i], beacon_id[i + 1], beacon_pos[i + 1])

    # move particles according to motion models
    PF.predict(DT, command[i][0], command[i][1], odom_pos[i], odom_pos[i + 1], 'velocity')
    PF2.predict(DT, command[i][0], command[i][1], odom_pos[i], odom_pos[i + 1], 'odom')

    if not (math.isnan(ID)):  # if there is a beacon nearby, update weights
        beacon_world = beacons[ID]

        robot.pose(beacon_relative)  # calculates range and bearing of robot
        robot2.pose(beacon_relative)

        PF.update(robot.range, robot.bear, beacon_world)  #
        PF2.update(robot2.range, robot2.bear, beacon_world)

        if (PF.is_degenerate()):
            PF.resample()

        if (PF2.is_degenerate()):
            PF2.resample()

    PF.mean_and_confidence(robot)
    PF2.mean_and_confidence(robot2)
    print(robot.state(), robot2.state(), '\n')
