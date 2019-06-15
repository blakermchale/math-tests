#!/bin/python3

# Using this tutorial as example:
# https://github.com/Garima13a/Landmark-Detection-Tracking-SLAM-/blob/master/1.%20Robot%20Moving%20and%20Sensing.ipynb

import math
import numpy as np
import matplotlib.pyplot as plt
import random
import time

import seaborn as sns
import matplotlib.pyplot as plt

import threading


import multiprocessing

from graph_plot import VisualSim, GraphViewer
from common_structs import VectorN, Clock


# ! Class for representing simulation world
class World:
    def __init__(self, origin=VectorN(np.array([0.0, 0.0])), world_size=np.array(
            [10.0, 10.0]), wind=False, landmarks=np.array([])):
        self.dimensions = world_size.size
        self.world_size = world_size
        self.area = self.set_area(world_size)
        self.landmarks = landmarks
        self.origin = origin
        self.actors = {}

        # TODO: Add way to represent infinite worlds, use math.inf and
        # float("inf") to check
        self.halfworld_size = VectorN((self.world_size / 2))
        self.upbounds = self.origin + self.halfworld_size
        self.lowbounds = self.origin - self.halfworld_size
        print(self.upbounds.points)
        print(self.lowbounds.points)

        self.count = 0

    # ! Creates random landmarks in bounds of simulation
    def create_landmarks(self):
        for i in range(5):
            values = np.array([])
            for length in world_size:
                np.append(values, random.random() * length)
            np.append(self.landmarks, VectorN(values) + self.origin)

    # ! Calculates area of simulation world based dimensions
    def set_area(self, world_size):
        area = 1
        for length in np.nditer(world_size):
            area *= length
        return area

    def add_actor(self, actor):
        self.actors[actor.name] = actor

    # TODO: implement map and checking for collision based on path
    def checkCollision(self, currpos, despos):
        if np.all(
            np.greater(
                despos.points,
                self.lowbounds.points)) and np.all(
            np.less(
                despos.points,
                self.upbounds.points)):
            return False
        else:
            # print("World: {2}, Greater: {0}, Lesser: {1}".format(np.greater(
            #     despos.points,
            #     self.lowbounds.points), np.less(
            #     despos.points,
            #     self.upbounds.points), self.upbounds.points))
            return True


class Robot:
    def __init__(self,
                 world,
                 clock,
                 name,
                 position=VectorN(np.array([0.0,
                                            0.0])),
                 velocity=None,
                 acceleration=None,
                 measurement_range=30.0,
                 gps_noise=0.2,
                 imu_noise=0.2,
                 motion_noise=0.2):
        self.world = world
        if world.dimensions != position.points.size:
            print("Robot position does not match world dimensions! Deleting robot!")
            self.position = VectorN(position.points[:world.dimensions])
            # del self
            # return
        else:
            self.position = position
            print(self.position.points)
        self.clock = clock
        self.measurement_range = measurement_range
        # self.measurement_noise = measurement_noise
        self.gps_noise = gps_noise
        self.imu_noise = imu_noise
        self.motion_noise = motion_noise
        if velocity is None:
            self.velocity = VectorN(
                np.zeros(
                    shape=self.world.dimensions,
                    dtype=float))
        else:
            self.velocity = velocity
        if acceleration is None:
            self.acceleration = VectorN(
                np.zeros(
                    shape=self.world.dimensions,
                    dtype=float))
        else:
            self.acceleration = acceleration

        self.name = name

        self.world.add_actor(self)

        #! Memory of past movement times
        self.lastMoveTime = self.clock.get_time()

        #! Generate minimums and maximums of degrees of motion
        self.maxAcc = 2.0  # 2 m/s
        self.minAcc = -2.0  # -2 m/s

    def randNoise(self, noiseLim):
        return np.random.randn(self.world.dimensions) * \
            noiseLim  # * 2 - noiseLim

    def moveByAcc(self, acc):
        dt = self.clock.get_time() - self.lastMoveTime
        self.lastMoveTime = self.clock.get_time()
        # print(dt)
        # Apply some noise to the acceleration value, so the models can mimic
        # having error
        velocity = self.velocity + \
            (acc + self.randNoise(self.motion_noise)) * dt
        # print(velocity.points)
        # des_position = self.position + (velocity + self.randNoise(self.motion_noise)) * dt
        des_position = self.position + velocity * dt

        if not self.world.checkCollision(self.position, des_position):
            self.position = des_position
            self.velocity = velocity
            self.acceleration = acc
            if self.world.count % 10000 == 0 or self.world.count == 1:
                print(self.world.count)
                print("Position: {0}, Velocity: {1}, Acceleration: {2}".format(
                    self.position.points, self.velocity.points, acc.points))
        else:
            self.velocity = VectorN(
                np.zeros(
                    shape=self.world.dimensions,
                    dtype=float))
            # if self.world.count % 10000 == 0 or self.world.count == 1:
            #     print(self.world.count)
                # print("Noise: {0}".format(np.random.rand(self.world.dimensions) * self.motion_noise))
            print("Collision! Position: {0}, Velocity: {1}, Acceleration: {2}".format(des_position.points, velocity.points, acc.points))
            # return
            # print("Collided!")

    def randMoveByAcc(self):
        # Generate random vector that is within minimum and maximum
        # acceleration
        acc = VectorN(np.random.rand(self.world.dimensions) *
                      (self.maxAcc - self.minAcc) + self.minAcc)
        # print(acc.points)
        self.moveByAcc(acc)

    def runGPS(self):
        # Return noisy measurement of position from sensor
        return self.position + self.randNoise(self.gps_noise)

    def runIMU(self):
        return self.acceleration + self.randNoise(self.imu_noise)


# ! Class representing entire simulation environment: includes all dynamic objects and world
class Simulation:
    def __init__(self, visual=True, world_size=np.array([10.0, 10.0]), view_size=None):
        # self.world = World(world_size=np.array([10.0]))
        self.world = World(world_size=world_size)
        self.clock = Clock(scale=3.0)
        # self.robot1 = Robot(self.world, self.clock, "robot1", position=VectorN(np.array([0.0])))
        robot1 = Robot(self.world, self.clock, "robot1", motion_noise=2.0, gps_noise=4.5)
        self.visual = visual
        if visual:
            self.visSim = VisualSim(self.world, view_size=view_size)

        self.lastUpdateTime = self.clock.get_time()

        self.kalman = KalmanFilter()

    def update(self):
        dt = self.clock.get_time() - self.lastUpdateTime
        self.lastUpdateTime = self.clock.get_time()
        robot = self.world.actors["robot1"]
        self.world.count += 1
        # self.robot1.randMoveByAcc()
        robot.moveByAcc(VectorN(np.zeros(
                    shape=self.world.dimensions,
                    dtype=float)))
        gpsLoc = robot.runGPS()

        # ! Kalman Filter updates
        self.kalman.pos = self.kalman.predict(self.kalman.pos[0], self.kalman.pos[1], robot.velocity.points[0], robot.motion_noise)    
        self.kalman.variance.append(self.kalman.pos[1])
        
        self.kalman.pos = self.kalman.update(self.kalman.pos[0], self.kalman.pos[1], gpsLoc.points, robot.gps_noise)
        self.kalman.positions.append(self.kalman.pos)
        if self.world.count >= 100000:
            viewGraph = GraphViewer()
            viewGraph.plotKalman(self.kalman.positions)
            print("Final Position: {0}".format(robot.position.points))
        if self.visual:
            self.visSim.update()

    def setVel(self, vel, actor_name):
        self.world.actors[actor_name].velocity = vel


# https://share.cocalc.com/share/7557a5ac1c870f1ec8f01271959b16b49df9d087/Kalman-and-Bayesian-Filters-in-Python/04-One-Dimensional-Kalman-Filters.ipynb?viewer=share
class KalmanFilter:
    def __init__(self):
        # # Initial Estimation Covariance Matrix
        # P = self.covariance2d(error_est_x, error_est_v)
        # A = np.array([[1, t],
        #               [0, 1]])

        # # Initial State Matrix
        # X = np.array([[z[0][0]],
        #               [v]])
        # n = len(z[0])
        self.positions, self.variance = [], []
        self.pos = (0, 400)

    # def prediction2d(self, x, v, t, a):
    #     A = np.array([[1, t],
    #                   [0, 1]])
    #     X = np.array([[x],
    #                   [v]])
    #     B = np.array([[0.5 * t ** 2],
    #                   [t]])
    #     X_prime = A.dot(X) + B.dot(a)
    #     return X_prime

    # def covariance2d(self, sigma1, sigma2):
    #     cov1_2 = sigma1 * sigma2
    #     cov2_1 = sigma2 * sigma1
    #     cov_matrix = np.array([[sigma1 ** 2, cov1_2],
    #                            [cov2_1, sigma2 ** 2]])
    #     return np.diag(np.diag(cov_matrix))

    def multiply(self, mu1, var1, mu2, var2):
        if var1 == 0.0:
            var1 = 1.e-80
        if var2 == 0:
            var2 = 1e-80

        mean = (var1 * mu2 + var2 * mu1) / (var1 + var2)
        variance = 1 / (1 / var1 + 1 / var2)
        return (mean, variance)

    # Same as adding to Gaussian distributions together
    def predict(self, pos, variance, motion, motion_variance):
        return (pos + motion, variance + motion_variance)

    def update(self, mean, variance, measurement, measurement_variance):
        return self.multiply(mean, variance, measurement, measurement_variance)


class ThreadManager:
    def __init__(self, visual=True, threadingBool=False):
        self.visual = visual  # True turns visual on through Sim class
        self.threadingBool = threadingBool  # True turns visual on through threading
        self.viewSize = None  #np.array([2.0, 2.0])
        self.worldSize = np.array([math.inf])
        if self.threadingBool:
            self.sim = Simulation(not self.threadingBool, world_size=self.worldSize)
            if self.visual:
                self.vissim = VisualSim(self.sim.world, view_size=self.viewSize)
            # self.simthread = multiprocessing.Process(target=self.SimThread)
            self.simthread = threading.Thread(target=self.SimThread)
            self.simthread.daemon = True
            self.simthread.start()
        else:
            self.sim = Simulation(self.visual, world_size=self.worldSize, view_size=self.viewSize)
            self.SimThread()
        # threading.Thread(target=self.VisSimThread).start()

    def SimThread(self):
        self.sim.setVel(VectorN(np.array([1.0])), "robot1")
        while True:
            try:
                self.sim.update()
            except KeyboardInterrupt:
                print("^C")
                break

    def VisSimThread(self):
        while True:
            try:
                self.vissim.update()
            except KeyboardInterrupt:
                print("^C")
                break


# class SimThread(threading.Thread):
#     sim = None
#     def run(self):
#         self.sim = Simulation()
#         while True:
#             try:
#                 self.sim.update()
#             except KeyboardInterrupt:
#                 print("^C")
#                 break
#     def get_sim(self):
#         return self.sim

# class VisSimThread(threading.Thread):
#     def __init__(self, world=None, name=None):
#         super(VisSimThread, self).__init__(name=name)
#         self.world = world
#     def run(self):
#         self.vissim = VisualSim(self.world)
#         while True:
#             try:
#                 self.vissim.update()
#             except KeyboardInterrupt:
#                 print("^C")
#                 break


# if __name__ == "__main__":
#     simthread = SimThread(name = "Thread-{}".format(0))  # ...Instantiate a thread and pass a unique ID to it
#     simthread.start()                                   # ...Start the thread, invoke the run method
#     vissimthread = VisSimThread(name = "Thread-{}".format(1), world=simthread.get_sim().world)  # ...Instantiate a thread and pass a unique ID to it
# vissimthread.start()                                   # ...Start the
# thread, invoke the run method
if __name__ == "__main__":
    threadManager = ThreadManager(visual=False)
    if threadManager.visual and threadManager.threadingBool:
        threadManager.VisSimThread()
    elif not threadManager.visual and threadManager.threadingBool:
        while True:
            try:
                continue
            except KeyboardInterrupt:
                print("^C")
                break
