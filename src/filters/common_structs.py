#!/bin/python3

import math
import numpy as np
import time


# ! Class for representing points and vectors in N-dimensions
class VectorN:
    def __init__(self, points=np.array([])):
        self.points = points

    def __add__(self, other):
        if isinstance(other, type(self)):
            return VectorN(np.add(self.points, other.points))
        elif isinstance(other, type(self.points)):
            return VectorN(np.add(self.points, other))
        else:
            return VectorN(self.points + other)

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return VectorN(np.subtract(self.points, other.points))
        else:
            return VectorN(self.points - other)

    def __mul__(self, other):
        if isinstance(other, type(self)):
            return VectorN(np.multiply(self.points, other.points))
        else:
            return VectorN(self.points * other)

    def __div__(self, other):
        if isinstance(other, type(self)):
            return VectorN(np.divide(self.points, other.points))
        else:
            return VectorN(self.points / other)


class Clock:
    def __init__(self, scale=0.0001):
        self.scale = scale
        # self.time = time.clock()

    def get_time(self):
        return time.clock() * self.scale


# class Vector3:
#     def __init__(self, x=0.0, y=0.0, z=0.0):
#         self.x = x
#         self.y = y
#         self.z = z

#     def __add__(self, other):
#         return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

#     def __sub__(self, other):
#         return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

#     # def __mul__(self, other):
#     #     return


