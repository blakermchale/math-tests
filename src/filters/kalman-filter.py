#!/bin/python3

from math import *
import matplotlib.pyplot as plt
import numpy as np

def f(mu, sigma2, x):
    # Function to solve gaussian equation
    coefficient = 1.0 / sqrt(2.0 * pi * sigma2)
    exponential = exp(-(x - mu)**2/(2 * sigma2))
    return coefficient * exponential