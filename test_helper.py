import math
import pandas as pd
import numpy as np

from ParallelOptimizer import ParallelOptimizer


def linear(x: list) -> float:
    return x[0] * 2 + 3.0


def foo2D(x):
    return sum(x)


def const2D(x):
    return 0.0


def decay(x):
    x0 = x[0]
    x1 = x[1]
    if abs(x0) < 1.0:
        x0 = 1.0
    if abs(x1) < 1.0:
        x1 = 1.0
    return 1.0 / (x0 * x0 + x1 * x1)


def sombrero(x):
    denom = math.sqrt(x[0] ** 2 + x[1] ** 2)
    if denom == 0:
        return 1.0
    return math.sin(math.sqrt(x[0] ** 2 + x[1] ** 2)) / denom



def CreateOptimizer_Instance(table: str) -> ParallelOptimizer:
    data = pd.read_csv(table)
    opt = ParallelOptimizer(linear)
    opt.known_values = data
    opt.major_axis = 0

    opt.mins = [-10, -10]
    opt.maxs = [10, 10]
    opt.names = ["x1", "x2"]
    return opt


def CreateOptimizer_Sum() -> ParallelOptimizer:
    opt = ParallelOptimizer(linear)
    values = np.arange(-10, 11)
    opt.known_values["x1"] = values
    opt.known_values["x2"] = values
    opt.known_values["Objective"] = values

    names = ["x1", "x2"]
    mins = [values[0], values[0]]
    maxs = [values[-1], values[-1]]

    opt.Init(names=names, mins=mins, maxs=maxs)

    opt.known_values["x1"] = values
    opt.known_values["x2"] = values
    opt.known_values["Objective"] = values + values
    opt.known_values.loc[:, 'blocked'] = False
    opt.known_values.loc[:, 'plato_block'] = False
    opt.known_values.loc[:, 'plato_index'] = -1
    opt.known_values.loc[:, 'plato_edge'] = False

    opt.major_axis = 0
    return opt