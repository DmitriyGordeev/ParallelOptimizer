import math
import pandas as pd
import numpy as np

from MulDimOptimizer import MulDimOptimizer


def linear(x):
    return x * 2 + 3.0


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



def CreateOptimizer_Instance(table: str) -> MulDimOptimizer:
    data = pd.read_csv(table)
    opt = MulDimOptimizer(linear)
    opt.known_values = data
    opt.major_axis = 0

    opt.mins = [-10, -10]
    opt.maxs = [10, 10]
    opt.names = ["x1", "x2"]
    return opt


def CreateOptimizer_Sum() -> MulDimOptimizer:
    opt = MulDimOptimizer(linear)
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



def GenerateTestTable1():
    data = pd.DataFrame()
    data["x1"] = [-10, -5, 0, 5, 10]
    data["x2"] = [-10, -5, 0, 5, 10]
    data["Objective"] = [0] * 5
    data["blocked"] = [False] * 5
    data["plato_block"] = [False] * 5
    data["plato_index"] = [-1, -1, 0, 0, 0]
    data["plato_edge"] = [False, False, True, False, True]
    data.to_csv("test_table1.csv", index=False)


def GenerateTestTable2():
    data = pd.DataFrame()
    data["x1"] = [-10, -5, 0, 5, 10]
    data["x2"] = [-10, -5, 0, 5, 10]
    data["Objective"] = [19.0, 18.0, -20.4, 16.0, 17.0]
    data["blocked"] = [False] * 5
    data["plato_block"] = [False] * 5
    data["plato_index"] = [-1] * 5
    data["plato_edge"] = [False] * 5
    data.to_csv("test_table2.csv", index=False)


def GenerateTestTable3():
    data = pd.DataFrame()
    data["x1"] = np.arange(-10.0, 11.0, 1.0)
    data["x2"] = np.arange(-10.0, 11.0, 1.0)
    data["Objective"] = data["x1"] + data["x2"]
    data["blocked"] = [False] * 21
    data["plato_block"] = [False] * 21
    data["plato_index"] = [0] * 5 + [-1] * 16
    data["plato_edge"] = [True] + [False] * 3 + [True] + [False] * 16
    data.to_csv("test_table3.csv", index=False)


def GenerateTestTable4():
    data = pd.DataFrame()
    data["x1"] = [-10.0, -5.0]
    data["x2"] = [-10.0, -5.0]
    data["Objective"] = [19.0, 18.0]
    data["blocked"] = [False, False]
    data["plato_block"] = [False, False]
    data["plato_index"] = [-1, -1]
    data["plato_edge"] = [False, False]
    data.to_csv("test_table4.csv", index=False)


def GenerateTestTable5():
    data = pd.DataFrame()
    data["x1"] = [-10.0, -5.0, 0.0, 1.8, 2.0, 3.0, 3.0001, 10.0]
    data["x2"] = [-10.0, -5.0, 0.0, 1.8, 2.0, 3.0, 3.0001, 10.0]
    data["Objective"] = [19.0, 18.0, -20.4, -20.8, -20.9, -20.9, -20.9, 17.0]
    data["blocked"] = [False] * 8
    data["plato_block"] = [False] * 8
    data["plato_index"] = [-1] * 8
    data["plato_edge"] = [False] * 8
    data.to_csv("test_table5.csv", index=False)


def GenerateTestTable6():
    data = pd.DataFrame()
    data["x1"] = [-10.0, -5.0, 0.0, 1.8, 2.0, 2.001, 3.0, 3.0001, 10.0]
    data["x2"] = [-10.0, -5.0, 0.0, 1.8, 2.0, 2.001, 3.0, 3.0001, 10.0]
    data["Objective"] = [19.0, 18.0, -20.4, -20.8, -20.9, -20.9, -20.9, -20.9, 17.0]
    data["blocked"] = [False] * 9
    data["plato_block"] = [False] * 5 + [True, False, False, False]
    data["plato_index"] = [-1, 0, 0, 0, 1, 1, 1, 1, -1]
    data["plato_edge"] = [False, True, False, True, True, False, False, True, False]
    data.to_csv("test_table6.csv", index=False)


def GeneratePlatoRegion1():
    data = pd.DataFrame()
    data["x1"] = [-5.0, 0.0, 1.8]
    data["x2"] = [-5.0, 0.0, 1.8]
    data["Objective"] = [18.0, -20.4, -20.8]
    data["blocked"] = [False] * 3
    data["plato_block"] = [False] * 3
    data["plato_index"] = [0] * 3
    data["plato_edge"] = [True, False, True]
    data["original_index"] = [1, 2, 3]
    data.to_csv("plato_region_1.csv", index=False)


def GeneratePlatoRegion2():
    data = pd.DataFrame()
    data["x1"] = [2.0, 3.0, 3.0001]
    data["x2"] = [2.0, 3.0, 3.0001]
    data["Objective"] = [-20.9, -20.9, -20.9]
    data["blocked"] = [False] * 3
    data["plato_block"] = [False] * 3
    data["plato_index"] = [1] * 3
    data["plato_edge"] = [True, False, True]
    data["original_index"] = [4, 6, 7]
    data.to_csv("plato_region_2.csv", index=False)