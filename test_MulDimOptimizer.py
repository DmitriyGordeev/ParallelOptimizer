import math
from multiprocessing.spawn import is_forking
from unittest import TestCase
import numpy as np
import pandas as pd

from MulDimOptimizer import MulDimOptimizer
from PlatoModule_MulDim import PlatoModule_MulDim


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


# TODO: дать возможность указывать как запускать вектор переменных с именами в функцию через функтор
class TestMulDimOptimizer(TestCase):
    def test_CreateTable_FilledTable(self):
        opt = MulDimOptimizer(foo2D)
        opt.mins = [-10, -10]
        opt.maxs = [10, 10]
        opt.names = ["x", "y"]
        opt.CreateTable()
        # TODO: y - названии переменной


    def test_RunCycle(self):
        opt = MulDimOptimizer(sombrero)
        # TODO: тест - сдвинуть сомбреро и увеличить диапазон Min-max чтобы вначале он не сразу нашел максимум
        opt.SetupEps(block_eps=0.01, plato_x_eps=5.0, plato_y_eps=0.1, plato_block_eps=0.1)
        opt.RunCycle(names=["x1", "x2"], mins=[-10, -10], maxs=[10, 10], max_epochs=10)


    # def test_create_syntetic_table(self):
    #     x1 = [0, 50, 80]
    #     x2 = [20, 10, 30]
    #     y  = [0, 0, 0]
    #     table = pd.DataFrame(columns=["X1", "X2", "Y"])
    #     table["X1"] = x1
    #     table["X2"] = x2
    #     table["Y"] = y
    #     table["blocked"] = [False] * len(x1)
    #     table["plato_block"] = [False] * len(x1)
    #     table["plato_index"] = [-1] * len(x1)
    #     table["plato_edge"] = [False] * len(x1)
    #
    #     table.to_csv("debug_values_mul_dim_0.csv", index=False)
    #     pass


    def test_CreateIntervalSet_Table1(self):
        data = pd.read_csv("debug_table1.csv")
        opt = MulDimOptimizer(linear)
        opt.known_values = data

        opt.mins = [-10, -10]
        opt.maxs = [10, 10]
        opt.names = ["x1", "x2"]
        opt.major_axis = 0

        out_indexes = opt.CreateIntervalSet()
        gt_indexes = {(0, 1), (1, 2)}
        self.assertTrue(gt_indexes, out_indexes)


    def test_CreateIntervalSet_Table2(self):
        data = pd.read_csv("debug_table2.csv")
        opt = MulDimOptimizer(linear)
        opt.known_values = data

        opt.mins = [-10, -10]
        opt.maxs = [10, 10]
        opt.names = ["x1", "x2"]
        opt.major_axis = 0

        out_indexes = opt.CreateIntervalSet()
        gt_indexes = {(0, 1), (1, 2), (2, 3), (3, 4)}
        self.assertTrue(gt_indexes, out_indexes)


    def test_CreateIntervalSet_Table3_HitsIntervalNumLimit(self):
        data = pd.read_csv("debug_table3.csv")
        opt = MulDimOptimizer(linear)
        opt.known_values = data

        opt.mins = [-10, -10]
        opt.maxs = [10, 10]
        opt.names = ["x1", "x2"]
        opt.major_axis = 0
        opt.num_forward_intervals = 4

        out_indexes = opt.CreateIntervalSet()
        gt_indexes = {(1, 2), (2, 3), (5, 6), (6, 7)}
        self.assertTrue(gt_indexes, out_indexes)



    def test_SelectSinglePointOnMinorAxis(self):
        data = pd.read_csv("debug_values_mul_dim_0.csv")
        opt = MulDimOptimizer(linear)
        opt.known_values = data

        opt.mins = [0, 0]
        opt.maxs = [100, 100]
        opt.names = ["X1", "X2"]
        opt.major_axis = 0

        axis_values = []
        for i in range(opt.n_probes):
            axis_values.append(opt.SelectSinglePointOnMinorAxis(1))
        pass



    def test_GeneratePoints(self):
        data = pd.read_csv("debug_values_mul_dim_0.csv")
        opt = MulDimOptimizer(foo2D)
        opt.known_values = data

        opt.mins = [0, 0]
        opt.maxs = [100, 100]
        opt.names = ["X1", "X2"]
        opt.major_axis = 0

        x_matrix = opt.GeneratePoints()
        opt.RunValues(x_matrix)

        x_matrix = opt.GeneratePoints()
        opt.RunValues(x_matrix)
        pass


    def test_Warmup(self):
        opt = MulDimOptimizer(foo2D)

        opt.mins = [0, 20]
        opt.maxs = [100, 80]
        opt.names = ["X1", "X2"]

        opt.CreateTable()
        opt.Warmup()



    def test_find_plato(self):
        data = pd.read_csv("debug_values_mul_dim_2.csv")
        data.loc[:, "Y"] = 0
        opt = MulDimOptimizer(foo2D)
        opt.known_values = data


        opt.mins = [0, 0]
        opt.maxs = [100, 100]
        opt.names = ["X1", "X2"]
        opt.major_axis = 0

        plato_module = PlatoModule_MulDim(opt)
        plato_module.plato_x_eps = 10

        plato_module.FindPlatoRegions()
        plato_module.MarkPlatoRegions()
        plato_module.GroupTables()
        plato_module.UnitMapRegions()
        matrix = plato_module.UnmapValues()
        pass
