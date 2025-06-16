from unittest import TestCase
import numpy as np
import pandas as pd

from MulDimOptimizer import MulDimOptimizer


def linear(x):
    return x * 2 + 3.0


def foo2D(x):
    return sum(x)


# TODO: дать возможность указывать как запускать вектор переменных с именами в функцию через функтор
class TestMulDimOptimizer(TestCase):
    def test_run_objective(self):
        opt1 = MulDimOptimizer(linear)
        out1 = opt1.RunObjective(5)

        opt2 = MulDimOptimizer(foo2D)
        out2 = opt2.RunObjective([9, 10])
        pass


    def test_CreateTable(self):
        opt = MulDimOptimizer(linear)
        opt.RunCycle(names=["x1", "x2"], mins=[0, 100], maxs=[0, 100], max_epochs=5)
        pass


    def test_np_mat(self):
        major_values = [0, 1, 2]
        mins = [-100, -100]
        out_matrix = [[0] * len(major_values)] * len(mins)
        pass


    def test_create_syntetic_table(self):
        x1 = [0, 50, 80]
        x2 = [20, 10, 30]
        y  = [0, 0, 0]
        table = pd.DataFrame(columns=["X1", "X2", "Y"])
        table["X1"] = x1
        table["X2"] = x2
        table["Y"] = y
        table["blocked"] = [False] * len(x1)
        table["plato_block"] = [False] * len(x1)
        table["plato_index"] = [-1] * len(x1)
        table["plato_edge"] = [False] * len(x1)

        table.to_csv("debug_values_mul_dim_0.csv", index=False)
        pass


    def test_SelectIntervals_MajorAxis(self):
        data = pd.read_csv("debug_values_mul_dim_1.csv")
        opt = MulDimOptimizer(linear)
        opt.known_values = data

        opt.mins = [0, 0]
        opt.maxs = [100, 100]
        opt.names = ["X1", "X2"]

        opt.major_axis = 0
        opt.SelectIntervals()
        opt.UnitMapping()
        major_values = opt.CreateProbePoints()

        pass



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
