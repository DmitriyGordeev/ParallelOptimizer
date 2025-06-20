import math
from unittest import TestCase
import numpy as np
import pandas as pd

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


# TODO: дать возможность указывать как запускать вектор переменных с именами в функцию через функтор
class TestMulDimOptimizer(TestCase):


    @staticmethod
    def CreateOptimizer_Instance(table: str) -> MulDimOptimizer:
        data = pd.read_csv(table)
        opt = MulDimOptimizer(linear)
        opt.known_values = data
        opt.major_axis = 0

        opt.mins = [-10, -10]
        opt.maxs = [10, 10]
        opt.names = ["x1", "x2"]
        return opt


    @staticmethod
    def CreateOptimizer_Sum() -> MulDimOptimizer:
        opt = MulDimOptimizer(linear)
        values = np.arange(-10, 11)
        opt.known_values["x1"] = values
        opt.known_values["x2"] = values
        opt.known_values["Y"] = values

        opt.mins = [values[0], values[0]]
        opt.maxs = [values[-1], values[-1]]
        opt.names = ["x1", "x2"]

        opt.CreateTable()
        opt.known_values["x1"] = values
        opt.known_values["x2"] = values
        opt.known_values["Y"] = values + values
        opt.known_values.loc[:, 'blocked'] = False
        opt.known_values.loc[:, 'plato_block'] = False
        opt.known_values.loc[:, 'plato_index'] = -1
        opt.known_values.loc[:, 'plato_edge'] = False

        opt.major_axis = 0
        return opt


    # ========================================================================================
    # CanSelectPoint()
    def test_CanSelectPoint_ReturnsTrue(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        result = opt.CanSelectPoint(l_index=0, r_index=1)
        self.assertTrue(result)


    def test_CanSelectPoint_PlatoEdge_ReturnsTrue(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        result = opt.CanSelectPoint(l_index=1, r_index=2)
        self.assertTrue(result)


    def test_CanSelectPoint_InsidePlato_ReturnsFalse(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        result = opt.CanSelectPoint(l_index=2, r_index=3)
        self.assertFalse(result)


    # ========================================================================================
    # CreateIntervalSet()
    def test_CreateIntervalSet_NoPlato(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        opt.known_values.loc[:, "plato_index"] = -1
        opt.known_values.loc[:, "plato_edge"] = False

        gt_intervals = {(0, 1), (1, 2), (2, 3), (3, 4)}
        out_intervals = opt.CreateIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    def test_CreateIntervalSet_LimitedNumber(self):
        opt = self.CreateOptimizer_Sum()
        opt.num_forward_intervals = 5

        gt_intervals = {(15, 16), (16, 17), (17, 18), (18, 19), (19, 20)}
        out_intervals = opt.CreateIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    def test_CreateIntervalSet_ExcludePlatoPoints(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        opt.num_forward_intervals = 10
        gt_intervals = {(0, 1), (1, 2)}
        out_intervals = opt.CreateIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    # ========================================================================================
    # CreateBackwardIntervalSet()
    def test_CreateBackwardIntervalSet_NoBackwardIntervalsAvailable(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        opt.known_values.loc[:, "plato_index"] = -1
        opt.known_values.loc[:, "plato_edge"] = False
        opt.num_forward_intervals = opt.known_values.shape[0] - 1
        out_intervals = opt.CreateBackwardIntervalSet()
        self.assertTrue(len(out_intervals) == 0)


    def test_CreateBackwardIntervalSet_ExcludePlato(self):
        opt = self.CreateOptimizer_Instance("test_table3.csv")
        opt.num_forward_intervals = 4

        gt_intervals = {(4, 5), (5, 6), (6, 7)}
        out_intervals = opt.CreateBackwardIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)
