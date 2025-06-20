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

# TODO: добавить функцую, которая генерирует таблицы test_table*.csv (пишет в файл) в начале тестов setup/teardown ?
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


    # ========================================================================================
    # SelectIntervals()
    def test_SelectIntervals_NoPlato(self):
        opt = self.CreateOptimizer_Instance("test_table2.csv")
        opt.known_values.loc[:, "Objective"] = 0

        self.assertTrue(opt.SelectIntervals())
        self.assertTrue(opt.major_axis_intervals.shape[0] > 0)

        gt_interval_points_xL = [-9.5, -4.5, 0.5, 5.5]
        gt_interval_points_xR = [-5.5, -0.5, 4.5, 9.5]
        gt_cost = 0.25

        out_interval_points_xL = sorted(opt.major_axis_intervals["xL"].to_list())
        out_interval_points_xR = sorted(opt.major_axis_intervals["xR"].to_list())
        out_cost = opt.major_axis_intervals["cost"].to_list()

        for i in range(opt.major_axis_intervals.shape[0]):
            self.assertAlmostEqual(gt_interval_points_xL[i], out_interval_points_xL[i], 3)
            self.assertAlmostEqual(gt_interval_points_xR[i], out_interval_points_xR[i], 3)
            self.assertEqual(gt_cost, out_cost[i])

        self.assertAlmostEqual(1.0, opt.major_axis_intervals["cost"].sum(), 5)


    def test_SelectIntervals_ExcludesPlatoRegions(self):
        opt = self.CreateOptimizer_Instance("test_table1.csv")
        opt.known_values.loc[:, "Objective"] = 0

        self.assertTrue(opt.SelectIntervals())
        self.assertTrue(opt.major_axis_intervals.shape[0] == 2)

        gt_xL = [-9.5, -4.5]
        gt_xR = [-5.5, -0.5]
        gt_cost = [0.5, 0.5]

        out_interval_points_xL = sorted(opt.major_axis_intervals["xL"].to_list())
        out_interval_points_xR = sorted(opt.major_axis_intervals["xR"].to_list())
        out_cost = opt.major_axis_intervals["cost"].to_list()

        for i in range(opt.major_axis_intervals.shape[0]):
            self.assertAlmostEqual(gt_xL[i], out_interval_points_xL[i], 3)
            self.assertAlmostEqual(gt_xR[i], out_interval_points_xR[i], 3)
            self.assertEqual(gt_cost[i], out_cost[i])

        self.assertAlmostEqual(1.0, opt.major_axis_intervals["cost"].sum(), 5)


    def test_SelectIntervals_ExcludesPlatoRegions_BackwardPass_NonConstObjective(self):
        opt = self.CreateOptimizer_Instance("test_table3.csv")
        self.assertTrue(opt.SelectIntervals(forward=False))
        self.assertTrue(opt.major_axis_intervals.shape[0] == 3)

        gt_xL = [-5.9, -4.9, -3.9]
        gt_xR = [-5.1, -4.1, -3.1]
        gt_cost = [0.2888888888888889, 0.3333333333333333, 0.37777777777777777]

        out_interval_points_xL = sorted(opt.major_axis_intervals["xL"].to_list())
        out_interval_points_xR = sorted(opt.major_axis_intervals["xR"].to_list())
        out_cost = opt.major_axis_intervals["cost"].to_list()

        for i in range(opt.major_axis_intervals.shape[0]):
            self.assertAlmostEqual(gt_xL[i], out_interval_points_xL[i], 3)
            self.assertAlmostEqual(gt_xR[i], out_interval_points_xR[i], 3)
            self.assertEqual(gt_cost[i], out_cost[i])

        self.assertAlmostEqual(1.0, opt.major_axis_intervals["cost"].sum(), 5)



    # =========================================================================================
    # UnitMapping()
    def test_UnitMapping_EmptyIntervals(self):
        opt = self.CreateOptimizer_Instance("test_table3.csv")
        opt.UnitMapping()
        self.assertTrue(len(opt.u_coords) == 0)


    def test_UnitMapping_CorrectValues(self):
        opt = self.CreateOptimizer_Instance("test_table2.csv")
        self.assertTrue(opt.SelectIntervals())
        self.assertTrue(opt.major_axis_intervals.shape[0] > 0)

        opt.UnitMapping()
        gt_u_coords = [
            (0.0, 0.332),
            (0.342, 0.658),
            (0.668, 0.848),
            (0.858, 1.03)]

        self.assertTrue(len(gt_u_coords), len(opt.u_coords))

        for i in range(len(gt_u_coords)):
            self.assertAlmostEqual(gt_u_coords[i][0], opt.u_coords[i][0], 2)
            self.assertAlmostEqual(gt_u_coords[i][1], opt.u_coords[i][1], 2)



    # =========================================================================================
    # UnmapValue()
    def test_UnmapValue_CorrectValues(self):
        opt = self.CreateOptimizer_Instance("test_table2.csv")

        self.assertTrue(opt.SelectIntervals())
        self.assertTrue(opt.major_axis_intervals.shape[0] > 0)

        opt.UnitMapping()
        self.assertTrue(len(opt.u_coords) > 0)

        gt_values = [
            (0.25, -6.4897),
            (0.335, 5.54),
            (0.56, 8.251),
            (0.848, -0.5071),
            (1.0, 3.801)
        ]

        for gv in gt_values:
            unmapped_x = opt.UnmapValue(gv[0])
            self.assertAlmostEqual(gv[1], unmapped_x, 3)
