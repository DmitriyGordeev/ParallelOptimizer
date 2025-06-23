
from unittest import TestCase
from test_helper import *


# TODO: дать возможность указывать как запускать вектор переменных с именами в функцию через функтор

# TODO: добавить функцую, которая генерирует таблицы test_table*.csv (пишет в файл) в начале тестов setup/teardown ?
class TestMulDimOptimizer(TestCase):

    # ========================================================================================
    # CanSelectPoint()
    def test_CanSelectPoint_ReturnsTrue(self):
        opt = CreateOptimizer_Instance("test_table1.csv")
        result = opt.CanSelectPoint(l_index=0, r_index=1)
        self.assertTrue(result)


    def test_CanSelectPoint_PlatoEdge_ReturnsTrue(self):
        opt = CreateOptimizer_Instance("test_table1.csv")
        result = opt.CanSelectPoint(l_index=1, r_index=2)
        self.assertTrue(result)


    def test_CanSelectPoint_InsidePlato_ReturnsFalse(self):
        opt = CreateOptimizer_Instance("test_table1.csv")
        result = opt.CanSelectPoint(l_index=2, r_index=3)
        self.assertFalse(result)


    # ========================================================================================
    # CreateIntervalSet()
    def test_CreateIntervalSet_NoPlato(self):
        opt = CreateOptimizer_Instance("test_table1.csv")
        opt.known_values.loc[:, "plato_index"] = -1
        opt.known_values.loc[:, "plato_edge"] = False

        gt_intervals = {(0, 1), (1, 2), (2, 3), (3, 4)}
        out_intervals = opt.CreateIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    def test_CreateIntervalSet_LimitedNumber(self):
        opt = CreateOptimizer_Sum()
        opt.num_forward_intervals = 5

        gt_intervals = {(15, 16), (16, 17), (17, 18), (18, 19), (19, 20)}
        out_intervals = opt.CreateIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    def test_CreateIntervalSet_ExcludePlatoPoints(self):
        opt = CreateOptimizer_Instance("test_table1.csv")
        opt.num_forward_intervals = 10
        gt_intervals = {(0, 1), (1, 2)}
        out_intervals = opt.CreateIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    # ========================================================================================
    # CreateBackwardIntervalSet()
    def test_CreateBackwardIntervalSet_NoBackwardIntervalsAvailable(self):
        opt = CreateOptimizer_Instance("test_table1.csv")
        opt.known_values.loc[:, "plato_index"] = -1
        opt.known_values.loc[:, "plato_edge"] = False
        opt.num_forward_intervals = opt.known_values.shape[0] - 1
        out_intervals = opt.CreateBackwardIntervalSet()
        self.assertTrue(len(out_intervals) == 0)


    def test_CreateBackwardIntervalSet_ExcludePlato(self):
        opt = CreateOptimizer_Instance("test_table3.csv")
        opt.num_forward_intervals = 4

        gt_intervals = {(4, 5), (5, 6), (6, 7)}
        out_intervals = opt.CreateBackwardIntervalSet()
        self.assertEqual(gt_intervals, out_intervals)


    # ========================================================================================
    # SelectIntervals()
    def test_SelectIntervals_NoPlato(self):
        opt = CreateOptimizer_Instance("test_table2.csv")
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
        opt = CreateOptimizer_Instance("test_table1.csv")
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
        opt = CreateOptimizer_Instance("test_table3.csv")
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
        opt = CreateOptimizer_Instance("test_table3.csv")
        opt.UnitMapping()
        self.assertTrue(len(opt.u_coords) == 0)


    def test_UnitMapping_CorrectValues(self):
        opt = CreateOptimizer_Instance("test_table2.csv")
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
        opt = CreateOptimizer_Instance("test_table2.csv")

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



    # =========================================================================================
    # SelectSinglePointOnMinorAxis
    def test_SelectSinglePointOnMinorAxis_ValueRange(self):
        opt = CreateOptimizer_Instance("test_table4.csv")
        opt.major_axis = 1

        axis_min = opt.known_values[opt.known_values.columns[0]].min()
        axis_max = opt.known_values[opt.known_values.columns[0]].max()

        for i in range(10):
            x = opt.SelectSinglePointOnMinorAxis(0)
            self.assertTrue(axis_min <= x <= axis_max)



    # =========================================================================================
    # GeneratePoints
    def test_GeneratePoints_ForwardPass_ValuesRangeCorrect(self):
        opt = CreateOptimizer_Instance("test_table4.csv")
        opt.major_axis = 0

        x1_min = opt.known_values[opt.known_values.columns[0]].min()
        x1_max = opt.known_values[opt.known_values.columns[0]].max()

        x2_min = opt.known_values[opt.known_values.columns[1]].min()
        x2_max = opt.known_values[opt.known_values.columns[1]].max()

        x_matrix = opt.GeneratePoints(is_forward=True)
        for i in range(x_matrix.shape[1]):
            self.assertTrue(x1_min <= x_matrix[0, i] <= x1_max)
            self.assertTrue(x2_min <= x_matrix[1, i] <= x2_max)


    def test_GeneratePoints_BackwardPass_ValuesRangeCorrect(self):
        opt = CreateOptimizer_Instance("test_table3.csv")
        opt.backward_intervals = 4

        opt.major_axis = 0
        for k in range(10):
            x_matrix = opt.GeneratePoints(is_forward=False)
            for i in range(x_matrix.shape[1]):
                value = x_matrix[opt.major_axis, i]
                self.assertTrue(-6.0 <= value <= 3.0)

        opt.major_axis = 1
        for k in range(10):
            x_matrix = opt.GeneratePoints(is_forward=False)
            for i in range(x_matrix.shape[1]):
                value = x_matrix[opt.major_axis, i]
                self.assertTrue(-6.0 <= value <= 3.0)


    def test_GeneratePoints_ExcludesPlatoRegions(self):
        opt = CreateOptimizer_Instance("test_table1.csv")

        # несколько раз проверяем, что диапазон значений по major_axis верен (плюс исключен плато-регион [0, 10] из таблицы)
        opt.major_axis = 0
        for k in range(10):
            x_matrix = opt.GeneratePoints(is_forward=True)
            for i in range(x_matrix.shape[1]):
                self.assertTrue(-10.0 <= x_matrix[opt.major_axis, i] <= 0.0)

        # то же самое по второй координате
        opt.major_axis = 1
        for k in range(10):
            x_matrix = opt.GeneratePoints(is_forward=True)
            for i in range(x_matrix.shape[1]):
                self.assertTrue(-10.0 <= x_matrix[opt.major_axis, i] <= 0.0)


    # =========================================================================================
    # Warmup
    def test_Warmup_ShapeAndValues(self):
        opt = MulDimOptimizer(linear)
        opt.Init(names=["x1", "x2"], mins=[-10, -10], maxs=[10, 10])
        opt.Warmup()

        self.assertEqual(3, opt.known_values.shape[0])

        gt_values = [-10, 0, 10]
        for i in range(3):
            self.assertEqual(gt_values[i], opt.known_values.iloc[i][0])
            self.assertEqual(gt_values[i], opt.known_values.iloc[i][1])


    # =========================================================================================
    # RunValues
    def test_RunValues(self):
        opt = CreateOptimizer_Instance("test_table2.csv")
        opt.major_axis = 0

        old_size = opt.known_values.shape[0]
        x_matrix = opt.GeneratePoints(is_forward=True)
        opt.RunValues(x_matrix)
        new_size = opt.known_values.shape[0]

        self.assertEqual(new_size, old_size + x_matrix.shape[1])
        self.assertEqual(1, opt.major_axis)

        # проверяем, что opt.known_values отсортирован после "очередного" прогона
        table = opt.known_values
        major_column = table.columns[opt.major_axis]
        table.sort_values(by=major_column)
        for i in range(table.shape[0]):
            gt_sorted_value = table.iloc[i][major_column]
            self.assertAlmostEqual(gt_sorted_value, opt.known_values.iloc[i][major_column], 5)



