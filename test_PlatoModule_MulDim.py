from unittest import TestCase
from test_helper import *



class TestPlatoModule_MulDim(TestCase):


    # ================================================================================
    # FindPlatoRegions()
    def test_FindPlatoRegions(self):
        opt = CreateOptimizer_Instance("test_table5.csv")
        opt.SetupEps(block_eps=0.001, plato_x_eps=3.0, plato_y_eps=0.5, plato_block_eps=0.05)
        opt.plato_module.FindPlatoRegions()

        self.assertTrue(len(opt.plato_module.plato_indexes) == 1)
        self.assertEqual(opt.plato_module.plato_indexes[0][0], 2)
        self.assertEqual(opt.plato_module.plato_indexes[0][1], 5)
        self.assertTrue(opt.known_values.iloc[4]["plato_block"])


    # ================================================================================
    # MarkPlatoRegions()
    def test_MarkPlatoRegions(self):
        opt = CreateOptimizer_Instance("test_table5.csv")
        opt.plato_module.plato_indexes = [(1, 2), (4, 5)]
        opt.plato_module.MarkPlatoRegions()

        plato_indexes = opt.known_values["plato_index"].to_list()
        plato_edge = opt.known_values["plato_edge"].to_list()

        gt_plato_indexes = [-1, 0, 0, 0, 1, 1, 1, -1]
        gt_plato_edge    = [False, True, False, True, True, False, True, False]

        for i in range(opt.known_values.shape[0]):
            self.assertEqual(gt_plato_indexes[i], plato_indexes[i])
            self.assertEqual(gt_plato_edge[i], plato_edge[i])


    # ================================================================================
    # GroupTables()
    def test_GroupTables(self):
        opt = CreateOptimizer_Instance("test_table6.csv")
        opt.SetupEps(block_eps=0.001, plato_x_eps=3.0, plato_y_eps=0.5, plato_block_eps=0.05)
        opt.plato_module.plato_indexes = [(1, 2), (4, 6)]
        opt.plato_module.GroupTables()

        self.assertEqual(len(opt.plato_module.plato_regions), 2)

        # --- region 0 -----
        region_0 = opt.plato_module.plato_regions[0]
        out_plato_edge = region_0["plato_edge"].to_list()
        gt_plato_edge = [True, False, True]

        out_plato_original_index = region_0["original_index"].to_list()
        gt_plato_original_index = [1, 2, 3]

        for i in range(region_0.shape[0]):
            self.assertEqual(out_plato_edge[i], gt_plato_edge[i])
            self.assertEqual(out_plato_original_index[i], gt_plato_original_index[i])


        # --- region 1 -----
        region_1 = opt.plato_module.plato_regions[1]
        out_plato_edge = region_1["plato_edge"].to_list()
        gt_plato_edge = [True, False, True]

        out_plato_original_index = region_1["original_index"].to_list()
        gt_plato_original_index = [4, 6, 7]

        for i in range(region_0.shape[0]):
            self.assertEqual(out_plato_edge[i], gt_plato_edge[i])
            self.assertEqual(out_plato_original_index[i], gt_plato_original_index[i])


    # ================================================================================
    # UnitMapRegions()
    def test_UnitMapRegions(self):
        opt = CreateOptimizer_Instance("test_table6.csv")

        region_1 = pd.read_csv("test_table1.csv")
        region_2 = pd.read_csv("test_table2.csv")
        region_3 = pd.read_csv("test_table3.csv")

        opt.plato_module.plato_regions = [
            region_1,
            region_2,
            region_3
        ]

        opt.plato_module.sum_shapes = region_1.shape[0] + region_2.shape[0] + region_3.shape[0]
        opt.plato_module.UnitMapRegions()

        gt_ucoords = {0: (0.0, 0.16129),
                      1: (0.17129, 0.3325),
                      2: (0.3425, 1.02)}

        for key, item in opt.plato_module.region_ucoords.items():
            self.assertAlmostEqual(gt_ucoords[key][0], item[0], 3)
            self.assertAlmostEqual(gt_ucoords[key][1], item[1], 3)


    # ================================================================================
    # UnitMapRegions()
    def test_UnmapX(self):
        opt = CreateOptimizer_Instance("test_table6.csv")


        # ---- region 1 ----
        region_1 = pd.read_csv("plato_region_1.csv")
        x = opt.plato_module.UnmapX(region_1, 0.0, (0.0, 1.0))
        self.assertEqual(-2.5, x[0])
        self.assertEqual(-2.5, x[1])

        x = opt.plato_module.UnmapX(region_1, 1.0, (0.0, 1.0))
        self.assertEqual(0.9, x[0])
        self.assertEqual(0.9, x[1])


        # ---- region 2 ----
        region_2 = pd.read_csv("plato_region_2.csv")
        x = opt.plato_module.UnmapX(region_2, 0.0, (0.0, 1.0))
        self.assertAlmostEqual(2.5, x[0], 3)
        self.assertAlmostEqual(2.5, x[1], 3)

        x = opt.plato_module.UnmapX(region_2, 1.0, (0.0, 1.0))
        self.assertAlmostEqual(3.0, x[0], 3)
        self.assertAlmostEqual(3.0, x[1], 3)


    # ================================================================================
    # UnmapValues()
    def test_UnmapValues(self):
        opt = CreateOptimizer_Instance("test_table6.csv")
        opt.SetupEps(block_eps=0.001, plato_x_eps=3.0, plato_y_eps=0.5, plato_block_eps=0.05)
        opt.plato_module.plato_indexes = [(1, 2), (4, 6)]
        opt.plato_module.GroupTables()
        opt.plato_module.UnitMapRegions()

        gt_x_plato_points = np.array([
            [-2.5, 0.9, 2.5, 3.00005],
            [-2.5, 0.9, 2.5, 3.0004999999999997]
        ])

        x_plato_points = opt.plato_module.UnmapValues()

        self.assertEqual(gt_x_plato_points.shape, x_plato_points.shape)

        for i in range(gt_x_plato_points.shape[0]):
            for j in range(gt_x_plato_points.shape[1]):
                gt_value = gt_x_plato_points[i, j]
                out_value = x_plato_points[i, j]
                self.assertAlmostEqual(gt_value, out_value, 3)



