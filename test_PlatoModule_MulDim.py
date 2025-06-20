from unittest import TestCase
import pandas as pd
import numpy as np

from MulDimOptimizer import MulDimOptimizer
from PlatoModule_MulDim import PlatoModule_MulDim
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