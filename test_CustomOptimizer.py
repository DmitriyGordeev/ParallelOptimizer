from unittest import TestCase
from CustomOptimizer import CustomOptimizer
import pandas as pd
import numpy as np


def func(x: float):
    # return 2.0 * x + 3.2
    return 0.0

class TestCustomOptimizer(TestCase):
    def test_general(self):
        opt = CustomOptimizer(objective=func)

        opt.known_values = opt.known_values._append({"X": 0, "Y": func(0)}, ignore_index=True)
        opt.known_values = opt.known_values._append({"X": 50, "Y": func(50)}, ignore_index=True)
        opt.known_values = opt.known_values._append({"X": 100, "Y": func(100)}, ignore_index=True)

        # Iteration 1
        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)

        # Iteration 2
        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)

        pass



    def test_RunValues_AddedValuesAreSortedByX(self):
        opt = CustomOptimizer(objective=func)
        opt.RunValues([0.0, 50.0, 100.0])

        sorted_X = np.sort(opt.known_values["X"].to_numpy())
        cached_X = opt.known_values["X"].to_numpy()
        self.assertTrue((sorted_X == cached_X).all())
