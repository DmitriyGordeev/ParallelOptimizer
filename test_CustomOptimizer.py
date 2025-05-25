from unittest import TestCase
from CustomOptimizer import CustomOptimizer
import pandas as pd


def func(x: float):
    return 2.0 * x + 3.2
    # return 0.0

class TestCustomOptimizer(TestCase):
    def test_general(self):
        opt = CustomOptimizer()

        opt.known_values = opt.known_values._append({"X": 0, "Y": func(0)}, ignore_index=True)
        opt.known_values = opt.known_values._append({"X": 50, "Y": func(50)}, ignore_index=True)
        opt.known_values = opt.known_values._append({"X": 100, "Y": func(100)}, ignore_index=True)

        opt.SelectIntervals()
        opt.UnitMapping()
        x = opt.UnmapValue(0.72)
        pass



    def test_UnitMapping(self):
        pass