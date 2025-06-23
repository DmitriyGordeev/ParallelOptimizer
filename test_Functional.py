from unittest import TestCase
from test_helper import *


class TestOptimizer_FunctionalRuns(TestCase):
    def test_functional_runs(self):
        opt = MulDimOptimizer(foo2D)
        opt.Init(names=["x1", "x2"], mins=[-10, -10], maxs=[10, 10])
        opt.RunCycle(10)
