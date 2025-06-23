from unittest import TestCase
from test_helper import *


class TestOptimizer_FunctionalRuns(TestCase):
    def test_functional_runs(self):
        opt = MulDimOptimizer(decay)
        opt.Init(names=["x1", "x2"], mins=[-10, -10], maxs=[10, 10])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(10)
