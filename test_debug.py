from unittest import TestCase
from ParallelOptimizer import ParallelOptimizer
from test_helper import linear



class TestDebug(TestCase):
    def test_1D_problem(self):
        opt = ParallelOptimizer(linear)
        opt.n_probes = 1
        opt.Init(names=["x"], mins=[-10], maxs=[10])
        opt.RunCycle(10)
        pass

