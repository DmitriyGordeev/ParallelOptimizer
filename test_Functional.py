from unittest import TestCase
from test_helper import *


class TestOptimizer_FunctionalRuns(TestCase):

    def test_functional_Linear(self):
        opt = MulDimOptimizer(linear)
        opt.n_probes = 10
        opt.num_workers = 1
        opt.Init(names=["x1", "x2"], mins=[-1, -1], maxs=[20, 20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(20)


    def test_functional_Const(self):
        opt = MulDimOptimizer(const2D)
        opt.n_probes = 3
        opt.num_workers = 1
        opt.Init(names=["x1", "x2"], mins=[-1, -1], maxs=[20, 20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(20)


    def test_functional_Sombrero(self):
        opt = MulDimOptimizer(sombrero)
        opt.n_probes = 10
        opt.num_workers = 5
        opt.Init(names=["x1", "x2"], mins=[-20, -20], maxs=[20, 20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(20)
