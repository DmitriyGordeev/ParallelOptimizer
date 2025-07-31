from unittest import TestCase
from test_helper import *

""" Тесты на разные математические функции """

class TestOptimizer_FunctionalRuns(TestCase):

    def test_functional_Linear1D(self):
        opt = ParallelOptimizer(linear1D)
        opt.n_probes = 1
        opt.num_workers = 1
        opt.Init(names=["x1"], mins=[-20], maxs=[20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(5)


    def test_functional_gaussian1D(self):
        opt = ParallelOptimizer(gaussian1D)
        opt.n_probes = 5
        opt.num_workers = 5
        opt.Init(names=["x1"], mins=[-5], maxs=[5])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(5)


    def test_functional_Linear2D(self):
        opt = ParallelOptimizer(linear2D)
        opt.n_probes = 10
        opt.num_workers = 1
        opt.Init(names=["x1", "x2"], mins=[-1, -1], maxs=[20, 20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(20)


    def test_functional_Const(self):
        opt = ParallelOptimizer(const2D)
        opt.n_probes = 3
        opt.num_workers = 1
        opt.Init(names=["x1", "x2"], mins=[-1, -1], maxs=[20, 20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(20)


    def test_functional_Sombrero(self):
        opt = ParallelOptimizer(sombrero)
        opt.n_probes = 10
        opt.num_workers = 5
        opt.Init(names=["x1", "x2"], mins=[-20, -20], maxs=[20, 20])
        opt.SetupEps(block_eps=0.01, plato_block_eps=0.05, plato_x_eps=4.0, plato_y_eps=0.1)
        opt.RunCycle(20)
