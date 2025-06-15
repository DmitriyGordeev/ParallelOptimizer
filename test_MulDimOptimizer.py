from unittest import TestCase
import numpy as np
from MulDimOptimizer import MulDimOptimizer


def linear(x):
    return x * 2 + 3.0


def foo2D(x):
    return sum(x)


# TODO: дать возможность указывать как запускать вектор переменных с именами в функцию через функтор
class TestMulDimOptimizer(TestCase):
    def test_run_objective(self):
        opt1 = MulDimOptimizer(linear)
        out1 = opt1.RunObjective(5)

        opt2 = MulDimOptimizer(foo2D)
        out2 = opt2.RunObjective([9, 10])
        pass


    def test_CreateTable(self):
        opt = MulDimOptimizer(linear)
        opt.RunCycle(names=["x1", "x2"], mins=[0, 100], maxs=[0, 100], max_epochs=5)
        pass


    def test_np_mat(self):
        # v = np.ones([2, 1])
        v = np.array([[2, 3], [1, 2]])
        pass