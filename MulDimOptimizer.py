import pandas as pd
import numpy as np


class MulDimOptimizer:
    def __init__(self, objective: callable):
        self.known_values = pd.DataFrame()
        self.squeeze_factor = 0.8
        self.major_axis_intervals = pd.DataFrame(columns=["x0", "x1", "cost"])
        self.u_coords = []
        self.n_probes = 5
        self.num_forward_intervals = 7
        self.backward_intervals = 3
        self.backward_fires_each_n = 4
        self.plato_fires_each_n = 2
        self.objective = objective
        self.epochs = 0
        self.internal_itr = 0

        self.names = []
        self.mins = []
        self.maxs = []

        self.plato_indexes = []

        self.block_eps = 0.01
        self.plato_X_eps = 10.0
        self.plato_block_eps = 0.5
        self.plato_Y_eps = 0.01

        # # Debug --------------
        # self.debug_old_X = []
        # self.debug_old_Y = []
        # self.debug_new_X = []
        # self.debug_new_Y = []



    def CreateTable(self):
        assert len(self.mins) > 0
        assert len(self.mins) == len(self.maxs) == len(self.names)
        # TODO: сделать так, чтобы Y не повторялся в названиях колонок, если одна из переменных тоже названа также
        columns = self.names + ["Y", "blocked", "plato_block", "plato_index", "plato_edge"]
        self.known_values = pd.DataFrame(columns=columns)




    def SelectIntervals(self):
        # TODO: отсортировать по major X в порядке возрастания
        pass







    def RunObjective(self, x):
        self.internal_itr += 1
        return self.objective(x)


    """ x_matrix - rows = axes, columns = points """
    def RunValues(self, x_matrix: np.array):
        assert len(x_matrix) > 0
        for column in range(x_matrix.shape[1]):
            x_point = x_matrix[:, column]

            y = self.RunObjective(x_point)
            # self.debug_new_X.append(x)
            # self.debug_new_Y.append(y)

            result_dict = {
                "Y": y,
                "blocked": False,
                "plato_block": False,
                "plato_index": -1,
                "plato_edge": False,
            }

            for i, name in enumerate(self.names):
                result_dict[name] = x_point[i]

            self.known_values = self.known_values._append(result_dict, ignore_index=True)

        # self.known_values = self.known_values.sort_values(by="X")
        # self.known_values.reset_index(inplace=True, drop=True)
        pass


    def RunCycle(self, names: list, mins: list, maxs: list, max_epochs: int):
        self.names = names
        self.mins = mins
        self.maxs = maxs
        self.CreateTable()
        # TODO:


