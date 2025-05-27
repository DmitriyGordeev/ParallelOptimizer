from unittest import TestCase
from CustomOptimizer import CustomOptimizer
import pandas as pd
import numpy as np

import matplotlib.pyplot as plot


def func(x: float):
    return (x - 50) * (x - 50)
    # return 0.0

class TestCustomOptimizer(TestCase):
    def test_general(self):
        opt = CustomOptimizer(objective=func)

        # opt.known_values = opt.known_values._append({"X": 0, "Y": func(0)}, ignore_index=True)
        # opt.known_values = opt.known_values._append({"X": 50, "Y": func(50)}, ignore_index=True)
        # opt.known_values = opt.known_values._append({"X": 100, "Y": func(100)}, ignore_index=True)

        # # Iteration 1
        # opt.SelectIntervals()
        # opt.UnitMapping()
        # new_values = opt.CreateProbePoints()
        # opt.RunValues(new_values)
        #
        # # Iteration 2
        # opt.SelectIntervals()
        # opt.UnitMapping()
        # new_values = opt.CreateProbePoints()
        # opt.RunValues(new_values)

        opt.RunCycle(names=["X"], mins=[0], maxs=[100])

        print(f"internal interations = {opt.internal_itr}")

        plot.plot(opt.known_values["X"], opt.known_values["Y"], 'g.')
        plot.grid()
        plot.show()

        pass



    def test_RunValues_AddedValuesAreSortedByX(self):
        opt = CustomOptimizer(objective=func)
        opt.RunValues([0.0, 50.0, 100.0])

        sorted_X = np.sort(opt.known_values["X"].to_numpy())
        cached_X = opt.known_values["X"].to_numpy()
        self.assertTrue((sorted_X == cached_X).all())



    def test_select_intervals_around(self):
        X_sort = pd.read_csv("debug_values.csv")
        X_sort.drop(X_sort.columns[0], axis=1, inplace=True)

        Y_sort = X_sort.sort_values(by="Y", ascending=False)

        area_indexes = set()
        max_areas = 10

        for row in Y_sort.iterrows():
            index = int(row[0])
            l_index = index - 1
            r_index = index + 1
            if l_index >= 0:
                area_indexes.add((l_index, index))

            if len(area_indexes) == max_areas:
                break

            if r_index < Y_sort.shape[0]:
                area_indexes.add((index, r_index))

            if len(area_indexes) == max_areas:
                break

        pass








