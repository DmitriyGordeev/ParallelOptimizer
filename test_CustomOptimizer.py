from unittest import TestCase
from CustomOptimizer import CustomOptimizer
import pandas as pd
import numpy as np

import matplotlib.pyplot as plot


def pow2(x: float):
    return (x - 50) * (x - 50)
    #

def const_func(x: float):
    return 0.0


def linear(x: float):
    return x


def gaussian(x):
    mu = 40.0
    sig = 10.0
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


class TestCustomOptimizer(TestCase):
    def test_general(self):
        opt = CustomOptimizer(objective=const_func)
        opt.n_probes = 4

        # opt.known_values = opt.known_values._append({"X": 0, "Y": func(0)}, ignore_index=True)
        # opt.known_values = opt.known_values._append({"X": 50, "Y": func(50)}, ignore_index=True)
        # opt.known_values = opt.known_values._append({"X": 100, "Y": func(100)}, ignore_index=True)

        opt.mins = [0]
        opt.maxs = [100]
        opt.Warmup()

        # Iteration 1
        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)

        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)

        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)

        # areas = opt.CreateBackwardIntervalSet()
        opt.SelectIntervals(forward=False)

        # Iteration 2
        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)


        # opt.RunCycle(names=["X"], mins=[0], maxs=[100], max_epochs=3)
        # plot.plot(opt.known_values["X"], opt.known_values["Y"], 'g.')
        # plot.grid()
        # plot.show()

        platos = opt.FindPlatoRegions()
        opt.PlatoUnitMapping(platos)
        pass



    def test_RunValues_AddedValuesAreSortedByX(self):
        opt = CustomOptimizer(objective=const_func)
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




    def test_mark_plato_regions(self):
        X = np.arange(0, 6.2831, 3.141592 / 12)
        Y = np.sin(X)
        Y = np.clip(Y, -0.4, 0.4)

        eps = 0.0001

        data = pd.DataFrame()
        data["X"] = X
        data["Y"] = Y


        # seek plato indexes
        l_index = -1
        r_index = -1
        platos = []

        for i in range(1, data.shape[0]):
            x = data.iloc[i]["X"]

            prev_y = data.iloc[i - 1]["Y"]
            y = data.iloc[i]["Y"]

            if abs(y - prev_y) < eps:
                if l_index < 0:
                    l_index = i - 1
                r_index = i
            else:
                if l_index != -1:
                    lx = data.iloc[l_index]["X"]
                    rx = data.iloc[r_index]["X"]
                    platos.append([lx, rx])
                    l_index = -1
                    r_index = -1

            pass

        plot.plot(X, Y, 'b.')
        plot.plot(X, Y, 'b')
        plot.grid()
        plot.show()



    def test_search_expanded_regions(self):
        old_map = [[10, 23], [30, 40], [60, 80]]
        new_region = [3, 46]

        new_map = []

        for item in old_map:
            if new_region[0] <= item[0] and new_region[1] >= item[1]:
                print(f"new_region {new_region} expanded {item}")
        pass




