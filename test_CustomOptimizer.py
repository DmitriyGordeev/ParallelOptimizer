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

        opt.mins = [0]
        opt.maxs = [100]
        opt.Warmup()

        # Iteration 1
        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)
        opt.plato_indexes = opt.FindPlatoRegions()
        opt.MarkPlatoRegions()


        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)
        opt.plato_indexes = opt.FindPlatoRegions()
        opt.MarkPlatoRegions()


        opt.SelectIntervals()
        opt.UnitMapping()
        new_values = opt.CreateProbePoints()
        opt.RunValues(new_values)
        opt.plato_indexes = opt.FindPlatoRegions()
        opt.MarkPlatoRegions()



        opt.SelectIntervals(forward=False)



        plot.plot(opt.known_values["X"], opt.known_values["Y"], 'g.')
        plot.grid()
        plot.show()
        pass



    def test_RunCycle(self):
        opt = CustomOptimizer(objective=gaussian)
        opt.squeeze_factor = 0.5
        opt.RunCycle(names=["X"], mins=[0], maxs=[100], max_epochs=6)

        plot.plot(opt.known_values["X"], opt.known_values["Y"], 'g.')
        plot.grid()
        plot.show()



    def test_GeneratePlatoPoints_TableSize1(self):
        data = pd.read_csv("debug_values_3.csv")
        opt = CustomOptimizer(objective=const_func)
        opt.known_values = data
        opt.mins = [0]
        opt.maxs = [100]

        plato_indexes = opt.FindPlatoRegions()
        opt.plato_indexes = plato_indexes
        opt.MarkPlatoRegions()
        # opt.GeneratePlatoPoints(plato_indexes)
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
        X = [0, 1, 2, 3, 4, 5]
        Y = [4, 5, 5, 5, 5, 8]

        eps = 0.01

        data = pd.DataFrame()
        data["X"] = X
        data["Y"] = Y

        # seek plato indexes
        l_index = -1
        r_index = -1
        platos = []

        for i in range(data.shape[0] - 1):
            x = data.iloc[i]["X"]
            y = data.iloc[i]["Y"]

            next_y = data.iloc[i + 1]["Y"]

            if abs(next_y - y) < eps:
                if l_index < 0:
                    l_index = i
                r_index = i
            else:
                if l_index != -1:
                    platos.append([l_index, r_index])
                    l_index = -1
                    r_index = -1

        if l_index != -1:
            platos.append([l_index, r_index])
            l_index = -1
            r_index = -1

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



    def test_pandas_exclude_regions(self):
        data = pd.DataFrame()
        data["X"] = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        data["Y"] = [19, 2, -88, -4, 3, 9, -8, 89, 100]
        pairs = [(0, 2), (5, 7)]

        data = data.drop(index=range(0, 3))
        data = data.drop(index=range(5, 7))
        pass



    def test_mark_plato(self):
        data = pd.DataFrame(columns=["Y", "plato"])
        data["Y"] = [4, 5, 5, 5, 8]
        data["plato"] = [False, False, False, False, False]
        region = (1, 2)

        if region[0] >= region[1]:
            print(f"Don't do anything")

        if region[0] > 0:
            data.loc[region[0] + 1 : region[1], 'plato'] = True
        else:
            data.loc[region[0] : region[1], 'plato'] = True

        pass

