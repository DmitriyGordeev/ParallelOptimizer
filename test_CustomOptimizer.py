from random import random
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



def pow2_plato(x):
    if 20 <= x <= 80:
        return (x - 50) * (x - 50)
    return 1000.0


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
        opt = CustomOptimizer(objective=pow2_plato)
        opt.squeeze_factor = 0.5
        opt.RunCycle(names=["X"], mins=[0], maxs=[100], max_epochs=20)

        plot.plot(opt.known_values["X"], opt.known_values["Y"], 'g.')
        plot.grid()
        plot.show()



    def test_GeneratePlatoPoints_BakedTable(self):
        data = pd.read_csv("debug_values_4.csv")
        opt = CustomOptimizer(objective=const_func)
        opt.known_values = data
        opt.mins = [0]
        opt.maxs = [100]

        plato_indexes = opt.FindPlatoRegions()
        opt.plato_indexes = plato_indexes
        opt.MarkPlatoRegions()
        out_values = opt.GeneratePlatoPoints(plato_indexes)

        out_values = sorted(out_values)
        test_values = sorted(list({16.021888000000008, 33.43104288000001, 47.2492, 69.27472863999999, 82.6336}))

        self.assertEqual(len(out_values), len(test_values))
        for i in range(len(test_values)):
            self.assertAlmostEqual(out_values[i], test_values[i], 5)





    def test_RunValues_AddedValuesAreSortedByX(self):
        opt = CustomOptimizer(objective=const_func)
        opt.RunValues([0.0, 50.0, 100.0])

        sorted_X = np.sort(opt.known_values["X"].to_numpy())
        cached_X = opt.known_values["X"].to_numpy()
        self.assertTrue((sorted_X == cached_X).all())



    def test_select_intervals(self):
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




    def test_find_plato_regions(self):
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




    def test_SecondaryAxisExperiment(self):
        data = pd.read_csv("debug_values_5.csv")
        opt = CustomOptimizer(objective=const_func)
        opt.known_values = data
        opt.mins = [0]
        opt.maxs = [100]

        Y_sort = opt.known_values.sample(frac=1)
        Y_sort = opt.known_values.sort_values(by="Y", ascending=False, kind='stable')

        # # filter rows
        # Y_sort = Y_sort[Y_sort["blocked"] == False]
        # Y_sort = Y_sort[(Y_sort["plato_index"] == -1) | (Y_sort["plato_edge"] == True)]

        Y_max = float(Y_sort.iloc[0]["Y"])
        Y_min = float(Y_sort.iloc[-1]["Y"])

        Y_range = Y_max - Y_min
        Y_sort["w"] = [1.0 / Y_sort.shape[0]] * Y_sort.shape[0]
        if Y_range != 0.0:
            Y_sort["w"] += (Y_sort["Y"] - Y_min) / (Y_max - Y_min)
            wsum = Y_sort["w"].sum()
            Y_sort["w"] /= wsum


        # Find lerp value from toss
        # toss = random()
        toss = 0.99
        w_accum = 0.0
        index = -1
        for i in range(Y_sort.shape[0]):
            w_accum += Y_sort.iloc[i]["w"]
            if toss <= w_accum:
                index = i
                break

        u_start = w_accum - Y_sort.iloc[index]["w"]
        u_end = w_accum
        lerp = (toss - u_start) / (u_end - u_start)

        pick_index = Y_sort.index[index]
        X_current = data.loc[pick_index, "X"]


        # # Выбор одной из соседних точек --------------------------------
        # side_toss = 0.4
        # next_point = False
        # if side_toss < 0.5:
        #     adjacent_index = pick_index - 1
        #     adj_row = data.iloc[adjacent_index]
        #     exclude_condition_1 = adj_row["blocked"]
        #     exclude_condition_2 = adj_row["plato_index"] != -1 and not adj_row["plato_edge"]
        #     exclude_condition_3 = adj_row["plato_block"]
        #
        #     if exclude_condition_1 or exclude_condition_2 or exclude_condition_3:
        #         next_point = True
        #     else:
        #         X_adjacent = adj_row["X"]
        # else:
        #     next_point = True
        #
        # if next_point:
        #     adjacent_index = pick_index + 1
        #     X_adjacent = data.loc[adjacent_index, "X"]


        side_toss = 0.4
        next_point = False
        if side_toss < 0.5 or pick_index == data.shape[0] - 1:
            adjacent_index = pick_index - 1
            if adjacent_index < 0:
                next_point = True
            else:
                X_adjacent = data.iloc[adjacent_index]["X"]

        else:
            next_point = True


        if next_point:
            adjacent_index = pick_index + 1
            X_adjacent = data.iloc[adjacent_index]["X"]


        # Выбор новой точки исходя из lerp - значения между соседними
        X_out = X_current + (X_adjacent - X_current) * lerp

        pass






