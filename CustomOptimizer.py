import enum

import numpy as np
import pandas as pd


class CustomOptimizer:
    def __init__(self, objective: callable):
        self.known_values = pd.DataFrame(columns=["X", "Y", "blocked"])
        self.squeeze_factor = 0.8
        self.intervals = pd.DataFrame(columns=["x0", "x1", "cost"])
        self.u_coords = []
        self.n_probes = 5
        self.forward_intervals = 7
        self.backward_intervals = 3
        self.backward_fires_each_n = 5
        self.objective = objective
        self.epochs = 0
        self.internal_itr = 0

        self.names = []
        self.mins = []
        self.maxs = []

        self.eps = 0.001
        self.min_interval_size_ratio = 0.01



    def SelectIntervals(self, forward=True) -> bool:
        min_y = self.known_values["Y"].min()
        max_y = self.known_values["Y"].max()

        # cleanup values from previous run
        self.intervals = self.intervals.drop(self.intervals.index)

        sum_cost = 0

        # Assuming known_values is sorted by X (increasing order)
        X = self.known_values["X"].to_numpy()
        Y = self.known_values["Y"].to_numpy()

        if forward:
            areas = self.CreateIntervalSet()
        else:
            areas = self.CreateBackwardIntervalSet()

        if len(areas) == 0:
            return False

        for pair in areas:
            iL = pair[0]
            iR = pair[1]

            dx = X[iR] - X[iL]
            mid_point = (X[iR] + X[iL]) / 2
            x0 = mid_point - dx * self.squeeze_factor / 2
            x1 = mid_point + dx * self.squeeze_factor / 2

            # TODO: сделать параметром eps (0.0001) -
            #  точность, при которой в наших масштабах
            #  мы считаем значения равными везде в оптимайзере?
            if max_y - min_y <= 0.00001:
                y1_cost = 1.0
                y2_cost = 1.0
            else:
                y1_cost = (Y[iL] - min_y) / (max_y - min_y) + 0.1
                y2_cost = (Y[iR] - min_y) / (max_y - min_y) + 0.1
            sum_cost += y1_cost + y2_cost

            self.intervals = self.intervals._append({
                "x0": x0,
                "x1": x1,
                "cost": y1_cost + y2_cost
            }, ignore_index=True)

        # rescale weights
        self.intervals["cost"] = self.intervals["cost"] / sum_cost
        assert abs(self.intervals["cost"].sum() - 1.0) <= 0.00001  # не eps!
        self.intervals = self.intervals.sample(frac=1)

        sort_order_ascending = True
        if forward:
            sort_order_ascending = False
        self.intervals = self.intervals.sort_values(by="cost", ascending=sort_order_ascending, kind='stable')
        return True


    def CreateIntervalSet(self):
        Y_sort = self.known_values.sample(frac=1)
        Y_sort = Y_sort.sort_values(by="Y", ascending=False, kind='stable')

        # select only from non-blocked intervals
        Y_sort = Y_sort[Y_sort["blocked"] == False]
        area_indexes = set()
        minmax_delta = (self.maxs[0] - self.mins[0])

        for row in Y_sort.iterrows():
            index = int(row[0])
            l_index = index - 1
            r_index = index + 1

            if l_index < 0 or r_index >= Y_sort.shape[0]:
                continue

            # Mark small interval (l_index, index) as blocked and skip its consideration
            xL = self.known_values.iloc[l_index]["X"]
            xR = self.known_values.iloc[index]["X"]
            dx = xR - xL
            if abs(dx / minmax_delta) <= self.min_interval_size_ratio:
                # self.known_values.iloc[l_index]["blocked"] = True
                self.known_values.at[l_index, 'blocked'] = True
            else:
                area_indexes.add((l_index, index))
                if len(area_indexes) == self.forward_intervals:
                    break

            # Mark small interval (index, r_index) as blocked and skip its consideration
            xL = self.known_values.iloc[index]["X"]
            xR = self.known_values.iloc[r_index]["X"]
            dx = xR - xL
            if abs(dx / minmax_delta) <= self.min_interval_size_ratio:
                # self.known_values.iloc[index]["blocked"] = True
                self.known_values.at[index, 'blocked'] = True
            else:
                area_indexes.add((index, r_index))
                if len(area_indexes) == self.forward_intervals:
                    break

        return area_indexes


    def CreateBackwardIntervalSet(self):
        if (self.known_values.shape[0] - 1) < self.forward_intervals:
            # TODO: сделать проверку там где используется вывод на set empty
            return set()

        # Рассчитываем максимальное количество backward-интервалов которые можем рассмотреть
        n_backwards_areas = (self.known_values.shape[0] - 1) - self.forward_intervals
        n_backwards_areas = min(n_backwards_areas, self.backward_intervals)
        if n_backwards_areas <= 0:
            return set()

        # Перемешиваем, чтобы не создавать преференцию по X
        # и сортриуем в обратном порядке нежели чем в CreateIntervalSet()
        Y_sort = self.known_values.sample(frac=1)
        Y_sort = Y_sort.sort_values(by="Y", ascending=True, kind='stable')

        # select only from non-blocked intervals
        Y_sort = Y_sort[Y_sort["blocked"] == False]
        area_indexes = set()
        minmax_delta = (self.maxs[0] - self.mins[0])

        for row in Y_sort.iterrows():
            index = int(row[0])
            l_index = index - 1
            r_index = index + 1

            if l_index < 0 or r_index >= Y_sort.shape[0]:
                continue

            # Mark small interval (l_index, index) as blocked and skip its consideration
            xL = self.known_values.iloc[l_index]["X"]
            xR = self.known_values.iloc[index]["X"]
            dx = xR - xL
            if abs(dx / minmax_delta) <= self.min_interval_size_ratio:
                # self.known_values.iloc[l_index]["blocked"] = True
                self.known_values.at[l_index, 'blocked'] = True
            else:
                area_indexes.add((l_index, index))
                if len(area_indexes) == self.backward_intervals:
                    break

            # Mark small interval (index, r_index) as blocked and skip its consideration
            xL = self.known_values.iloc[index]["X"]
            xR = self.known_values.iloc[r_index]["X"]
            dx = xR - xL
            if abs(dx / minmax_delta) <= self.min_interval_size_ratio:
                # self.known_values.iloc[index]["blocked"] = True
                self.known_values.at[index, 'blocked'] = True
            else:
                area_indexes.add((index, r_index))
                if len(area_indexes) == self.backward_intervals:
                    break

        return area_indexes





    def UnitMapping(self):
        # Placing on unit-len:
        u_coords = []
        u_cursor = 0.0
        u_gap = 0.01
        for row in self.intervals.iterrows():
            item = row[1]
            w = item["cost"]
            u_coords.append((u_cursor, u_cursor + w))
            u_cursor += w + u_gap
        self.u_coords.clear()
        self.u_coords = u_coords


    def UnmapValue(self, u_pick: float) -> float:
        # Unmapping picked values
        u_potential_gap_hit = False
        X_unmapped = 0.0
        for i, u_pair in enumerate(self.u_coords):
            if u_potential_gap_hit:
                if u_pick < u_pair[0]:
                    u_pick = u_pair[0] + 0.01 * (u_pair[1] - u_pair[0])

            if u_pair[0] <= u_pick <= u_pair[1]:
                alpha = (u_pick - u_pair[0]) / (u_pair[1] - u_pair[0])
                x0 = self.intervals.iloc[i]["x0"]
                x1 = self.intervals.iloc[i]["x1"]
                X_unmapped = x0 + (x1 - x0) * alpha
                break

            else:
                u_potential_gap_hit = True
        return X_unmapped


    def CreateProbePoints(self) -> list:
        # num_probes = min(self.n_probes, len(self.u_coords))
        num_probes = self.n_probes
        assert num_probes > 0

        u_len = self.u_coords[-1][1]
        u_step = u_len / (num_probes + 1)
        out = [0.0] * num_probes
        u_pick = u_step
        for i in range(num_probes):
            out[i] = self.UnmapValue(u_pick)
            u_pick += u_step
        return out


    def RunValues(self, values: list):
        assert len(values) > 0
        for x in values:
            y = self.objective(x)
            self.internal_itr += 1
            self.known_values = self.known_values._append({
                "X": x,
                "Y": y,
                "blocked": False
            }, ignore_index=True)
        self.known_values = self.known_values.sort_values(by="X")
        self.known_values.reset_index(inplace=True, drop=True)
        pass



    def Warmup(self):
        # TODO: обобщить на вектор параметров:

        y = self.objective(self.mins[0])
        self.known_values = self.known_values._append({
            "X": self.mins[0],
            "Y": y,
            "blocked": False
        }, ignore_index=True)

        y = self.objective(self.maxs[0])
        self.known_values = self.known_values._append({
            "X": self.maxs[0],
            "Y": y,
            "blocked": False
        }, ignore_index=True)

        middle = (self.mins[0] + self.maxs[0]) / 2.0
        y = self.objective(middle)
        self.known_values = self.known_values._append({
            "X": middle,
            "Y": y,
            "blocked": False
        }, ignore_index=True)

        # TODO: засунуть в функцию к self.objective(...) и вызывать вместе
        self.internal_itr += 3
        self.known_values = self.known_values.sort_values(by="X")
        self.known_values.reset_index(inplace=True, drop=True)



    def RunCycle(self, names: list, mins: list, maxs: list, max_epochs: int):
        assert len(names) == len(mins) == len(maxs) != 0
        self.names = names
        self.mins = mins
        self.maxs = maxs

        backward_counter = self.backward_fires_each_n
        for i in range(max_epochs):
            if self.epochs == 0:
                self.Warmup()
            else:

                forward = True
                if backward_counter == 0:
                    forward = False
                    backward_counter = self.backward_fires_each_n
                    print(f"Running backward iteration at epoch = {i}")
                else:
                    backward_counter -= 1

                if not self.SelectIntervals(forward):
                    print(f"Stop iterations, no areas left to divide further")
                    break
                self.UnitMapping()
                new_X = self.CreateProbePoints()
                self.RunValues(new_X)
                self.FindPlatoRegions()
                print(f"epoch = {self.epochs}, known_values.size = {self.known_values.shape[0]}")

            self.epochs += 1

        self.known_values.sort_values(by="Y", ascending=False, inplace=True)
        print(f"epochs = {self.epochs}, internal_iterations = {self.internal_itr}\n"
              f"\n-------- top values:\n")
        print(self.known_values.head(5))


    def FindPlatoRegions(self):

        plato_x_eps = 10.0     # TODO: min_max_delta , min_size_ratio
        plato_y_eps = 0.0001

        # seek plato indexes
        l_index = -1
        r_index = -1
        platos = []

        for i in range(1, self.known_values.shape[0]):
            prev_x = self.known_values.iloc[i - 1]["X"]
            prev_y = self.known_values.iloc[i - 1]["Y"]

            x = self.known_values.iloc[i]["X"]
            y = self.known_values.iloc[i]["Y"]

            if abs(x - prev_x) < plato_x_eps and abs(y - prev_y) < plato_y_eps:
                if l_index < 0:
                    l_index = i - 1
                r_index = i
            else:
                if l_index != -1:
                    lx = self.known_values.iloc[l_index]
                    rx = self.known_values.iloc[r_index]
                    platos.append([lx, rx])
                    l_index = -1
                    r_index = -1
        return platos

