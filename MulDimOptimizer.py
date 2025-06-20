from random import random
from typing import Optional

import pandas as pd
import numpy as np



class MulDimOptimizer:
    def __init__(self, objective: callable):
        self.known_values = pd.DataFrame()
        self.squeeze_factor = 0.8
        self.major_axis_intervals = pd.DataFrame(columns=["xL", "xR", "cost"])
        self.u_coords = []
        self.n_probes = 5
        self.num_forward_intervals = 7
        self.backward_intervals = 3
        self.backward_fires_each_n = 4
        self.plato_fires_each_n = 2
        self.objective = objective
        self.objective_column = "Objective"
        self.epochs = 0
        self.internal_itr = 0

        self.names = []
        self.mins = []
        self.maxs = []
        self.major_axis = -1

        self.block_eps = 0.01
        self.plato_module = PlatoModule_MulDim(self)

        # # Debug --------------
        # self.debug_old_X = []
        # self.debug_old_Y = []
        # self.debug_new_X = []
        # self.debug_new_Y = []



    def SetupEps(self, block_eps = 0.01, plato_x_eps = 0.1, plato_y_eps = 0.1, plato_block_eps = 0.01):
        self.block_eps = block_eps
        self.plato_module.plato_x_eps = plato_x_eps
        self.plato_module.plato_y_eps = plato_y_eps
        self.plato_module.plato_x_block_eps = plato_block_eps



    def Init(self, names: list, mins: list, maxs: list):
        assert len(mins) > 0
        assert len(mins) == len(maxs) == len(names)
        self.names = names
        self.mins = mins
        self.maxs = maxs

        objective_column = "Objective"
        if "Objective" in self.names:
            for n in names:
                objective_column += "_" + n
        
        columns = self.names + [objective_column, "blocked", "plato_block", "plato_index", "plato_edge"]
        self.known_values = pd.DataFrame(columns=columns)
        self.objective_column = objective_column
        self.major_axis = 0



    def SelectIntervals(self, forward=True) -> bool:
        min_y = self.known_values[self.objective_column].min()
        max_y = self.known_values[self.objective_column].max()

        # cleanup values from previous run
        self.major_axis_intervals = self.major_axis_intervals.drop(self.major_axis_intervals.index)

        assert self.major_axis >= 0

        major_column = self.known_values.columns[self.major_axis]
        X = self.known_values[major_column].to_numpy()
        Y = self.known_values[self.objective_column].to_numpy()

        if forward:
            areas = self.CreateIntervalSet()
        else:
            areas = self.CreateBackwardIntervalSet()

        if len(areas) == 0:
            return False

        sum_cost = 0
        for pair in areas:
            iL = pair[0]
            iR = pair[1]

            dx = X[iR] - X[iL]
            mid_point = (X[iR] + X[iL]) / 2
            xL = mid_point - dx * self.squeeze_factor / 2
            xR = mid_point + dx * self.squeeze_factor / 2

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

            self.major_axis_intervals = self.major_axis_intervals._append({
                "xL": xL,
                "xR": xR,
                "cost": y1_cost + y2_cost
            }, ignore_index=True)

        # rescale weights
        self.major_axis_intervals["cost"] = self.major_axis_intervals["cost"] / sum_cost
        assert abs(self.major_axis_intervals["cost"].sum() - 1.0) <= 0.00001  # не eps!
        self.major_axis_intervals = self.major_axis_intervals.sample(frac=1)

        sort_order_ascending = True
        if forward:
            sort_order_ascending = False
        self.major_axis_intervals = self.major_axis_intervals.sort_values(by="cost", ascending=sort_order_ascending, kind='stable')
        return True




    def CreateIntervalSet(self):
        Y_sort = self.known_values.sample(frac=1)
        Y_sort = Y_sort.sort_values(by=self.objective_column, ascending=False, kind='stable')

        # select only from non-blocked intervals
        Y_sort = Y_sort[Y_sort["blocked"] == False]

        # Also filter out potential plato regions
        Y_sort = Y_sort[(Y_sort["plato_index"] == -1) | (Y_sort["plato_edge"] == True)]

        area_indexes = set()
        # minmax_delta = (self.maxs[0] - self.mins[0])

        for row in Y_sort.iterrows():
            index = int(row[0])
            l_index = index - 1
            r_index = index + 1

            if l_index >= 0:
                if self.CanSelectPoint(l_index, index):
                    area_indexes.add((l_index, index))
                    if len(area_indexes) == self.num_forward_intervals:
                            break

            if r_index < Y_sort.shape[0]:
                if self.CanSelectPoint(index, r_index):
                    area_indexes.add((index, r_index))
                    if len(area_indexes) == self.num_forward_intervals:
                            break

        return area_indexes


    def CreateBackwardIntervalSet(self):
        if (self.known_values.shape[0] - 1) < self.num_forward_intervals:
            # TODO: сделать проверку там где используется вывод на set empty
            return set()

        # Рассчитываем максимальное количество backward-интервалов которые можем рассмотреть
        n_backwards_areas = (self.known_values.shape[0] - 1) - self.num_forward_intervals
        n_backwards_areas = min(n_backwards_areas, self.backward_intervals)
        if n_backwards_areas <= 0:
            return set()

        # Перемешиваем, чтобы не создавать преференцию по X
        # и сортриуем в обратном порядке нежели чем в CreateIntervalSet()
        Y_sort = self.known_values.sample(frac=1)
        Y_sort = Y_sort.sort_values(by=self.objective_column, ascending=True, kind='stable')

        # select only from non-blocked intervals
        Y_sort = Y_sort[Y_sort["blocked"] == False]

        # Also filter out potential plato regions
        Y_sort = Y_sort[(Y_sort["plato_index"] == -1) | (Y_sort["plato_edge"] == True)]

        area_indexes = set()
        # minmax_delta = (self.maxs[0] - self.mins[0])

        for row in Y_sort.iterrows():
            index = int(row[0])
            l_index = index - 1
            r_index = index + 1

            if l_index >= 0:
                if self.CanSelectPoint(l_index, index):
                    area_indexes.add((l_index, index))
                    if len(area_indexes) == self.backward_intervals:
                        break

            if r_index < Y_sort.shape[0]:
                if self.CanSelectPoint(index, r_index):
                    area_indexes.add((index, r_index))
                    if len(area_indexes) == self.backward_intervals:
                        break

        return area_indexes



    def CanSelectPoint(self, l_index: int, r_index: int) -> bool:
        assert l_index >= 0 and r_index >= 0

        plato_index_l = self.known_values.iloc[l_index]["plato_index"]
        plato_index_r = self.known_values.iloc[r_index]["plato_index"]

        # Check if selected X points are not from plato region
        # and not marked as 'blocked'
        is_not_inside_plato = plato_index_l == -1 or plato_index_r == -1 or (plato_index_l != plato_index_r)
        is_not_blocked = not self.known_values.iloc[l_index]["blocked"]

        major_column = self.known_values.columns[self.major_axis]
        if is_not_inside_plato and is_not_blocked:
            xL = self.known_values.iloc[l_index][major_column]
            xR = self.known_values.iloc[r_index][major_column]
            dx = xR - xL

            # Mark small interval (l_index, index) as blocked
            if abs(dx) <= self.block_eps:
                self.known_values.at[l_index, 'blocked'] = True
            else:
                return True
        return False


    def UnitMapping(self):
        # Placing on unit-len:
        u_coords = []
        u_cursor = 0.0
        u_gap = 0.01
        for row in self.major_axis_intervals.iterrows():
            item = row[1]
            w = item["cost"]
            u_coords.append((u_cursor, u_cursor + w))
            u_cursor += w + u_gap
        self.u_coords.clear()
        self.u_coords = u_coords


    def UnmapValue(self, u_pick: float) -> float:
        # Unmapping picked values
        gap_hit = False
        X_unmapped = 0.0
        for i, u_pair in enumerate(self.u_coords):
            if gap_hit:
                if u_pick < u_pair[0]:
                    u_pick = u_pair[0] + 0.01 * (u_pair[1] - u_pair[0])

            if u_pair[0] <= u_pick <= u_pair[1]:
                alpha = (u_pick - u_pair[0]) / (u_pair[1] - u_pair[0])
                xL = self.major_axis_intervals.iloc[i]["xL"]
                xR = self.major_axis_intervals.iloc[i]["xR"]
                X_unmapped = xL + (xR - xL) * alpha
                break

            else:
                gap_hit = True
        return X_unmapped



    def CreateProbePoints(self) -> list:
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



    def SelectMajorAxisPoints(self, is_forward: bool) -> list:
        if not self.SelectIntervals(is_forward):
            return []
        self.UnitMapping()
        new_X = self.CreateProbePoints()
        return new_X



    def SelectSinglePointOnMinorAxis(self, axis: int) -> float:
        assert 0 <= axis < len(self.mins)
        assert axis != self.major_axis
        axis = self.known_values.columns[axis]

        Y_sort = self.known_values.sample(frac=1)
        Y_sort = Y_sort.sort_values(by=self.objective_column, ascending=False, kind='stable')
        Y_max = float(Y_sort.iloc[0][self.objective_column])
        Y_min = float(Y_sort.iloc[-1][self.objective_column])

        Y_range = Y_max - Y_min
        Y_sort["w"] = [1.0 / Y_sort.shape[0]] * Y_sort.shape[0]
        if Y_range != 0.0:
            Y_sort["w"] += (Y_sort[self.objective_column] - Y_min) / (Y_max - Y_min)
            wsum = Y_sort["w"].sum()
            Y_sort["w"] /= wsum

        # Find lerp value from toss
        toss = random()
        # toss = 0.99
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
        X_current = self.known_values.loc[pick_index, axis]

        # случайно выбираем предыдущую или следующую точку как примыкающую
        side_toss = random()
        next_point = False
        if side_toss < 0.5 or pick_index == self.known_values.shape[0] - 1:
            adjacent_index = pick_index - 1
            if adjacent_index < 0:
                next_point = True
            else:
                X_adjacent = self.known_values.iloc[adjacent_index][axis]

        else:
            next_point = True

        if next_point:
            adjacent_index = pick_index + 1
            X_adjacent = self.known_values.iloc[adjacent_index][axis]

        # Выбор новой точки исходя из lerp - значения между соседними
        X_out = X_current + (X_adjacent - X_current) * lerp
        return X_out




    """ returns matrix (n-axes x m-points) of new coords for objective to run """
    def GeneratePoints(self, is_forward=True) -> np.array:
        # предполагаем что на данный момент уже выбрана major_axis из предыдущей итерации

        major_values = self.SelectMajorAxisPoints(is_forward)
        if len(major_values) == 0:
            print(f"stop iterations, no areas left to divide further")
            return np.array([])

        num_points = len(major_values)

        out_matrix = [[0] * len(major_values)] * len(self.mins)
        out_matrix[self.major_axis] = major_values

        # идем по остальным осям и берем значения
        for i in range(len(self.mins)):
            if i == self.major_axis:
                # пропускаем major axis так как уже добавлен
                continue

            axis_values = []
            for j in range(num_points):
                value = self.SelectSinglePointOnMinorAxis(i)
                axis_values.append(value)
                # TODO: проверить что значения в axis_values не повторяются

            assert len(axis_values) == len(major_values)
            out_matrix[i] = axis_values
        return np.array(out_matrix)



    def Warmup(self):
        # TODO: можно сделать случайным:
        self.major_axis = 0

        x_warmup = np.zeros([len(self.mins), 3])
        for axis_index in range(len(self.mins)):
            x_warmup[axis_index, 0] = self.mins[axis_index]
            x_warmup[axis_index, 1] = (self.mins[axis_index] + self.maxs[axis_index]) / 2.0
            x_warmup[axis_index, 2] = self.maxs[axis_index]

        self.RunValues(x_warmup)



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
                self.objective_column: y,
                "blocked": False,
                "plato_block": False,
                "plato_index": -1,
                "plato_edge": False,
            }

            for i, name in enumerate(self.names):
                result_dict[name] = x_point[i]

            self.known_values = self.known_values._append(result_dict, ignore_index=True)

        # Переход на следующую major_axis
        self.major_axis += 1
        if self.major_axis == len(self.mins):
            self.major_axis = 0

        # Сортировка и сброс индексов для следующей стадии SelectIntervals() по новой оси
        major_column = self.known_values.columns[self.major_axis]
        self.known_values = self.known_values.sort_values(by=major_column)
        self.known_values.reset_index(inplace=True, drop=True)
        pass


    def RunCycle(self, max_epochs: int):
        assert max_epochs > 0

        backward_countdown = self.backward_fires_each_n
        plato_countdown = self.plato_fires_each_n

        for i in range(max_epochs):
            self.epochs += 1
            if self.epochs == 1:
                self.Warmup()
            else:

                # Check if it is a plato stage
                if plato_countdown == 0:
                    print(f" >>> running plato iteration at epoch = {self.epochs}")
                    plato_countdown = self.plato_fires_each_n
                    if len(self.plato_module.plato_indexes) == 0:
                        print(f"no plato detected, skip plato iteration")
                        continue
                    new_X = self.plato_module.GeneratePlatoPoints()
                    # self.debug_old_X = self.known_values["X"].to_list()
                    # self.debug_old_Y = self.known_values[self.objective_column].to_list()

                    if len(new_X) == 0:
                        print("Plato run: new_X is empty array, skip this iteration")
                        continue

                    self.RunValues(new_X)
                    # self.DebugPlot("plato")

                    self.plato_module.FindPlatoRegions()
                    self.plato_module.MarkPlatoRegions()
                    continue
                else:
                    plato_countdown -= 1


                # Check if it is backward pass stage
                is_forward = True
                if backward_countdown == 0:
                    is_forward = False
                    debug_name = "backward"
                    backward_countdown = self.backward_fires_each_n
                    print(f" >>> running backward iteration at epoch = {i}")
                else:
                    backward_countdown -= 1

                x_matrix = self.GeneratePoints(is_forward=is_forward)
                if x_matrix.shape[0] == 0:
                    if len(self.plato_module.plato_indexes) > 0:
                        print(f"only plato regions left at this moment, continue next epoch with plato run")
                        plato_countdown = 0
                        continue
                    break

                self.RunValues(x_matrix)
                self.plato_module.FindPlatoRegions()
                self.plato_module.MarkPlatoRegions()

                print(f"epoch = {self.epochs}, known_values.size = {self.known_values.shape[0]}")

        self.known_values.sort_values(by=self.objective_column, ascending=False, inplace=True)
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> ")
        print(f"Total epochs = {self.epochs}, internal_iterations = {self.internal_itr}\n"
              f"\ntop values: -------------------- \n")
        print(self.known_values.head(5))




from PlatoModule_MulDim import PlatoModule_MulDim