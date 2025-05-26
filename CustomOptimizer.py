import numpy as np
import pandas as pd


class CustomOptimizer:
    def __init__(self, objective: callable):
        self.known_values = pd.DataFrame(columns=["X", "Y"])    # TODO: нужно чтобы были отсортированы по Y
                                                                #   - чтобы брать max = row[0], min = row[-1] ?
        self.squeeze_factor = 0.8
        self.intervals = pd.DataFrame(columns=["x0", "x1", "cost"])
        self.u_coords = []
        self.n_probes = 4
        self.max_intervals = 10
        self.objective = objective
        self.itr = 0


    def SelectIntervals(self):
        min_y = self.known_values["Y"].min()
        max_y = self.known_values["Y"].max()

        # TODO: добавить параметр - максимальное количество областей,
        #  которые можем просматривать (из которых можем собирать u_coords) ?

        # cleanup values from previous run
        self.intervals = self.intervals.drop(self.intervals.index)

        sum_cost = 0

        # Assuming known_values is sorted by X (increasing order)
        X = self.known_values["X"].to_numpy()
        Y = self.known_values["Y"].to_numpy()

        areas = self.CreateIntervalSet()
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
        self.intervals = self.intervals.sort_values(by="cost", ascending=False)



    def CreateIntervalSet(self):
        Y_sort = self.known_values.sort_values(by="Y", ascending=False)
        area_indexes = set()

        for row in Y_sort.iterrows():
            index = int(row[0])
            l_index = index - 1
            r_index = index + 1
            if l_index >= 0:
                area_indexes.add((l_index, index))

            if len(area_indexes) == self.max_intervals:
                break

            if r_index < Y_sort.shape[0]:
                area_indexes.add((index, r_index))

            if len(area_indexes) == self.max_intervals:
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
        num_probes = min(self.n_probes, len(self.u_coords))
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
            self.known_values = self.known_values._append({
                "X": x,
                "Y": y
            }, ignore_index=True)
        self.known_values = self.known_values.sort_values(by="X")
        self.known_values.reset_index(inplace=True, drop=True)
        pass



    def Warmup(self):
        # TODO
        pass



    def RunCycle(self):
        for i in range(10):
            if self.itr == 0:
                self.Warmup()
            else:
                self.SelectIntervals()
                self.UnitMapping()
                new_X = self.CreateProbePoints()
                self.RunValues(new_X)

            self.itr += 1


