import numpy as np
import pandas as pd


class CustomOptimizer:
    def __init__(self):
        self.known_values = pd.DataFrame(columns=["X", "Y"])    # TODO: нужно чтобы были отсортированы по Y
                                                                #   - чтобы брать max = row[0], min = row[-1] ?
        self.squeeze_factor = 0.8
        self.intervals = pd.DataFrame(columns=["x0", "x1", "cost"])
        self.u_coords = []


    def SelectIntervals(self):
        min_y = self.known_values["Y"].min()
        max_y = self.known_values["Y"].max()

        # cleanup values from previous run
        self.intervals = self.intervals.drop(self.intervals.index)

        sum_cost = 0
        X = self.known_values["X"].to_numpy()
        Y = self.known_values["Y"].to_numpy()
        for i in range(self.known_values.shape[0] - 1):
            dx = X[i + 1] - X[i]
            mid_point = (X[i + 1] + X[i]) / 2
            x0 = mid_point - dx * self.squeeze_factor / 2
            x1 = mid_point + dx * self.squeeze_factor / 2

            # TODO: сделать параметром eps (0.0001) -
            #  точность, при которой в наших масштабах
            #  мы считаем значения равными везде в оптимайзере?
            if max_y - min_y <= 0.00001:
                y1_cost = 1.0
                y2_cost = 1.0
            else:
                y1_cost = (Y[i + 1] - min_y) / (max_y - min_y) + 0.1
                y2_cost = (Y[i + 1] - min_y) / (max_y - min_y) + 0.1
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


    def RunCycle(self):
        pass