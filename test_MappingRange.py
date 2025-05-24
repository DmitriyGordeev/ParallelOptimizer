from unittest import TestCase
import numpy as np
import pandas as pd

def func(x: float):
    return 2.0 * x + 3.2


class TestMappingRange(TestCase):
    def test_CostFunction(self):

        gap_len = 0.01
        interval_squezze = 0.8
        X = list()
        Y = list()

        X.append(0)
        Y.append(func(0))

        X.append(50)
        Y.append(func(50))

        X.append(100)
        Y.append(func(100))

        func_MAX = max(Y)
        func_MIN = min(Y)

        # Selecting intervals
        # interval item : (tuple) -> cost : float
        intervals = pd.DataFrame(columns=["x0", "x1", "cost"])
        sum_cost = 0
        for i in range(len(X) - 1):
            dx = X[i + 1] - X[i]
            mid_point = (X[i + 1] + X[i]) / 2
            x0 = mid_point - dx * 0.8 / 2
            x1 = mid_point + dx * 0.8 / 2

            # TODO: сделать параметром eps (0.0001) -
            #  точность, при которой в наших масштабах
            #  мы считаем значения равными везде в оптимайзере?
            if func_MAX - func_MIN <= 0.00001:
                y1_cost = 1.0
                y2_cost = 1.0
            else:
                y1_cost = (Y[i + 1] - func_MIN) / (func_MAX - func_MIN) + 0.1
                y2_cost = (Y[i + 1] - func_MIN) / (func_MAX - func_MIN) + 0.1
            sum_cost += y1_cost + y2_cost

            intervals = intervals._append({
                "x0": x0,
                "x1": x1,
                "cost": y1_cost + y2_cost
            }, ignore_index=True)

        pass

        # TODO: плато детектор - тоже через eps?


        # renormalize weights
        intervals["cost"] = intervals["cost"] / sum_cost

        assert abs(intervals["cost"].sum() - 1.0) <= 0.00001        # не eps!

        intervals = intervals.sort_values(by="cost", ascending=False)

        # # Placing on unit-len:
        u_coords = []
        u_cursor = 0.0
        u_gap = 0.01
        for row in intervals.iterrows():
            item = row[1]
            w = item["cost"]
            u_coords.append((u_cursor, u_cursor + w))
            u_cursor += w + u_gap
            pass


        # # Unmapping picked values
        # picked_value = 0.634
        #
        # index = 0
        # for i, u_pair in enumerate(u_coords):
        #     if u_pair[0] <= picked_value <= u_pair[1]:
        #         index = i
        #         break
        #
        # pass








