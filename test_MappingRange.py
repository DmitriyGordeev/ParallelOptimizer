from unittest import TestCase
import numpy as np
import pandas as pd

def func(x: float):
    # return 2.0 * x + 3.2
    return 0.0


class TestMappingRange(TestCase):
    def test_CostFunction(self):

        interval_squeeze = 0.8
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
            x0 = mid_point - dx * interval_squeeze / 2
            x1 = mid_point + dx * interval_squeeze / 2

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


        # Unmapping picked values
        # TODO: если попал в u_gap
        # TODO: покрыть юнит-тестами этот выбор
        u_pick = 0.71
        u_potential_gap_hit = False
        X_unmapped = 0.0
        for i, u_pair in enumerate(u_coords):

            if u_potential_gap_hit:
                if u_pick < u_pair[0]:
                    u_pick = u_pair[0] + 0.01 * (u_pair[1] - u_pair[0])

            if u_pair[0] <= u_pick <= u_pair[1]:
                alpha = (u_pick - u_pair[0]) / (u_pair[1] - u_pair[0])
                x0 = intervals.iloc[i]["x0"]
                x1 = intervals.iloc[i]["x1"]
                X_unmapped = x0 + (x1 - x0) * alpha
                break

            else:
                u_potential_gap_hit = True

        print(f"X_unmapped = {X_unmapped}")



    def test_Drop(self):
        df = pd.DataFrame(columns=["Cost", "Value"])
        df["Cost"] = [1, 2, 1, 3, 1, 2]
        df["Value"] = [17, 4, 5, 7, 4, 21]
        df = df.sample(frac=1)
        df = df.sort_values(by="Cost", kind="stable")
        pass



    def test_table_reindex(self):
        data = pd.DataFrame(columns=["X", "Y", "plato_block"])
        data["X"] = np.arange(0, 10) * 0.1
        data["Y"] = 2 * np.arange(0, 10) * 0.1 + 0.98
        data["plato_block"] = [False, False, True, False, False, False, True, True, False, False]

        non_blocked = data[data["plato_block"] == False]
        non_blocked["original_index"] = non_blocked.index
        non_blocked.reset_index(inplace=True, drop=True)

        tables = [non_blocked]

        table_dict = dict()
        table_dict[0] = (0.0, 0.26)

        u_pick = 0.17
        for k, v in table_dict.items():
            if v[0] <= u_pick <= v[1]:
                lerp_coeff = (u_pick - v[0]) / (v[1] - v[0])
                tgt_table = tables[k]
                idx_range = tgt_table.index
                row_index = int((idx_range.stop - idx_range.start) * lerp_coeff)
                unmapped_index = tgt_table.iloc[row_index]["original_index"]
                print(f"unmapped_index = {unmapped_index} => X = {data.iloc[unmapped_index]["X"]}")
                break



