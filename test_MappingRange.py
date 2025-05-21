from unittest import TestCase
import numpy as np

def func(x: float):
    return x * 2.3 - 0.23


class TestMappingRange(TestCase):
    def test_CostFunction(self):
        # x = np.arange(0, 100, 0.1)
        # y = x * 2.3 - 0.23

        yA = func(0)
        yB = func(100)
        yC = func(50)

        gap_len = 0.01
        max_SF = max(yA, yB, yC)

        intervals = [(12.3, 48.9),
                     (89.3, 96.4)]

        weights = []
        for item in intervals:


        # # Slice 1
        # x1 = 34.3
        # x2 = 67.8
        # y1 = func(x1)
        # y2 = func(x2)
        #
        # # Slice 2:
        # x3 = 89.3
        # x4 = 96.3
        # y3 = func(x3)
        # y4 = func(x4)
        #
        # # Costs:
        # cost1 = 2 * max_SF - y1 - y2
        # cost2 = 2 * max_SF - y3 - y4
        #
        # sum_Cost = cost1 + cost2


        # w1 = cost1 / sum_Cost
        # w2 = cost2 / sum_Cost

        weights = list(reversed(sorted([w1, w2])))
        S_renorm = sum(weights) + (2 - 1) * gap_len

        for i in range(len(weights)):
            weights[i] = weights[i] / S_renorm

        print(f"w1 = {w1}, w2 = {w2}")

        # Placing on unit-len:
        u_coords = []
        cursor = 0
        for w in weights:
            u_coords.append((cursor, cursor + w))
            cursor += w + gap_len

        # Unmapping picked values
        picked_value = 0.634

        index = 0
        for i, u_pair in enumerate(u_coords):
            if u_pair[0] <= picked_value <= u_pair[1]:
                index = i
                break

        pass








