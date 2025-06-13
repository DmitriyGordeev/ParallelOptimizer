from CustomOptimizer import CustomOptimizer
import pandas as pd


class PlatoModule:
    def __init__(self, optimizer: CustomOptimizer):
        self.optimizer = optimizer
        self.plato_regions = []
        self.sum_shapes = 0
        self.region_ucoords = dict()
        self.u_gap = 0.01


    def GroupTables(self, plato_indexes: list):
        known_values = self.optimizer.known_values
        self.sum_shapes = 0.0
        for p in plato_indexes:
            l_index = p[0]
            r_index = p[1]

            # Select plato-region
            region = known_values.iloc[l_index: r_index + 2]

            # filter out "plato_block" rows
            non_blocked = region[region["plato_block"] == False]

            # saving original table index column to unmap values from it later
            non_blocked["original_index"] = non_blocked.index
            non_blocked.reset_index(inplace=True, drop=True)

            self.plato_regions.append(non_blocked)
            self.sum_shapes += non_blocked.shape[0]


    def UnitMapRegions(self) -> float:
        assert self.sum_shapes > 0

        # table index -> tuple(u_start, u_end)
        self.region_ucoords.clear()
        u_cursor = 0.0
        for i, t in enumerate(self.plato_regions):
            w = float(t.shape[0]) / self.sum_shapes
            self.region_ucoords[i] = (u_cursor, u_cursor + w)
            u_cursor += w + self.u_gap
        return u_cursor


    # Transforms unit-space value into region table's index (u -> i)
    @staticmethod
    def UnmapX(region: pd.DataFrame, u_pick: float, u_coords: tuple) -> float:
        alpha = (u_pick - u_coords[0]) / (u_coords[1] - u_coords[0])
        idx_range = region.index
        region_row = int(((idx_range.stop - 1) - idx_range.start) * alpha)

        if region_row < region.shape[0] - 1:
            X_l = region.iloc[region_row]["X"]
            X_r = region.iloc[region_row + 1]["X"]
            X_value = (X_l + X_r) / 2.0

        else:
            X_l = region.iloc[region_row - 1]["X"]
            X_r = region.iloc[region_row]["X"]
            X_value = (X_l + X_r) / 2.0

        return X_value




    def UnmapValues(self, u_cursor: float) -> list:
        num_probes = self.optimizer.n_probes
        assert num_probes > 0

        u_len = u_cursor - self.u_gap
        u_step = u_len / (num_probes + 1)
        out_X = set()
        u_pick = u_step

        for i in range(num_probes):
            hit_gap = False

            for k, u_coords in self.region_ucoords.items():
                if hit_gap:
                    if u_pick < u_coords[0]:
                        u_pick = u_coords[0] + 0.01 * (u_coords[1] - u_coords[0])

                if u_coords[0] <= u_pick <= u_coords[1]:
                    X_value = self.UnmapX(self.plato_regions[k], u_pick, u_coords)
                    out_X.add(X_value)
                    break

                else:
                    hit_gap = True

            u_pick += u_step
        return list(out_X)