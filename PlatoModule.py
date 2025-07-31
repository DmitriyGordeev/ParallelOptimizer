from ParallelOptimizer import ParallelOptimizer
import numpy as np
import pandas as pd


class PlatoModule:
    def __init__(self, optimizer: ParallelOptimizer):
        self.optimizer = optimizer
        self.plato_indexes = []
        self.plato_regions = []
        self.sum_shapes = 0
        self.region_ucoords = dict()
        self.u_gap = 0.01
        self.plato_x_eps = 0.1
        self.plato_y_eps = 0.1
        self.plato_x_block_eps = 0.001


    def FindPlatoRegions(self):
        # reset plato before new recalculation
        self.optimizer.known_values.loc[:, "plato_index"] = -1
        self.optimizer.known_values.loc[:, "plato_edge"] = False
        self.optimizer.known_values.loc[:, 'plato_block'] = False

        # seek plato indexes
        l_index = -1
        r_index = -1
        platos = []
        obj_column = self.optimizer.objective_column

        values = self.optimizer.known_values
        for i in range(values.shape[0] - 1):
            x_vector = values.iloc[i][0 : len(self.optimizer.mins)].to_numpy()
            y_objective = values.iloc[i][obj_column]

            x_next = values.iloc[i + 1][0 : len(self.optimizer.mins)].to_numpy()
            y_next = values.iloc[i + 1][obj_column]

            # distance
            distance = np.linalg.norm(x_vector - x_next)
            if distance < self.plato_x_eps and abs(y_next - y_objective) < self.plato_y_eps:
                if l_index < 0:
                    l_index = i
                r_index = i

                if distance < self.plato_x_block_eps:
                    values.at[i - 1, 'plato_block'] = True

            else:
                if l_index != -1:
                    platos.append([l_index, r_index])
                    l_index = -1
                    r_index = -1

        if l_index != -1:
            platos.append([l_index, r_index])

        self.plato_indexes = platos



    def MarkPlatoRegions(self):
        if len(self.plato_indexes) == 0:
            return

        for j, index_pair in enumerate(self.plato_indexes):
            iL = index_pair[0]
            iR = index_pair[1]

            if iL == iR:
                continue

            self.optimizer.known_values.loc[iL : iR + 1, "plato_index"] = j
            self.optimizer.known_values.loc[iL, "plato_edge"] = True
            self.optimizer.known_values.loc[iR + 1, "plato_edge"] = True



    def GroupTables(self):
        known_values = self.optimizer.known_values
        self.sum_shapes = 0.0
        for p in self.plato_indexes:
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


    def UnitMapRegions(self):
        assert self.sum_shapes > 0

        # table index -> tuple(u_start, u_end)
        self.region_ucoords.clear()
        u_cursor = 0.0
        for i, t in enumerate(self.plato_regions):
            w = float(t.shape[0]) / self.sum_shapes
            self.region_ucoords[i] = (u_cursor, u_cursor + w)
            u_cursor += w + self.u_gap


    # Transforms unit-space value into region table's index (u -> i)
    def UnmapX(self, region: pd.DataFrame, u_pick: float, u_coords: tuple) -> np.array:
        alpha = (u_pick - u_coords[0]) / (u_coords[1] - u_coords[0])
        idx_range = region.index
        dims = len(self.optimizer.mins)
        region_row = int(((idx_range.stop - 1) - idx_range.start) * alpha)

        if region_row < region.shape[0] - 1:
            X_l = region.iloc[region_row][0 : dims].to_numpy()
            X_r = region.iloc[region_row + 1][0 : dims].to_numpy()
            X_value = (X_l + X_r) / 2.0

        else:
            X_l = region.iloc[region_row - 1][0: dims].to_numpy()
            X_r = region.iloc[region_row][0: dims].to_numpy()
            X_value = (X_l + X_r) / 2.0
        return X_value


    def UnmapValues(self) -> np.array:
        num_probes = self.optimizer.n_probes
        assert num_probes > 0

        u_cursor = list(self.region_ucoords.values())[-1][1]
        u_len = u_cursor - self.u_gap
        u_step = u_len / (num_probes + 1)
        out_X = []
        u_pick = u_step

        for i in range(num_probes):
            hit_gap = False

            for k, u_coords in self.region_ucoords.items():
                if hit_gap:
                    if u_pick < u_coords[0]:
                        u_pick = u_coords[0] + 0.01 * (u_coords[1] - u_coords[0])

                if u_coords[0] <= u_pick <= u_coords[1]:
                    X_value = self.UnmapX(self.plato_regions[k], u_pick, u_coords)
                    out_X.append(X_value)
                    break

                else:
                    hit_gap = True

            u_pick += u_step

        out_X = np.array(out_X)
        out_X = np.transpose(out_X)     # rows = axes, columns = points
        out_X = np.unique(out_X.astype(float), axis=1)
        return out_X


    def GeneratePlatoPoints(self) -> np.array:
        self.GroupTables()
        self.UnitMapRegions()
        return self.UnmapValues()