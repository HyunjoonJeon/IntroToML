import numpy as np


class NpUtils:
    @classmethod
    def row_count(cls, np_arr):
        return np_arr.transpose()[0].size

    @classmethod
    def unique_col_values(cls, np_arr, col_idx):
        return set(np.unique(np_arr[:, col_idx]))
