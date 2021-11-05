import numpy as np


class NpUtils:
    @classmethod
    def row_count(cls, np_arr):
        """
        Returns the number of rows in a numpy array
        :param np_arr: numpy array
        :return: number of rows
        """
        return np_arr.transpose()[0].size

    @classmethod
    def unique_col_values(cls, np_arr, col_idx):
        """
        Returns a set containing the unique values in a column of a numpy array
        :param np_arr:
        :param col_idx:
        :return:
        """
        return set(np.unique(np_arr[:, col_idx]))
