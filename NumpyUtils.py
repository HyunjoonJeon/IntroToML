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

    @classmethod
    def nparray_to_latex(cls, nparr):
        # Source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix/17131750
        """Returns a LaTeX bmatrix

        :nparr: numpy array
        :returns: LaTeX bmatrix as a string
        """
        if len(nparr.shape) > 2:
            raise ValueError(
                'nparray_to_latex can at most display two dimensions')
        lines = str(nparr).replace('[', '').replace(']', '').splitlines()
        rv = [r'\begin{bmatrix}']
        rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
        rv += [r'\end{bmatrix}']
        return '\n'.join(rv)

    @classmethod
    def nparray_to_latex_row(cls, label, nparr):
        """Returns a LaTeX row

        :label: row label
        :nparr: numpy array
        :returns: LaTeX row as a string
        """
        entries = [label]
        entries += str(nparr).replace('[', '').replace(']', '').split(" ")
        entries = list(filter(lambda entry: len(entry) > 0, entries))
        return " & ".join(entries)
