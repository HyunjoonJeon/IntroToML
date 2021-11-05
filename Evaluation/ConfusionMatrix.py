import numpy as np
from NumpyUtils import NpUtils


class ConfusionMatrix:
    npy_rep = None

    def __init__(self, npy_rep):
        self.npy_rep = npy_rep

    def fn_measure(self, beta, as_latex=False):
        precisions = self.precision()
        recalls = self.recall()

        # just to make sure they are of the same length
        num_precisions = len(precisions)
        assert num_precisions == len(recalls)

        # Compute the per-class F measure
        f_measures = list()

        beta_sqr = beta ** 2

        for i in range(num_precisions):
            i_recall = recalls[i]
            i_precision = precisions[i]
            i_f_measure = (1 + beta_sqr) * i_precision * i_recall / \
                ((beta_sqr * i_precision) + i_recall)
            f_measures.append(i_f_measure)

        ret = np.array(f_measures)
        if as_latex:
            return NpUtils.nparray_to_latex_row(f"F{beta} measure", ret)

    def f1_measure(self, as_latex=False):
        return self.fn_measure(1, as_latex)

    def precision(self, as_latex=False):
        # Compute the precision per class
        p = list()
        for i in range(len(self.npy_rep)):
            i_col_sum = self.npy_rep[:, i].sum()
            i_col_tp = self.npy_rep[i][i]
            i_col_precision = i_col_tp / i_col_sum
            p.append(i_col_precision)
        ret = np.array(p)
        if as_latex:
            return NpUtils.nparray_to_latex_row("Precision", ret)
        return ret

    def recall(self, as_latex=False):
        # Compute the recall per class
        r = list()
        for i in range(len(self.npy_rep)):
            i_row_sum = self.npy_rep[i, :].sum()
            i_col_tp = self.npy_rep[i][i]
            i_col_recall = i_col_tp / i_row_sum
            r.append(i_col_recall)

        ret = np.array(r)
        if as_latex:
            return NpUtils.nparray_to_latex_row("Recall", ret)
        return ret

    def accuracy(self):
        elem_sum = np.sum(self.npy_rep)
        if elem_sum > 0:
            return float(np.trace(self.npy_rep) / elem_sum)
        else:
            return 0.

    @classmethod
    def construct(cls, y_gold, y_prediction, class_labels=None, forceSize=0):
        # if no class_labels are given, we obtain the set of unique class labels from
        # the union of the ground truth annotation and the prediction
        if not class_labels:
            class_labels = np.unique(
                np.concatenate((y_gold, y_prediction)))

        if forceSize > 0:
            num_labels = forceSize
        else:
            num_labels = len(class_labels)
        np_rep = np.zeros(
            (num_labels, num_labels), dtype=np.float64)

        # Create map from: class_label to number
        class_label_num_dict = dict()
        for (num, class_label) in enumerate(list(class_labels)):
            class_label_num_dict[class_label] = num

        for entry_idx in range(len(y_gold)):
            gold_label = y_gold[entry_idx]
            pred_label = y_prediction[entry_idx]
            gold_label_num = class_label_num_dict[gold_label]
            pred_label_num = class_label_num_dict[pred_label]
            np_rep[gold_label_num][pred_label_num] += 1

        return ConfusionMatrix(np_rep)

    @classmethod
    def construct_avg_confusion_matrix(cls, tree_test_dataset_pairs, size):
        total_confusion = np.zeros(
            (size, size), dtype=np.float64)
        for (tree, test_dataset) in tree_test_dataset_pairs:
            predicted_values = list()
            gold_values = list()
            for test in test_dataset:
                predicted_values.append(tree.predict(test))
                gold_values.append(test[-1])

            confusion = ConfusionMatrix.construct(
                gold_values, predicted_values, forceSize=size).npy_rep

            total_confusion += confusion
        total_confusion /= len(tree_test_dataset_pairs)
        # returns average confusion matrix
        return ConfusionMatrix(total_confusion)

    def get_npy_rep(self, as_latex=False):
        if as_latex:
            return NpUtils.nparray_to_latex(self.npy_rep)
        return self.npy_rep

    def __repr__(self):
        return self.npy_rep.__repr__()
