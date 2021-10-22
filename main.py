import numpy as np
from tree import DTree


def information_entropy(dataset):
    labels = dataset.transpose()[0]

    _, unique_label_counts = np.unique(labels, return_counts=True)
    total_entries = labels.size

    ret = 0
    for label_cnt in unique_label_counts:
        px = label_cnt/total_entries
        ret -= px * np.log2(px)
    return ret


def remainder(s_left, s_right):
    """PRE: "s_left" and "s_right" have labels on their last column.
    """
    def no_labels_size(s): return s[:, 0].size

    left_size = no_labels_size(s_left)
    right_size = no_labels_size(s_right)

    h_left = information_entropy(s_left)
    h_right = information_entropy(s_right)

    total = left_size + right_size
    return (left_size/total) * h_left + (right_size/total) * h_right


def information_gain(dataset, left, right):
    h_dataset = information_entropy(dataset)
    return h_dataset - remainder(left, right)


def find_split(dataset):
    max_attr_gain = float("-inf")
    max_attr_idx = None
    for i in range(dataset[0].size-1):
        subset = dataset[np.argsort(dataset[:, i], axis=0)][:, (i, -1)]
        left, right = np.vsplit(subset, 2)
        attr_i_gain = information_gain(dataset, left, right)
        if attr_i_gain > max_attr_gain:
            max_attr_gain = attr_i_gain
            max_attr_idx = i
    return max_attr_idx


def decision_tree_learning(training_dataset, depth):

    labels = training_dataset[:, -1]

    if len(set(training_dataset)) == 1:
        return

    # 2: if all samples have the same label then
    # 3:    return (a leaf node with this value, depth)
    # 4: else
    # 5:    split← find split(training dataset)
    # 6:    node← a new decision tree with root as split value
    # 7:    l branch, l depth ← DECISION TREE LEARNING(l dataset, depth+1)
    # 8:    r branch, r depth ← DECISION TREE LEARNING(r dataset, depth+1)
    # 9:    return (node, max(l depth, r depth))
    # 10: end if
    # 11: end procedure
    pass


def load_dataset(filepath):
    return np.loadtxt(filepath)


clean_dataset = load_dataset("wifi_db/clean_dataset.txt")
noisy_dataset = load_dataset("wifi_db/noisy_dataset.txt")
print(find_split(clean_dataset))
print(clean_dataset.size)
