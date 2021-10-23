import numpy as np
from tree import DTree


def load_dataset(filepath):
    return np.loadtxt(filepath)


clean_dataset = load_dataset("wifi_db/clean_dataset.txt")
noisy_dataset = load_dataset("wifi_db/noisy_dataset.txt")
skewed_dataset = load_dataset("wifi_db/skewed_dataset.txt")


def information_entropy(dataset):
    labels = dataset.transpose()[-1]

    _, unique_label_counts = np.unique(labels, return_counts=True)
    total_entries = labels.size
    ret = 0
    for label_cnt in unique_label_counts:
        px = label_cnt/total_entries
        ret -= px * np.log2(px)
    return ret


info_ent = information_entropy(skewed_dataset)
assert info_ent <= 1, f"Expected 1, Actual: {info_ent}"


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


def get_rows(arr):
    return arr.transpose()[0].size


def find_split(dataset):
    max_attr_gain = float("-inf")
    max_attr_idx = None
    split_value = None
    left_ret, right_ret = None, None
    for i in range(dataset[0].size-1):
        # Sort array 'a' by column 'i' == a[np.argsort(a[:, i], axis=0)]
        argsorted_subset = dataset[np.argsort(dataset[:, i], axis=0)]
        sorted_subset = argsorted_subset[:, :]
        # Not sure if below is needed
        # subset_for_find_split = argsorted_subset[:, (i, -1)]

        left_right_splits = np.array_split(sorted_subset, 2, axis=0)
        left, right = left_right_splits[0], left_right_splits[1]
        attr_i_gain = information_gain(dataset, left, right)

        if attr_i_gain > max_attr_gain:
            max_attr_gain = attr_i_gain
            max_attr_idx = i
            max_left, max_right = left.max(axis=0)[0], right.min(axis=0)[0]
            split_value = (max_left + max_right) / 2
            left_ret, right_ret = left, right

    return max_attr_idx, split_value, left_ret, right_ret


def decision_tree_learning(training_dataset, depth=0):
    labels = training_dataset[:, -1]
    if np.unique(labels).size == 1:
        return DTree.LeafNode(labels[0]), depth
    split_idx, split_value, l_dataset, r_dataset = find_split(training_dataset)

    node = DTree.Node(split_idx, split_value)
    l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
    node.l_tree, node.r_tree = l_branch, r_branch
    return node, max(l_depth, r_depth)


# print(decision_tree_learning(clean_dataset[495:506]))
print(decision_tree_learning(skewed_dataset))
