import numpy as np


def information_entropy(dataset):
    labels = dataset.transpose()[0]

    _, unique_label_counts = np.unique(labels, return_counts=True)
    total_entries = labels.size

    ret = 0
    for label_cnt in unique_label_counts:
        px = label_cnt/total_entries
        ret -= px * np.log2(px)
    return ret


def information_gain(dataset):
    h_dataset = information_entropy(dataset)
    dataset_cnt = h_dataset.size
    single_data_att_cnt = h_dataset[0].size
    s = 0
    for subset in np.vsplit(dataset, [single_data_att_cnt]):
        s = (subset.size/dataset_cnt) * information_entropy(subset)
    return h_dataset - s


def remainder(s_left, s_right):
    """PRE: "s_left" and "s_right" have labels on their last column.
    """
    def no_labels_size(s): return s[:, 0].size

    left_size = no_labels_size(s_left)
    right_size = no_labels_size(s_right)

    total = left_size + right_size
    return (left_size/total) * information_entropy(s_left) + (right_size/total) * information_entropy(s_right)


def find_split(dataset):
    for i in range(dataset[0].size-1):
        subset = dataset[np.argsort(dataset[:, i], axis=0)][:, (i, -1)]
        left, right = np.vsplit(subset, 2)

        remainder(left, right)


def decision_tree_learning(training_dataset, depth):
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
