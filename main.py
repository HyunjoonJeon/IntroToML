import numpy as np
from numpy.random import default_rng
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
            max_left, min_right = left.max(axis=0)[i], right.min(axis=0)[i]
            split_value = (max_left + min_right) / 2
            left_ret, right_ret = left, right

    return max_attr_idx, split_value, left_ret, right_ret


def decision_tree_learning(training_dataset, depth=0):
    labels = training_dataset[:, -1]
    if np.unique(labels).size == 1:
        return DTree.LeafNode(labels[0], depth), depth
    split_idx, split_value, l_dataset, r_dataset = find_split(training_dataset)

    node = DTree.Node(split_idx, split_value, depth)
    l_branch, l_depth = decision_tree_learning(l_dataset, depth+1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth+1)
    node.l_tree, node.r_tree = l_branch, r_branch
    return node, max(l_depth, r_depth)


# Step 3 - Evaluation

# From tutorial
def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    if random_generator is not None:
        shuffled_indices = random_generator.permutation(n_instances)
    else:
        shuffled_indices = range(0, n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.

    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits (n folds)
    # these are "row" indices
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        test_indices = np.array(split_indices[k])
        # Concatenate all other folds
        train_indices = np.concatenate(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds

# End from tutorial.


def compute_fn_measure(conf_matrix, beta):
    precisions = precision(conf_matrix)
    recalls = recall(conf_matrix)

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

    return np.array(f_measures)


def compute_f1_measure(conf_matrix):
    return compute_fn_measure(conf_matrix, 1)


def precision(conf_matrix):
    # Compute the precision per class
    p = list()
    for i in range(len(conf_matrix)):
        i_col_sum = conf_matrix[:, i].sum()
        i_col_tp = conf_matrix[i][i]
        i_col_precision = i_col_tp / i_col_sum
        p.append(i_col_precision)

    return np.array(p)


def recall(conf_matrix):
    # Compute the recall per class
    r = list()
    for i in range(len(conf_matrix)):
        i_row_sum = conf_matrix[i, :].sum()
        i_col_tp = conf_matrix[i][i]
        i_col_recall = i_col_tp / i_row_sum
        r.append(i_col_recall)

    return np.array(r)


def accuracy_from_confusion(conf_matrix):
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes. 
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """
    elem_sum = np.sum(conf_matrix)
    if elem_sum > 0:
        return np.trace(conf_matrix) / elem_sum
    else:
        return 0.


def confusion_matrix(y_gold, y_prediction, class_labels=None, forceSize=0):
    """ Compute the confusion matrix.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    if forceSize > 0:
        num_labels = forceSize
    else:
        num_labels = len(class_labels)
    confusion = np.zeros((num_labels, num_labels), dtype=np.float64)

    # Create map from: class_label to number
    class_label_num_dict = dict()
    for (num, class_label) in enumerate(list(class_labels)):
        class_label_num_dict[class_label] = num

    for entry_idx in range(len(y_gold)):
        gold_label = y_gold[entry_idx]
        pred_label = y_prediction[entry_idx]
        gold_label_num = class_label_num_dict[gold_label]
        pred_label_num = class_label_num_dict[pred_label]
        confusion[gold_label_num][pred_label_num] += 1

    return confusion


def construct_avg_confusion_matrix(tree_test_dataset_pairs):
    NUM_LABELS = 4
    total_confusion = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.float64)
    for (tree, test_dataset) in tree_test_dataset_pairs:
        predicted_values = list()
        gold_values = list()
        for test in test_dataset:
            predicted_values.append(tree.predict(test))
            gold_values.append(test[-1])

        confusion = confusion_matrix(
            gold_values, predicted_values, forceSize=NUM_LABELS)

        total_confusion += confusion
    total_confusion /= len(tree_test_dataset_pairs)
    # returns average confusion matrix
    return total_confusion


def evaluate(test_db, trained_tree, init_counts=False):
    """Returns the accuracy of the tree
    """
    if init_counts:
        NUM_LABELS = 4
        trained_tree.init_counts(NUM_LABELS)

    correct_predictions = 0
    total = test_db.transpose()[0].size
    for i in range(total):
        test_row = test_db[i]
        predicted_label = trained_tree.predict(test_row)
        correct_label = test_row[-1]
        is_correct_prediction = predicted_label == correct_label
        if is_correct_prediction:
            correct_predictions += 1
    return float(correct_predictions / total)


def k_cross_validation(k, dataset, random_generator=default_rng()):
    # number of rows
    n_instances = dataset.transpose()[0].size

    final_models = list()
    for (train_indices, test_indices) in train_test_k_fold(k, n_instances, random_generator):
        training_dataset = dataset[train_indices, :]
        test_dataset = dataset[test_indices, :]
        trained_tree, depth = decision_tree_learning(training_dataset)
        final_models.append([trained_tree, test_dataset])
    return final_models


def nested_k_cross_validation(k, dataset, random_generator=default_rng()):
    # number of rows
    n_instances = dataset.transpose()[0].size
    n_of_trees = 0
    final_models = list()
    for (train_indices, test_indices) in train_test_k_fold(k, n_instances, random_generator):
        training_and_val_dataset = dataset[train_indices, :]
        test_dataset = dataset[test_indices, :]
        for (train_indices, val_indices) in train_test_k_fold(k-1, train_indices.size, random_generator=None):
            # Train new tree, but don't shuffle
            training_dataset = training_and_val_dataset[train_indices, :]
            validation_dataset = training_and_val_dataset[val_indices, :]
            trained_tree, depth = decision_tree_learning(training_dataset)
            prune(validation_dataset, trained_tree, trained_tree)
            final_models.append([trained_tree, test_dataset])
    return final_models


def print_cross_validation(dataset):
    best_trees = k_cross_validation(10, dataset)
    avg_conf_matr = construct_avg_confusion_matrix(best_trees)
    print(avg_conf_matr)
    print("===========================================================")
    print("Accuracy vals: ", accuracy_from_confusion(avg_conf_matr))
    print("Recall vals: ", recall(avg_conf_matr))
    print("Precision vals: ", precision(avg_conf_matr))
    print("F1 vals: ", compute_f1_measure(avg_conf_matr))
    print("===========================================================")


def print_nested_cross_validation(dataset):
    best_trees = nested_k_cross_validation(10, dataset)
    avg_conf_matr = construct_avg_confusion_matrix(best_trees)
    print(avg_conf_matr)
    print("===========================================================")
    print("Pruned Accuracy vals: ", accuracy_from_confusion(avg_conf_matr))
    print("Pruned Recall vals: ", recall(avg_conf_matr))
    print("Pruned Precision vals: ", precision(avg_conf_matr))
    print("Pruned F1 vals: ", compute_f1_measure(avg_conf_matr))
    print("===========================================================")


def prune(val_db, tree, root_tree):
    if tree.is_only_leaf_parent():
        # prune
        # convert into a leaf whose value is the majority class label
        before_prune_accuracy = evaluate(
            val_db, root_tree, init_counts=True)
        attr, val, l_tree, r_tree, counts = tree.convert_to_leaf()
        after_prune_accuracy = evaluate(
            val_db, root_tree, init_counts=True)

        # Evaluate the resulting “pruned” tree using the “validation set”; prune if accuracy is higher than unpruned
        # "pruned_tree" is side-effected as "pruned"
        # print(f"Compare: after: {after_prune_accuracy} before: {before_prune_accuracy} | counts: {tree.counts}")
        if after_prune_accuracy <= before_prune_accuracy:
            # worse tree, revert back
            tree.convert_back(attr, val, l_tree, r_tree, counts)
            return False
        # better tree, keep it

        return True

    l_tree_pruned = tree.l_tree and prune(val_db, tree.l_tree, root_tree)
    r_tree_pruned = tree.r_tree and prune(val_db, tree.r_tree, root_tree)

    return l_tree_pruned and r_tree_pruned and prune(val_db, tree, root_tree)

# tree, depth = decision_tree_learning(skewed_dataset)
# trained_tree, depth = decision_tree_learning(clean_dataset)


# print_cross_validation(clean_dataset)
print_cross_validation(noisy_dataset)
print_nested_cross_validation(noisy_dataset)
# print(tree)
