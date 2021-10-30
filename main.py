import numpy as np
from Evaluation import EvalUtils, ConfusionMatrix
from ModelBuilders import DTree
from numpy.random import default_rng
from NumpyUtils import NpUtils

MAIN_DB_DIR = "wifi_db"


def from_main_db_dir(path):
    return f"{MAIN_DB_DIR}{path}"


def load_dataset(filepath):
    return np.loadtxt(filepath)


def print_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, rng=default_rng()):
    trees = EvalUtils.k_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    avg_conf_matr = ConfusionMatrix.construct_avg_confusion_matrix(
        trees, num_class_labels)
    print(avg_conf_matr)
    print("===========================================================")
    print("Accuracy vals: ", avg_conf_matr.accuracy())
    print("Recall vals: ", avg_conf_matr.recall())
    print("Precision vals: ", avg_conf_matr.precision())
    print("F1 vals: ", avg_conf_matr.f1_measure())
    print("===========================================================")


def pruning_with_nested_cross_validation(num_folds, dataset, trained_model_constructor, rng=default_rng()):
    def apply_validation_set(
        val_db, model): return DTree.prune(val_db, model, model)
    return EvalUtils.nested_k_cross_validation(num_folds, dataset, trained_model_constructor,
                                               apply_validation_set, rng)


def print_pruning_with_nested_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, rng=default_rng()):
    pruned_trees = pruning_with_nested_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    avg_conf_matr = ConfusionMatrix.construct_avg_confusion_matrix(
        pruned_trees, num_class_labels)
    print(avg_conf_matr)
    print("===========================================================")
    print("Pruned Accuracy vals: ",
          avg_conf_matr.accuracy())
    print("Pruned Recall vals: ", avg_conf_matr.recall())
    print("Pruned Precision vals: ",
          avg_conf_matr.precision())
    print("Pruned F1 vals: ", avg_conf_matr.f1_measure())
    print("===========================================================")


# tree, depth = decision_tree_learning(skewed_dataset)
# trained_tree, depth = decision_tree_learning(clean_dataset)
# print_cross_validation(clean_dataset)
def trained_model_constructor(
    training_set): return DTree.construct(training_set)


if __name__ == '__main__':
    clean_dataset = load_dataset(from_main_db_dir("/clean_dataset.txt"))
    noisy_dataset = load_dataset(from_main_db_dir("/noisy_dataset.txt"))
    skewed_dataset = load_dataset(from_main_db_dir("/skewed_dataset.txt"))
    # BAD COMBINATIONS: seed=666, num_folds=10
    seed = 666
    rng = default_rng(seed)
    num_folds = 10
    dataset_used = noisy_dataset
    num_class_labels = len(NpUtils.unique_col_values(dataset_used, -1))
    print_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, rng)
    print_pruning_with_nested_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, rng)
    # print(tree)
