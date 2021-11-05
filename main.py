from matplotlib.pyplot import show
import numpy as np
from Evaluation import EvalUtils, ConfusionMatrix
from ModelBuilders import DTree, DecisionTree
from numpy.random import default_rng
from NumpyUtils import NpUtils
import sys
import os

from Utils import Utils


def load_dataset(filepath):
    """
    Returns numpy array with dataset loaded from filepath.
    :param filepath: path to dataset file
    """
    return np.loadtxt(filepath)


def print_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, visualise_cnt, show_latex, rng=default_rng()):
    tree_testdataset_pairs = EvalUtils.k_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    trees = Utils.project_fst_from_pairs(tree_testdataset_pairs)
    print(len(trees))
    avg_conf_matr = ConfusionMatrix.construct_avg_confusion_matrix(
        tree_testdataset_pairs, num_class_labels)

    boundary_name = f"{num_folds}-cross validation"
    boundary = wrap_as_boundary(boundary_name)

    print(boundary)
    print(wrap_as_boundary("Average confusion matrix"),
          avg_conf_matr.get_npy_rep(show_latex))
    print(wrap_as_boundary("Accuracy vals"), avg_conf_matr.accuracy())
    print(wrap_as_boundary("Recall vals"), avg_conf_matr.recall(show_latex))
    print(wrap_as_boundary("Precision vals"),
          avg_conf_matr.precision(show_latex))
    print(wrap_as_boundary("F1 vals"), avg_conf_matr.f1_measure(show_latex))
    visualise_trees(visualise_cnt, [trees, boundary_name])
    print(boundary)


def pruning_with_nested_cross_validation(num_folds, dataset, trained_model_constructor, rng=default_rng()):
    """
    Returns (k * k-1) pruned trees using nested k-cross validation on the dataset inputted.
    :param num_folds: number of folds to use in nested cross validation
    :param dataset: numpy array with dataset
    :param trained_model_constructor: function that returns a trained model
    :param rng: random number generator
    """
    def apply_validation_set(
            val_db, model):
        DTree.prune(val_db, model)
    return EvalUtils.nested_k_cross_validation(num_folds, dataset, trained_model_constructor,
                                               apply_validation_set, rng)


def print_pruning_with_nested_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, visualise_cnt, show_latex, rng=default_rng()):
    """
    Prints all average metrics of (k * k-1) pruned trees using nested k-cross validation on the dataset inputted.
    :param dataset: numpy array with dataset
    :param trained_model_constructor: function that returns a trained model
    :param num_folds: number of folds to use in nested cross validation
    :param num_class_labels: number of class labels
    :param visualise_cnt: number of trees to visualise
    :param show_latex: shows the matrices as latex
    :param rng: random number generator
    """
    pruned_trees, unpruned_trees = pruning_with_nested_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    print_comparison_stats(f"Pruning nested {num_folds}-cross validation", pruned_trees, unpruned_trees,
                           num_class_labels, visualise_cnt, show_latex)


def wrap_as_boundary(value):
    return f"<!--- {value} --->"


def print_comparison_separator(label, name):
    print(wrap_as_boundary(f"{label} {name}"))


def print_comparison_stats(name, tree_testdataset_pairs_1, tree_testdataset_pairs_2, num_class_labels,
                           visualise_cnt, show_latex):
    name_1 = f"With {name}"
    name_2 = f"Without {name}"

    def wrap_print_comparison(label, value_1, value_2):
        print_comparison_separator(label, name_1)
        print(value_1)
        print_comparison_separator(label, name_2)
        print(value_2)

    print(wrap_as_boundary(f"Start {name}"))
    avg_conf_matr_1 = ConfusionMatrix.construct_avg_confusion_matrix(
        tree_testdataset_pairs_1, num_class_labels)
    avg_conf_matr_2 = ConfusionMatrix.construct_avg_confusion_matrix(
        tree_testdataset_pairs_2, num_class_labels)
    wrap_print_comparison("Average confusion matrix",
                          avg_conf_matr_1.get_npy_rep(show_latex), avg_conf_matr_2.get_npy_rep(show_latex))

    wrap_print_comparison("Accuracy vals",
                          avg_conf_matr_1.accuracy(), avg_conf_matr_2.accuracy())
    wrap_print_comparison("Recall vals",
                          avg_conf_matr_1.recall(show_latex), avg_conf_matr_2.recall(show_latex))
    wrap_print_comparison("Precision vals",
                          avg_conf_matr_1.precision(show_latex), avg_conf_matr_2.precision(show_latex))
    wrap_print_comparison("F1 vals",
                          avg_conf_matr_1.f1_measure(show_latex), avg_conf_matr_2.f1_measure(show_latex))

    def compute_tree_depth_avg(ts):
        return float(sum(list(map(lambda t: t.get_depth(), ts)))/len(ts))

    trees_1 = Utils.project_fst_from_pairs(tree_testdataset_pairs_1)
    trees_2 = Utils.project_fst_from_pairs(tree_testdataset_pairs_2)
    wrap_print_comparison("Average maximum depth",
                          compute_tree_depth_avg(trees_1), compute_tree_depth_avg(trees_2))
    visualise_trees(visualise_cnt, [trees_1, name_1], [trees_2, name_2])
    print(wrap_as_boundary(f"End {name}."))


def visualise_trees(count, trees1_name1_pair, trees2_name2_pair=None):
    trees_1 = trees1_name1_pair[0]
    name_1 = trees1_name1_pair[1]

    trees_2 = trees2_name2_pair and trees2_name2_pair[0]
    name_2 = trees_2 and trees2_name2_pair[1]

    show_trees_2 = trees_2
    for i in range(0, min(count, len(trees_1))):
        visualise_label = f"Visualise tree #{i+1}"
        print_comparison_separator(visualise_label, name_1)
        trees_1[i].visualise(not show_trees_2)
        if show_trees_2:
            print_comparison_separator(visualise_label, name_2)
            trees_2[i].visualise(show=True)


def trained_model_constructor(training_set):
    """
    Returns a trained decision tree model.
    :param training_set: numpy array with training set
    """
    return DTree.construct(training_set)


def run(filepath, num_folds, seed, visualise_cnt, show_latex, prune):
    """
    Runs the program.
    :param filepath: path to dataset file
    :param num_folds: number of folds to use in cross validation
    :param seed: seed for random number generator
    :param visualise_cnt: number of trees to visualise
    :param show_latex: shows the matrices as latex
    :param prune: enables the nested k-cross prune vs. unpruned comparison
    """
    dataset_used = load_dataset(filepath)
    rng = default_rng(seed)
    num_class_labels = len(NpUtils.unique_col_values(dataset_used, -1))
    if prune:
        print_pruning_with_nested_cross_validation(
            dataset_used, trained_model_constructor, num_folds, num_class_labels, visualise_cnt, show_latex, rng)
    else:
        print_cross_validation(dataset_used, trained_model_constructor,
                               num_folds, num_class_labels, visualise_cnt, show_latex, rng)


DB_CLI_ARG_NAME = "db"
FOLDS_CLI_ARG_NAME = "folds"
SEED_CLI_ARG_NAME = "seed"
VISUALISE_CLI_ARG_NAME = "visualise_cnt"
SHOW_LATEX_MATRICES_CLI_ARG_NAME = "latex"
PRUNE_CLI_ARG_NAME = "prune"
BIND_CLI_ARG_VALUE_CHAR = '='


def print_invalid_cli_call():
    """
    Prints invalid CLI call message.
    """
    print(
        f"Invalid cli call, argument format: [{PRUNE_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{SHOW_LATEX_MATRICES_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{VISUALISE_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{DB_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{FOLDS_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{SEED_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>]")


def print_bad_arg(argName):
    """
    Prints invalid argument message.
    """
    print(f"Bad argument given: {argName}")


def visualise_trained_tree(path):
    # Only used for CW task
    dataset = load_dataset(path)
    trained_model_constructor(dataset).visualise(show=True)


if __name__ == '__main__':
    num_cli_args = len(sys.argv)
    is_cli_call = num_cli_args > 1

    num_folds = 10
    seed = 666
    db_path = "wifi_db/clean_dataset.txt"
    # visualise_trained_tree(db_path)
    visualise_cnt = 0
    show_latex = False
    prune = False
    if is_cli_call:
        for arg in sys.argv[1:]:
            # split on '='
            cli_arg_value_pair = arg.split(BIND_CLI_ARG_VALUE_CHAR)
            if len(cli_arg_value_pair) == 2:
                cli_arg = cli_arg_value_pair[0]
                cli_arg_value = cli_arg_value_pair[1]
                cli_arg_value_is_numeric = cli_arg_value.isnumeric()
                if cli_arg == SEED_CLI_ARG_NAME:
                    if cli_arg_value_is_numeric:
                        seed = int(cli_arg_value)
                    else:
                        print_bad_arg(SEED_CLI_ARG_NAME)
                        exit()
                elif cli_arg == FOLDS_CLI_ARG_NAME:
                    if cli_arg_value_is_numeric:
                        num_folds = int(cli_arg_value)
                    else:
                        print_bad_arg(FOLDS_CLI_ARG_NAME)
                        exit()
                elif cli_arg == DB_CLI_ARG_NAME:
                    db_path = cli_arg_value
                elif cli_arg == VISUALISE_CLI_ARG_NAME:
                    if cli_arg_value_is_numeric:
                        visualise_cnt = int(cli_arg_value)
                    else:
                        print_bad_arg(VISUALISE_CLI_ARG_NAME)
                        exit()
                elif cli_arg == SHOW_LATEX_MATRICES_CLI_ARG_NAME:
                    show_latex = True
                elif cli_arg == PRUNE_CLI_ARG_NAME:
                    prune = True
                else:
                    print_invalid_cli_call()
                    exit()
            else:
                print_invalid_cli_call()
                exit()

    if os.path.exists(db_path):
        print("Running...")
        run(db_path, num_folds, seed, visualise_cnt, show_latex, prune)
    else:
        print(f"File path {db_path} does not exist.")
