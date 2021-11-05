from matplotlib.pyplot import show
import numpy as np
from Evaluation import EvalUtils, ConfusionMatrix
from ModelBuilders import DTree
from numpy.random import default_rng
from NumpyUtils import NpUtils
import sys
import os


def load_dataset(filepath):
    """
    Returns numpy array with dataset loaded from filepath.
    :param filepath: path to dataset file
    """
    return np.loadtxt(filepath)


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


def print_pruning_with_nested_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, visualise_cnt, rng=default_rng()):
    """
    Prints all average metrics of (k * k-1) pruned trees using nested k-cross validation on the dataset inputted.
    :param dataset: numpy array with dataset
    :param trained_model_constructor: function that returns a trained model
    :param num_folds: number of folds to use in nested cross validation
    :param num_class_labels: number of class labels
    :param visualise_cnt: number of trees to visualise
    :param rng: random number generator
    """
    pruned_trees, unpruned_trees = pruning_with_nested_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    print_comparison_stats(f"Pruning nested {num_folds}-cross validation", pruned_trees, unpruned_trees,
                           num_class_labels, visualise_cnt)


def wrap_as_boundary(value):
    return f"<!--- {value} --->"


def print_comparison_stats(name, tree_testdataset_pairs_1, tree_testdataset_pairs_2, num_class_labels,
                           visualise_cnt):
    name_1 = f"With {name}"
    name_2 = f"Without {name}"

    def print_comparison_separator(label, name_):
        print(wrap_as_boundary(f"{label} {name_}"))

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
                          avg_conf_matr_1, avg_conf_matr_2)

    wrap_print_comparison("Accuracy vals",
                          avg_conf_matr_1.accuracy(), avg_conf_matr_2.accuracy())
    wrap_print_comparison("Recall vals",
                          avg_conf_matr_1.recall(), avg_conf_matr_2.recall())
    wrap_print_comparison("Precision vals",
                          avg_conf_matr_1.precision(), avg_conf_matr_2.precision())
    wrap_print_comparison("F1 vals",
                          avg_conf_matr_1.f1_measure(), avg_conf_matr_2.f1_measure())

    def compute_tree_depth_avg(ts):
        return float(sum(list(map(lambda t: t.get_depth(), ts)))/len(ts))

    trees_1 = list(map(lambda pair: pair[0], tree_testdataset_pairs_1))
    trees_2 = list(map(lambda pair: pair[0], tree_testdataset_pairs_2))
    wrap_print_comparison("Average maximum depth",
                          compute_tree_depth_avg(trees_1), compute_tree_depth_avg(trees_2))

    for i in range(0, min(visualise_cnt, len(trees_1))):
        visualise_label = f"Visualise tree #{i+1}"
        print_comparison_separator(visualise_label, name_1)
        trees_1[i].visualise()
        print_comparison_separator(visualise_label, name_2)
        trees_2[i].visualise(show=True)
    print(wrap_as_boundary(f"End {name}."))


def trained_model_constructor(training_set):
    """
    Returns a trained decision tree model.
    :param training_set: numpy array with training set
    """
    return DTree.construct(training_set)


def run(filepath, num_folds, seed, visualise_cnt):
    """
    Runs the program.
    :param filepath: path to dataset file
    :param num_folds: number of folds to use in cross validation
    :param seed: seed for random number generator
    :param visualise_cnt: number of trees to visualise
    """
    dataset_used = load_dataset(filepath)
    rng = default_rng(seed)
    num_class_labels = len(NpUtils.unique_col_values(dataset_used, -1))
    print_pruning_with_nested_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, visualise_cnt, rng)


DB_CLI_ARG_NAME = "db"
FOLDS_CLI_ARG_NAME = "folds"
SEED_CLI_ARG_NAME = "seed"
VISUALISE_CLI_ARG_NAME = "visualise_cnt"
BIND_CLI_ARG_VALUE_CHAR = '='


def print_invalid_cli_call():
    """
    Prints invalid CLI call message.
    """
    print(
        f"Invalid cli call, argument format: [{VISUALISE_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{DB_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{FOLDS_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{SEED_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>]")


def print_bad_arg(argName):
    """
    Prints invalid argument message.
    """
    print(f"Bad argument given: {argName}")


if __name__ == '__main__':
    num_cli_args = len(sys.argv)
    is_cli_call = num_cli_args > 1

    num_folds = 10
    seed = 666
    db_path = "wifi_db/clean_dataset.txt"
    visualise_cnt = False
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
                else:
                    print_invalid_cli_call()
                    exit()
            else:
                print_invalid_cli_call()
                exit()

    if os.path.exists(db_path):
        print("Running...")
        run(db_path, num_folds, seed, visualise_cnt)
    else:
        print(f"File path {db_path} does not exist.")
