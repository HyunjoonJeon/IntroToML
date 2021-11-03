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


def print_no_pruning_with_k_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, rng=default_rng()):
    """
    Prints all average metrics of k trees using k-cross validation on the dataset inputted.
    :param dataset: numpy array with dataset
    :param trained_model_constructor: function that returns a trained model
    :param num_folds: number of folds to use in k-cross validation
    :param num_class_labels: number of class labels
    :param rng: random number generator
    """
    trees = EvalUtils.k_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    print_stats(
        f"No Pruning, {num_folds}-cross validation", trees, num_class_labels)


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
        DTree.prune(val_db, model, model)
    return EvalUtils.nested_k_cross_validation(num_folds, dataset, trained_model_constructor,
                                               apply_validation_set, rng)


def print_pruning_with_nested_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, rng=default_rng()):
    """
    Prints all average metrics of (k * k-1) pruned trees using nested k-cross validation on the dataset inputted.
    :param dataset: numpy array with dataset
    :param trained_model_constructor: function that returns a trained model
    :param num_folds: number of folds to use in nested cross validation
    :param num_class_labels: number of class labels
    :param rng: random number generator
    """
    pruned_trees = pruning_with_nested_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    print_stats(
        f"With Pruning nested {num_folds}-cross validation", pruned_trees, num_class_labels)


def print_stats(name, trees, num_class_labels):
    """
    Prints average metrics of inputted trees including confusion matrix, accuracy, precision, recall, and f1 score per class.
    :param name: name of the set of trees
    :param trees: list of trees
    :param num_class_labels: number of class labels
    """
    print(f"<!--- Start {name} --->")
    avg_conf_matr = ConfusionMatrix.construct_avg_confusion_matrix(
        trees, num_class_labels)
    print("Average confusion matrix: \n", avg_conf_matr)
    print("Accuracy vals: ",
          avg_conf_matr.accuracy())
    print("Recall vals: ", avg_conf_matr.recall())
    print("Precision vals: ",
          avg_conf_matr.precision())
    print("F1 vals: ", avg_conf_matr.f1_measure())
    print(f"<!--- End {name}. --->")

def trained_model_constructor(training_set):
    """
    Returns a trained decision tree model.
    :param training_set: numpy array with training set
    """
    return DTree.construct(training_set)


def run(filepath, num_folds, seed):
    """
    Runs the program.
    :param filepath: path to dataset file
    :param num_folds: number of folds to use in cross validation
    :param seed: seed for random number generator
    """
    dataset_used = load_dataset(filepath)
    rng = default_rng(seed)
    num_class_labels = len(NpUtils.unique_col_values(dataset_used, -1))
    print_no_pruning_with_k_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, rng)
    print_pruning_with_nested_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, rng)


DB_CLI_ARG_NAME = "db"
FOLDS_CLI_ARG_NAME = "folds"
SEED_CLI_ARG_NAME = "seed"
BIND_CLI_ARG_VALUE_CHAR = '='


def print_invalid_cli_call():
    """
    Prints invalid CLI call message.
    """
    print(
    f"Invalid cli call, argument format: [{DB_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{FOLDS_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{SEED_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>]")


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
                else:
                    print_invalid_cli_call()
                    exit()
            else:
                print_invalid_cli_call()
                exit()

    if os.path.exists(db_path):
        print("Running...")
        run(db_path, num_folds, seed)
    else:
        print(f"File path {db_path} does not exist.")
