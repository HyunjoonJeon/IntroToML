import numpy as np
from Evaluation import EvalUtils, ConfusionMatrix
from ModelBuilders import DTree
from numpy.random import default_rng
from NumpyUtils import NpUtils
import sys
import os


def load_dataset(filepath):
    return np.loadtxt(filepath)


def print_no_pruning_with_nested_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, rng=default_rng()):
    trees = EvalUtils.k_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    print_stats(
        f"No Pruning, nested {num_folds}-cross validation", trees, num_class_labels)


def pruning_with_nested_cross_validation(num_folds, dataset, trained_model_constructor, rng=default_rng()):
    def apply_validation_set(
            val_db, model):
        DTree.prune(val_db, model, model)
    return EvalUtils.nested_k_cross_validation(num_folds, dataset, trained_model_constructor,
                                               apply_validation_set, rng)


def print_pruning_with_nested_cross_validation(dataset, trained_model_constructor, num_folds, num_class_labels, rng=default_rng()):
    pruned_trees = pruning_with_nested_cross_validation(
        num_folds, dataset, trained_model_constructor, rng)
    print_stats(
        f"With Pruning nested {num_folds}-cross validation", pruned_trees, num_class_labels)


def print_stats(name, trees, num_class_labels):
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


# tree, depth = decision_tree_learning(skewed_dataset)
# trained_tree, depth = decision_tree_learning(clean_dataset)
# print_cross_validation(clean_dataset)
def trained_model_constructor(
    training_set): return DTree.construct(training_set)


def run(filepath, num_folds, seed):
    dataset_used = load_dataset(filepath)
    rng = default_rng(seed)
    num_class_labels = len(NpUtils.unique_col_values(dataset_used, -1))
    print_no_pruning_with_nested_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, rng)
    print_pruning_with_nested_cross_validation(
        dataset_used, trained_model_constructor, num_folds, num_class_labels, rng)


DB_CLI_ARG_NAME = "db"
FOLDS_CLI_ARG_NAME = "folds"
SEED_CLI_ARG_NAME = "seed"
BIND_CLI_ARG_VALUE_CHAR = '='


def print_invalid_cli_call():
    print(
        f"Invalid cli call, argument format: [{DB_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{FOLDS_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>] [{SEED_CLI_ARG_NAME}{BIND_CLI_ARG_VALUE_CHAR}<value>]")


def print_bad_arg(argName):
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
