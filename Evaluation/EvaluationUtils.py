
import numpy as np
from ModelBuilders.DecisionTree import DTree
from NumpyUtils import NpUtils
from numpy.random import default_rng


class EvalUtils:
    # Copy-paste from tutorial
    @classmethod
    def k_fold_split(cls, n_splits, n_instances, random_generator=default_rng()):
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

    @classmethod
    def train_test_k_fold(cls, n_folds, n_instances, random_generator=default_rng()):
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
        split_indices = EvalUtils.k_fold_split(
            n_folds, n_instances, random_generator)

        folds = []
        for k in range(n_folds):
            test_indices = np.array(split_indices[k])
            # Concatenate all other folds
            train_indices = np.concatenate(
                split_indices[:k] + split_indices[k+1:])

            folds.append([train_indices, test_indices])

        return folds

    # End Copy-paste from tutorial

    @classmethod
    def k_cross_validation(cls, k, dataset, trained_model_constructor, random_generator=default_rng()):
        """
        Return a list of models after performing k-fold cross validation on the dataset.
        :param k: Number of folds
        :param dataset: Numpy dataset to perform cross validation on
        :param trained_model_constructor: A function that takes in a training dataset and returns a trained model
        :param random_generator: A random generator
        """
        # number of rows
        n_instances = NpUtils.row_count(dataset)

        final_models = list()
        for (train_indices, test_indices) in EvalUtils.train_test_k_fold(k, n_instances, random_generator):
            training_dataset = dataset[train_indices, :]
            test_dataset = dataset[test_indices, :]
            trained_model = trained_model_constructor(training_dataset)
            final_models.append([trained_model, test_dataset])
        return final_models

    @classmethod
    def nested_k_cross_validation(cls, k, dataset, trained_model_constructor, apply_validation_set,
                                  random_generator=default_rng()):
        """
        Return a list of models after performing nested k-fold cross validation on the dataset.
        :param k: Number of folds
        :param dataset: Numpy dataset to perform cross validation on
        :param trained_model_constructor: A function that takes in a training dataset and returns a trained model
        :param apply_validation_set: A function that takes in the validation dataset and tunes the trained model
        :param random_generator: A random generator
        """
        # number of rows
        n_instances = NpUtils.row_count(dataset)
        final_models = list()
        for (train_indices, test_indices) in EvalUtils.train_test_k_fold(k, n_instances, random_generator):
            training_and_val_dataset = dataset[train_indices, :]
            test_dataset = dataset[test_indices, :]
            for (train_indices, val_indices) in EvalUtils.train_test_k_fold(k-1, train_indices.size, random_generator=None):
                # Train new model, but don't shuffle
                training_dataset = training_and_val_dataset[train_indices, :]
                validation_dataset = training_and_val_dataset[val_indices, :]
                trained_model = trained_model_constructor(training_dataset)
                apply_validation_set(validation_dataset, trained_model)
                final_models.append([trained_model, test_dataset])
        return final_models
