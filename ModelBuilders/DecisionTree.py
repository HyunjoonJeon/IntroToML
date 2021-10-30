import numpy as np
import matplotlib.pyplot as plt
from NumpyUtils import NpUtils
from Utils import Utils


def _visualise(tree, xs, ys, anns, xval):

    xs.append(xval)
    ys.append(tree.depth * -1)

    if tree.is_leaf:
        anns.append(f"leaf:{tree.val}")
    else:
        anns.append(f"[X{tree.attr} < {tree.val}]")
        offset = 50
        if tree.depth != 0:
            offset = offset - 2 * tree.depth

        _visualise(tree.l_tree, xs, ys, anns, xval + offset)
        _visualise(tree.r_tree, xs, ys, anns, xval - offset)


class DTree:

    def __init__(self, attr, val, l_tree, r_tree, depth, is_leaf, unique_labels):
        self.attr = attr
        self.val = val
        self.l_tree = l_tree
        self.r_tree = r_tree
        self.depth = depth
        self.is_leaf = is_leaf
        self.counts = None
        self.unique_labels = unique_labels

    def visualise(self):
        xs, ys, anns = [], [], []
        _visualise(self, xs, ys, anns, 0)
        # print(xs, ys, anns)

        fig, dtree = plt.subplots()
        plt.scatter(x=xs, y=ys)

        for i, ann in enumerate(anns):
            dtree.annotate(ann, (xs[i], ys[i]))
        plt.show()

    def init_counts(self, num_labels):
        # reset all internal counts
        self.counts = [0] * num_labels
        self.r_tree and self.r_tree.init_counts(num_labels)
        self.l_tree and self.l_tree.init_counts(num_labels)

    def predict(self, attrs_row):
        # last column is the correct classification label
        attr_value = attrs_row[self.attr]
        if (self.is_leaf):
            if self.counts:
                self.counts[int(attrs_row[-1]) - 1] += 1
            return self.val
        if (attr_value < self.val):
            # take left
            return self.l_tree.predict(attrs_row)
        return self.r_tree.predict(attrs_row)

    def is_only_leaf_parent(self):
        return (self.l_tree and self.l_tree.is_leaf) and (self.r_tree and self.r_tree.is_leaf)

    def get_all_leaf_only_parents(self):
        ret = []
        if self.l_tree and not self.l_tree.is_leaf:
            ret += self.l_tree.get_all_leaf_only_parents()
        if self.r_tree and not self.r_tree.is_leaf:
            ret += self.r_tree.get_all_leaf_only_parents()

        is_all_leaf_only_parent = ((self.l_tree is None) or self.l_tree.is_leaf) and (
            (self.r_tree is None) or self.r_tree.is_leaf)
        if is_all_leaf_only_parent:
            ret += [self]
        return ret

    def get_majority_class_label(self):
        total_counts = self.counts  # 0s
        if self.r_tree:
            # add to total_counts
            total_counts = Utils.elementwise_list_sum(
                total_counts, self.r_tree.counts)
        if self.l_tree:
            # add to total_counts
            total_counts = Utils.elementwise_list_sum(
                total_counts, self.l_tree.counts)
            # pick majority
        return max(enumerate(total_counts), key=lambda total_count_idx: total_count_idx[1])[0] + 1

    def convert_to_leaf(self):
        """Convert node to a leaf whose value is the majority class label, returns tuple of old attributes for "convert_back"."""
        old_attrs = (self.attr, self.val, self.l_tree,
                     self.r_tree, self.counts)
        self.val = self.get_majority_class_label()

        self.is_leaf = True
        self.counts = Utils.elementwise_list_sum(
            self.l_tree.counts, self.r_tree.counts)
        self.l_tree, self.r_tree = None, None
        self.attr = None
        return old_attrs

    def convert_back(self, attr, val, l_tree, r_tree, counts):
        # (attr, val, l_tree, r_tree)
        self.val = val
        self.is_leaf = False
        self.l_tree, self.r_tree = l_tree, r_tree
        self.attr = attr
        self.counts = counts

    @classmethod
    def evaluate(cls, test_db, root, init_counts=False):
        """Returns the accuracy of the tree
        """
        # Modify "unique_labels", test data may have unseen labels
        root.unique_labels = root.unique_labels.union(
            NpUtils.unique_col_values(test_db, -1))

        num_labels = len(root.unique_labels)
        if init_counts:
            root.init_counts(num_labels)

        correct_predictions = 0
        total = NpUtils.row_count(test_db)
        for i in range(total):
            test_row = test_db[i]
            predicted_label = root.predict(test_row)
            correct_label = test_row[-1]
            is_correct_prediction = predicted_label == correct_label
            if is_correct_prediction:
                correct_predictions += 1
        return float(correct_predictions / total)

    @classmethod
    def construct(cls, training_dataset):
        unique_labels = NpUtils.unique_col_values(training_dataset, -1)
        built_tree, built_tree_depth = DTree.decision_tree_learning(
            training_dataset, unique_labels)
        return built_tree

    @classmethod
    def prune(cls, val_db, tree, root_tree):
        if tree.is_only_leaf_parent():
            # prune
            # convert into a leaf whose value is the majority class label
            before_prune_accuracy = DTree.evaluate(
                val_db, root_tree, init_counts=True)
            attr, val, l_tree, r_tree, counts = tree.convert_to_leaf()
            after_prune_accuracy = DTree.evaluate(
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

        l_tree_pruned = tree.l_tree and DTree.prune(
            val_db, tree.l_tree, root_tree)
        r_tree_pruned = tree.r_tree and DTree.prune(
            val_db, tree.r_tree, root_tree)

        return l_tree_pruned and r_tree_pruned and DTree.prune(val_db, tree, root_tree)

    @classmethod
    def decision_tree_learning(cls, training_dataset, unique_labels, depth=0):
        labels = training_dataset[:, -1]
        if np.unique(labels).size == 1:
            return DTree.LeafNode(labels[0], depth, unique_labels), depth
        split_idx, split_value, l_dataset, r_dataset = DTree.SplitUtils.find_split(
            training_dataset)

        node = DTree.Node(split_idx, split_value, depth, unique_labels)
        l_branch, l_depth = DTree.decision_tree_learning(
            l_dataset, depth+1)
        r_branch, r_depth = DTree.decision_tree_learning(
            r_dataset, depth+1)
        node.l_tree, node.r_tree = l_branch, r_branch
        return node, max(l_depth, r_depth)

    class SplitUtils:
        @classmethod
        def information_entropy(cls, dataset):
            labels = dataset.transpose()[-1]

            _, unique_label_counts = np.unique(labels, return_counts=True)
            total_entries = labels.size
            ret = 0
            for label_cnt in unique_label_counts:
                px = label_cnt/total_entries
                ret -= px * np.log2(px)
            return ret

        @classmethod
        def remainder(cls, s_left, s_right):
            """PRE: "s_left" and "s_right" have labels on their last column.
            """
            def no_labels_size(s): return s[:, 0].size

            left_size = no_labels_size(s_left)
            right_size = no_labels_size(s_right)

            h_left = DTree.SplitUtils.information_entropy(s_left)
            h_right = DTree.SplitUtils.information_entropy(s_right)

            total = left_size + right_size
            return (left_size/total) * h_left + (right_size/total) * h_right

        @classmethod
        def information_gain(cls, dataset, left, right):
            h_dataset = DTree.SplitUtils.information_entropy(dataset)
            return h_dataset - DTree.SplitUtils.remainder(left, right)

        @classmethod
        def find_split(cls, dataset):
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
                attr_i_gain = DTree.SplitUtils.information_gain(
                    dataset, left, right)

                if attr_i_gain > max_attr_gain:
                    max_attr_gain = attr_i_gain
                    max_attr_idx = i
                    max_left, min_right = left.max(
                        axis=0)[i], right.min(axis=0)[i]
                    split_value = (max_left + min_right) / 2
                    left_ret, right_ret = left, right

            return max_attr_idx, split_value, left_ret, right_ret

    # Factory Methods

    @ classmethod
    def LeafNode(cls, val, depth, unique_labels):
        return DTree(None, val, None, None, depth, True, unique_labels)

    @ classmethod
    def Node(cls, attr, val, depth, unique_labels):
        return DTree(attr, val, None, None, depth, False, unique_labels)

    def __repr__(self):
        isLeaf = (self.l_tree is None) and (self.r_tree is None)
        tabs = "\t" * self.depth
        if isLeaf:
            return f"{tabs}leaf:{self.val}"

        condition = f"[X{self.attr} < {self.val}]"
        left_tree = f"{tabs}LEFT: \n{self.l_tree}"
        right_tree = f"{tabs}RIGHT: \n{self.r_tree}"
        return "\n".join([condition, left_tree, right_tree])
