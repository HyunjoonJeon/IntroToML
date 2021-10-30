import matplotlib.pyplot as plt


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

    def __init__(self, attr, val, l_tree, r_tree, depth, is_leaf):
        self.attr = attr
        self.val = val
        self.l_tree = l_tree
        self.r_tree = r_tree
        self.depth = depth
        self.is_leaf = is_leaf
        self.counts = None

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
            total_counts = [sum(x)
                            for x in zip(total_counts, self.r_tree.counts)]
        if self.l_tree:
            # add to total_counts
            total_counts = [sum(x)
                            for x in zip(total_counts, self.l_tree.counts)]
            # pick majority
        return max(enumerate(total_counts), key=lambda total_count_idx: total_count_idx[1])[0] + 1

    def convert_to_leaf(self):
        """Convert node to a leaf whose value is the majority class label, returns tuple of old attributes for "convert_back"."""
        old_attrs = (self.attr, self.val, self.l_tree,
                     self.r_tree, self.counts)
        self.val = self.get_majority_class_label()

        self.is_leaf = True
        self.counts = DTree.elementwise_list_sum(
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
    def elementwise_list_sum(cls, list1, list2):
        return [sum(x) for x in zip(list1, list2)]

    # Factory Methods

    @ classmethod
    def LeafNode(cls, val, depth):
        return DTree(None, val, None, None, depth, True)

    @ classmethod
    def Node(cls, attr, val, depth):
        return DTree(attr, val, None, None, depth, False)

    def __repr__(self):
        isLeaf = (self.l_tree is None) and (self.r_tree is None)
        tabs = "\t" * self.depth
        if isLeaf:
            return f"{tabs}leaf:{self.val}"

        condition = f"[X{self.attr} < {self.val}]"
        left_tree = f"{tabs}LEFT: \n{self.l_tree}"
        right_tree = f"{tabs}RIGHT: \n{self.r_tree}"
        return "\n".join([condition, left_tree, right_tree])
