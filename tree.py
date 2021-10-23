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
        self.root = None
        self.depth = depth
        self.is_leaf = is_leaf

    def visualise(self):
        xs, ys, anns = [], [], []
        _visualise(self, xs, ys, anns, 0)
        # print(xs, ys, anns)

        fig, dtree = plt.subplots()
        plt.scatter(x=xs, y=ys)

        for i, ann in enumerate(anns):
            dtree.annotate(ann, (xs[i], ys[i]))
        plt.show()

    # Factory Methods

    @classmethod
    def LeafNode(cls, val, depth):
        return DTree(None, val, None, None, depth, True)

    @classmethod
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
