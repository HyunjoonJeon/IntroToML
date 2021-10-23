class DTree:
    def __init__(self, attr, val, l_tree, r_tree):
        self.attr = attr
        self.val = val
        self.l_tree = l_tree
        self.r_tree = r_tree
        self.root = None

    # Factory Methods
    @classmethod
    def LeafNode(cls, val):
        return DTree(None, val, None, None)

    @classmethod
    def Node(cls, attr, val):
        return DTree(attr, val, None, None)

    def __repr__(self):
        isLeaf = (self.l_tree is None) and (self.r_tree is None)
        if isLeaf:
            return f"leaf:{self.val}"
        return f"( [X{self.attr} < {self.val}] LEFT: {self.l_tree} RIGHT: {self.r_tree} )"
