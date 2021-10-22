class DTree:

    def __init__(self, attr, val, l_tree, r_tree):
        self.attr = attr
        self.val = val
        self.l_tree = l_tree
        self.r_tree = r_tree

    # Factory Methods
    @classmethod
    def LeafNode(cls, val):
        return DTree(val, None, None, None)

    @classmethod
    def Node(cls, val):
        return DTree(val, )
