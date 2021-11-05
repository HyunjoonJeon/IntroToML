
class Utils:
    @classmethod
    def elementwise_list_sum(cls, list1, list2):
        return [sum(x) for x in zip(list1, list2)]

    @classmethod
    def project_fst_from_pairs(cls, pairs):
        return list(map(lambda pair: pair[0], pairs))

    @classmethod
    def construct_zeros_list(cls, num_zeros):
        ret = list()
        for i in range(0, num_zeros):
            ret.append(0)
        return ret
