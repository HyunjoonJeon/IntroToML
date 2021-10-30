
class Utils:
    @classmethod
    def elementwise_list_sum(cls, list1, list2):
        return [sum(x) for x in zip(list1, list2)]
