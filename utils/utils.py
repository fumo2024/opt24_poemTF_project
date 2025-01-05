import collections.abc
from itertools import repeat


###### 定义用到的部分工具函数 ######


# 产生维度为n的tuple
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)