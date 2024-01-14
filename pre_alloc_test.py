from typing import Callable, Tuple
from torch.utils.data import DataLoader, Dataset
from tensordict import TensorDict, tensorclass
import numpy as np
import torch
import time
import torch.nn as nn
from torch_test import  NumericTensorClass, NumericDataSet, get_time_cost
import gc


def get_data(dim_len: int):
    x = [np.random.random((400,100,10)) for i in range(dim_len)]
    y = [np.random.random((400,100,1)) for i in range(dim_len)]
    return x, y

@get_time_cost
def tensordict_test(data, pre_alloc=True, **kwargs):
    x, y = data
    if pre_alloc:
        t = TensorDict({}, batch_size=[len(x),400, 100], device=0)
        for i in range(len(x)):
            t[i]["features"] = x[i]
            t[i]["target"] = y[i]
        return NumericTensorClass.from_tensordict(t)
    else:
        features = torch.cat(tuple(torch.from_numpy(i).cuda().reshape((1, 400, 100, 10)) for i in x), 0)
        target = torch.cat(tuple(torch.from_numpy(i).cuda().reshape((1, 400, 100, 1)) for i in y), 0)
        return NumericTensorClass(features, target, batch_size=[len(x), 400, 100])


@get_time_cost
def dataloader_test(data, **kwargs):
    x, y = data
    features = torch.cat(tuple(torch.from_numpy(i).cuda().reshape((1, 400, 100, 10)) for i in x), 0)
    target = torch.cat(tuple(torch.from_numpy(i).cuda().reshape((1, 400, 100, 1)) for i in y), 0)
    t = NumericDataSet(features, target)
    return DataLoader(t, batch_size=5)

def pre_alloc_loading_test():
    print("-"*50)
    print("loading testing:")
    testlist = [
        (dataloader_test, {"data": ()}),
        (tensordict_test, {"data": (), "pre_alloc": False}),
        (tensordict_test, {"data": (), "pre_alloc": True}),
    ]
    for len in (50, 300, 500, 800):
        print(f"---> size={len}")
        for test in testlist:
            x, y = get_data(len)
            test[1]["data"] = (x, y)
            test[0](**test[1])
            del x, y
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    pre_alloc_loading_test()