import gc
import math
from typing import Callable, Tuple
from torch.utils.data import DataLoader, Dataset
from tensordict import TensorDict, tensorclass
import numpy as np
import torch
import time
import torch.nn as nn

BATCH_SIZE = 20

class NumericDataSet(Dataset):

    def __init__(self, features, target):
        self.len = len(features)
        self.features = features
        self.target = target

    def __getitem__(self, index):
        return self.features[index], self.target[index]

    def __len__(self):
        return self.len

@tensorclass
class NumericTensorClass(Dataset):
    features: torch.Tensor
    target: torch.Tensor

    @classmethod
    def from_data(cls, features, target):
        data = cls(
            features = torch.from_numpy(features).cuda(),
            target = torch.from_numpy(target).cuda(),
            batch_size = [len(features)],
            device = 0
        )
        # data.memmap_()
        return data

class LinearRegression(torch.nn.Module):
    def __init__(self, xlen, ylen) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(xlen, ylen, dtype=torch.float64).cuda()

    def forward(self, x):
        return self.linear(x)

    def train(self, train, **kwargs):
        resid = torch.zeros(size=(BATCH_SIZE, 1)).cuda()
        if isinstance(train, DataLoader):
            for x, y in train:
                resid += y-self.forward(x)
            return resid.mean().cpu()
        elif isinstance(train, NumericTensorClass):
            if kwargs.get("slice", False):
                _step = math.ceil(len(train)/20)
                for idx in range(BATCH_SIZE):
                    batch = train[idx::_step]
                    resid += batch.target-self.forward(batch.features)
            else:
                for batch in DataLoader(dataset=train, batch_size=BATCH_SIZE, collate_fn=lambda x:x, shuffle=False):
                    resid += batch.target-self.forward(batch.features)
            return resid.mean().cpu()


def get_time_cost(func: Callable):
    def _wrapper(t=10, *args, **kwargs,):
        t_total = 0
        max_t = 0; min_t = 0x7FFFFFFF
        for i in range(t):
            start = time.time()
            val = func(*args, **kwargs)
            end = time.time()
            t_total += end-start
            max_t = max(end-start, max_t)
            min_t = min(end-start, min_t)
        ave_t = (t_total-max_t - min_t)/(t-2)
        try:
            kwargs.pop('data')
        except: pass
        print(f"function: {func.__name__},  ave time cost: {ave_t:.2f}s,  args={kwargs}")
        return val, (end-start)
    return _wrapper


# 共同外部数据
def get_test_data(r: int, c: int) -> Tuple[np.ndarray]:
    return (np.random.random((r, c)),  # X (c features)
            np.random.random((r, 1)))  # Y


@get_time_cost
def dataloader_test(data: Tuple[np.ndarray], load_only=True, **kwargs):
    train = NumericDataSet(torch.from_numpy(data[0]).cuda(),
                           torch.from_numpy(data[1]).cuda())
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE,shuffle=False)
    if (load_only): return

    #training process (Linear Regression on GPU) to test Iteration speed
    linear_model = LinearRegression(train.features.shape[1],train.target.shape[1]).cuda()
    return linear_model.train(train_loader)


@get_time_cost
def tensorclass_test(data: Tuple[np.array], load_only=True, **kwargs):
    ## loading 方式一： 使用from_data直接构造Dataset
    train = NumericTensorClass.from_data(features=data[0], target=data[1])
    ## loading 方式二： 使用TensorDict 内存pre-alloc，随后将数据填入。
    # td = TensorDict({}, [len(data[0])], device=0)
    # td["features"] = data[0]; td["target"] = data[1]
    # train = NumericTensorClass.from_tensordict(td)
    if (load_only): return True

    # training process
    linear_model = LinearRegression(train.features.shape[1],train.target.shape[1]).cuda()
    return linear_model.train(train, **kwargs)


def iterating_test():
    print("-"*50)
    print("Iterating Test: ")
    testlist = [
        (dataloader_test, {"data": (), "load_only": True}),
        (tensorclass_test, {"data": (), "load_only": True}),
        (dataloader_test, {"data": (), }),
        (tensorclass_test, {"data": (), }),
        (tensorclass_test, {"data": (), "slice": True}),
    ]

    for size in [(15000, 5000), (30000, 10000)]:
        print(f"------> size: {size}")
        for test in testlist:
            data = get_test_data(size[0], size[1])
            test[1]["data"] = data
            test[0](**test[1])
            del data
            gc.collect()
            torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    torch.cuda.set_device(0)
    iterating_test()


