from typing import Callable, Tuple
from torch.utils.data import DataLoader, Dataset
from tensordict import tensorclass
import numpy as np
import torch
import time
import torch.nn as nn

BATCH_SIZE = 20

class NumericDataSet(Dataset):

    def __init__(self, features, target):
        self.len = len(features)
        self.features = torch.from_numpy(features).cuda()
        self.target = torch.from_numpy(target).cuda()

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
            features = torch.from_numpy(features),
            target = torch.from_numpy(target),
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

    def train(self, train):
        resid = torch.zeros(size=(BATCH_SIZE, 1)).cuda()
        if isinstance(train, DataLoader):
            for x, y in train:
                resid += y-self.forward(x)
            return resid.mean().cpu()
        elif isinstance(train, NumericTensorClass):
            for idx in range(0, len(train), BATCH_SIZE):
                batch = train[idx:idx+BATCH_SIZE]
                x = batch.features
                y = batch.target
                resid += y-self.forward(x)
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
        print(f"function: {func.__name__},  ave time cost: {ave_t:.2f}s")
        return val, (end-start)
    return _wrapper


# 共同外部数据
def get_test_data(r: int, c: int) -> Tuple[np.ndarray]:
    return (np.random.random((r, c)),  # X (c features)
            np.random.random((r, 1)))  # Y


@get_time_cost
def dataloader_test(data: Tuple[np.ndarray], cuda=True, load_only=True):
    train = NumericDataSet(data[0], data[1])
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE,shuffle=True)
    if (load_only): return

    #training process (Linear Regression on GPU) to test Iteration speed
    linear_model = LinearRegression(train.features.shape[1],train.target.shape[1]).cuda()
    return linear_model.train(train_loader)



@get_time_cost
def tensorclass_test(data: Tuple[np.array], cuda=True, load_only=True):
    train = NumericTensorClass.from_data(features=data[0], target=data[1])
    if (load_only): return True

    linear_model = LinearRegression(train.features.shape[1],train.target.shape[1]).cuda()

    # training process
    return linear_model.train(train)


def main():
    for size in [(5000, 10000), (10000, 50000)]:
        print(f"------> tensor size: {size}")
        for load in (True, False):
            print("Loading Time:" if load else "Loading + Traing(Iteration) Time:")
            data = get_test_data(size[0], size[1])
            dataloader_test(data=data, cuda=True, load_only=load)
            data = get_test_data(size[0], size[1])
            tensorclass_test(data=data, cuda=True, load_only=load)
    return


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()


