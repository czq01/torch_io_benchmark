from typing import Callable, Tuple
from torch.utils.data import DataLoader, Dataset
from tensordict import tensorclass
import numpy as np
import torch
import time

BATCH_SIZE = 5

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
            features = torch.from_numpy(features).cuda(),
            target = torch.from_numpy(target).cuda(),
            batch_size = [len(features)]
        )
        # data.memmap_()
        return data



def get_time_cost(func: Callable):
    def _wrapper(*args, **kwargs):
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        print(f"function: {func.__name__},  time cost: {end-start:.2f}s")
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

    #training process (Simple Iteration)
    resid = torch.zeros(size=(BATCH_SIZE, 1))
    if (cuda): resid = resid.cuda()
    for epoch in range(10**1):
        i=0
        for x, y in train_loader:
            # any custom func
            resid += y-x.mean(1).unsqueeze(1)
            i += 1
    return resid.mean().cpu()


@get_time_cost
def tensorclass_test(data: Tuple[np.array], cuda=True, load_only=True):
    train = NumericTensorClass.from_data(features = data[0], target = data[1])
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=lambda x: x)
    if (load_only): return True

    # training process
    resid = torch.zeros(size=(BATCH_SIZE, 1))
    if (cuda): resid = resid.cuda()
    for epoch in range(10**1):
        i=0
        for batch in train_loader:
            x = batch.features
            y = batch.target
            # any custom func
            resid += y-x.mean(1).unsqueeze(1)
            i += 1
    return resid.mean().cpu()


def main():
    for size in [(1000, 500), (10000, 5000), (50000, 25000)]:
        print(f"------> tensor size: {size}")
        data = get_test_data(size[0], size[1])
        for load in (True, False):
            print("Loading Time:" if load else "Loading + Traing(Iteration) Time:")
            dataloader_test(data, True, load_only=load)
            tensorclass_test(data, True, load_only=load)
    return


if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()


