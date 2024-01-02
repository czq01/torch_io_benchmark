from typing import Callable, Tuple
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import time

BATCH_SIZE = 1

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


# 训练函数
def train_func(data: DataLoader[Tuple[torch.Tensor,...]], cuda = False):
    c = data.dataset.tensors[0].shape[1]
    resid = torch.zeros(size=(c, 1))
    if (cuda): resid = resid.cuda()
    for epoch in range(10**1):
        i=0
        for x, y in data:
            if (cuda):
                # TODO if any better way ??? (one-time copy?)
                x = x.cuda()
                y = y.cuda()
            # any custom func
            resid += y-x.mean()
            i += 1
    return resid.cpu()


@get_time_cost
def dataloader_test(data: Tuple[np.ndarray], cuda=False):
    x = torch.from_numpy(data[0])
    y = torch.from_numpy(data[1])
    train = TensorDataset(x, y)
    train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=1)
    val = train_func(train_loader, cuda)
    return val


def main():
    # TODO  for data_size in (100, 50), (1000, 500) ...
    data = get_test_data(1000, 500)
    result = dataloader_test(data, True)
    # TODO result = tensorDict_test()

    return

# training_data = DataLoader()
if __name__ == "__main__":
    torch.cuda.set_device(0)
    main()


