# Project

PyTorch DataLoader和TorchDict的IO性能评价

## Introduction

在分布式机器学习训练中，如何保持高效的训练数据读取是一项非常重要的课题。`PyTorch`提供了`DataLoader`来解决这个问题，但是我们在使用过程中总会觉得并不是那么高效。`TensorDict`则是给出了一个`tensorclass`的解决方案，并且看上去很promising。那么，事实真的如此吗？

## Requirement && Evaluation

### Objective

设计实验，比较`DataLoader`和`tensorclass`的性能，并进行分析。如无单机多卡的实验环境，可先实现单机单卡，然后联系[zhanghan](maito:zhanghan@higgsasset.com)安排多卡环境测试。

### Evaluation

- 完成benchmark设计并能获得基础的测试结果
- 对测试结果进行简单分析
- 实现多机多卡加分
- 规范使用`git`，规范使用`README`、`CHANGELOG`、`.gitignore`和`License`
- 代码条理清晰，且符合`PEP 8`规范

### Related Works

- [TorchDict Tutorial](https://pytorch.org/tensordict/tutorials/tensordict_memory.html)
- [PyTorch/TorchDict Intergration](https://github.com/pytorch/pytorch/pull/112441)

## Project Design

使用共同外部NDArray数据，基于相同训练函数，针对DataLoader及TensorDict分部记录运行时间并对照分析。

控制变量在不同数据量的情况下进行分析。

## Test Env & Instruction

Dependable package in `requirements.txt`.

可以使用以下命令初始化并使用名为`torch_test`的环境:

```bash
virutalenv torch_env
source ./torch_env/bin/activate
pip3 install -r requirements.txt
```

随后可以使用以下命令运行测试:

```bash
python ./torch_test.py
```

## Test Result & Analysis

本机单卡测试基于Python3.8.7，多次运行典型测试结果如下图：

![pic/result.png loading failed](./pic/result.png)

依据结果可以看出：

1. 在数据loading并进入GPU这一过程中， tensordict的性能与普通dataloader性能整体相差不大。
2. 考虑training时Iteration的过程，从tensorclass嵌套DataLoader取batch数据的速度在数据量大时则显著慢于普通dataloader， 而若是直接使用tensorclass的切片操作绕过Dataloader，则会有一个非常明显的性能提升。也即性能排序： tensorclass切片>普通Dataloader>tensorclass套Dataloader。

除此之外，在代码编写过程中依据时间花费分析的思路进行了如下对比测试：
1. 将loading的目标设备设置为CPU，会发现二者在加载数据过程的时间花费均降低至为0.00s，即二者加载数据的主要时间瓶颈均发生在由CPU向GPU复制的过程，而此过程并不存在提升空间。
2. 测试了tensorclass的两种数据load方式(参见tensorclass_test中的注释部分)：tensordict预分配内存和现场直接创建obj的方式。经测试在无nested data的情况下速度相同，因为不存在多次内存分配的调用开销。但若是存在大量多次load Tensor某一维度截面数据的情况可能会有不同。
3. 根据上述分析，loading过程的另一处瓶颈可能存在于disk IO read的过程。但由于tensorclass文档中所述磁盘内存映射的`class MemoryMapperTensor`在当前版本的package中尚未实现无法调用，故而未进行测试。