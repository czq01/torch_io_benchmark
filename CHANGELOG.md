# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.2.0] - 2024-01-14

### Added

* 新增pre_alloc_test,测试tensorclass内存预分配对数据loading过程性能的影响。
* 新增main.py 将两项测试整合，并提供入口
* 添加loading部分的测试结论，修改了readme

### Changed

* 修改实验组结构，使用testlist[(func, *args)]的结构进行测试
* 修改项目结构名称， 将torch_test中main函数修改为iteration_test
* 将torch_test文件改名为iter_test
* 将pre_alloc_teset文件改名为loading_test

## [1.1.2] - 2024-01-14

### Fixed

* 修复测试的扰动项，在每项测试完成后清除所有数据，并清空GPU
* 更正最新结论，即slicing不会带来遍历性能提升

## [1.1.1] - 2024-01-12

### Added

* 新增tensorclass切片操作在training过程中的效率分析

### Update

* 新增切片操作对比测试结果
* Readme新增tensordict pre-allocate 二维tensor测试结果
* Readme新增loading部分时间花费瓶颈分析。

### Fix Bugs

* 修复未控制变量的对比实验，更改最新实验结果。

## [1.1.0] - 2024-01-11

### Added

* 新增LinearRegression训练模型，用于模拟全GPU计算，减少CPU与GPU的通信瓶颈，更加接近实际情况

### Changed

* 去除tensorclass与DataLoader交互，改为使用Slice切片组装训练集Batch
* 修改默认Batch大小为20


## [1.0.1] - 2024-01-05

### Added

* 在main()函数中新增不同数据size的对比测试。
* 在main()函数中新增纯loading时间对比。
* 添加了测试函数中load和train部分的分隔注释。

### Changed

* 删去NumericTensorClass.from_data()函数的无用参数batch_size
* 删去get_test_data()函数返回的无用label数据。

## [1.0.0] - 2024-01-05

### Added

* 添加DataLoader的NumericDataset类
* 添加tensorclass的NumericTensorClass类
* 建立相同训练函数，可以对总体时间进行对比测试
* 添加可选load_only参数，支持进行loading过程单独测试

### Changed

* 修改Dataloader加载方式，将feature及target以NumericDataSet的形式载入，并在创建Tensor时即复制进入GPU
* 修改默认batch_size为5
* 删去训练函数`train_func()`的显式定义，将其内联至测试函数内部
* 将数据大小size修改为10000x5000



## [0.1.0] - 2024-01-01

### Added

* 添加数据函数`get_test_data()`, 生成随机numpy矩阵数据
* 添加装饰器`@get_time_cost`, 输出函数运行时间
* 添加训练函数`train_func()`
* 添加`dataloader_test()`函数，用以测试Dataloader性能
* 设置torch.cuda数据为单GPU卡，默认使用GPU:0进行计算
