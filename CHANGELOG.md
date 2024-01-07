# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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