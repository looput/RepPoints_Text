## Introduction
1. deformable conv 节点代码 `layers/dcn`, 具体编译所依赖的环境见`layers/dcn/README.md`
1. 模型代码 `modeling\reppoint_detector.py`
2. 数据加载和处理代码 `dataset\text.py`, `data.py`
3. 训练框架TF,tensorpack


## TODO
- [x] DCN算子
- [x] 完成数据加载模块
- [x] 推理网络及后处理
- [ ] pytorch版本训练的参数迁移，验证推理网络的正确性
- [x] 训练代码(两个阶段，初始预测阶段和回归阶段)
- [x] 训练模型，验证正确性
- [x] 效果评估及可视化
- [ ] 数据扩增优化
- [ ] 配置文件优化
- [ ] batch支持（数据加载、部分后处理）
- [ ] Mask RCNN 和 RepPoints代码的融合，通过配置文件的更改直接训练


## 优化
1. IOU计算优化，使用Rotated Box
2. DCN offset的监督调整，对于Rotated box 可以添加约束
3. FPN融合方式更改


## Other

一个优化的版本 https://github.com/aws-samples/mask-rcnn-tensorflow 
issue: https://github.com/tensorflow/tensorflow/issues/32383