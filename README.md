## Introduction
该代码复现主基于TensorPack, 网络构建使用TF1的操作完成。整个代码结构参考FastRCNN/MaskRCNN，具有一定的拓展能力，后续可添加其他检测和识别方法，对代码进行整合。

### 核心代码
- deformable conv 节点代码 `layers/dcn`, 具体编译所依赖的环境见`layers/dcn/README.md`
- 模型代码 `modeling/reppoint_detector.py`
- 数据加载和处理代码 `dataset/text.py`, `data.py`
- 评测脚本在`dataset/eval_tools`之中，采用多边形方式描述

### 实现细节
- 改代码基于TensorPack实现，训练逻辑、分布式训练等由TensorPack来处理。复现的部分主要为数据加载、模型定义、训练过程中target生成loss计算、以及后处理部分。
- RepPoints可以看做1.5阶段的算法，第一次回归的GT由于是固定的，在数据加载`data.py`中生成，避免在Graph处理，第二次回归的target和分类的label由于跟第一阶段预测结果有关，故在训练Graph中处理。
- 后处理通过tf的操作实现，因此Inference Graph可直接输出label和polygon。

### 使用
- 训练框架TF,[tensorpack](https://github.com/tensorpack/tensorpack)，具体依赖见`requirements.txt`。由于DCN编译链接的是TF1.14，使用其他版本会出问题。
- 预训练模型 [models](http://models.tensorpack.com/#FasterRCNN) 下载。
- 模型训练 `sh train.sh`. 目前将配置参数放在config.py中，亦可在代码启动时指定参数。
- 模型推理 `sh eval.sh`. 具体参数可更改。`--output-pb 'xxx.pb'` 增加该参数，即可对模型进行导出为pb文件。
- 模型调试 数据接口调试直接使用 `python data.py`即可可视化数据加载结果。网络部分由于Tensorpackd的逻辑基于TF1x，因此在整体调试时只能以动态图的形式来进行，不过初始的调试到可以开启eager模式，直接运行`python modeling/reppoint_detector.py`（需要进行简单的修改）,这样即可动态观察输出结果，提高调试效率。
- `NOTES.md` 对原始的FastRCNN进行介绍，部分内容是相似的，可参考。


## TODO
- [x] DCN算子
- [x] 完成数据加载模块
- [x] 推理网络及后处理
- [ ] ~~pytorch版本训练的参数迁移，验证推理网络的正确性~~
- [x] 训练代码(两个阶段，初始预测阶段和回归阶段)
- [x] 训练模型，验证正确性
- [x] 效果评估及可视化
- [x] 数据扩增优化
- [ ] 数据加载进行比例控制
- [ ] 配置文件优化
- [ ] batch支持（数据加载、部分后处理）
- [ ] Mask RCNN 和 RepPoints代码的融合，通过配置文件的更改直接训练


## 优化
1. IOU计算优化，使用Rotated Box
2. DCN offset的监督调整，对于Rotated box 可以添加约束
3. FPN融合方式更改


## Other

Reference Code: https://github.com/aws-samples/mask-rcnn-tensorflow  
issue: https://github.com/tensorflow/tensorflow/issues/32383