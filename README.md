# 深度学习中的最优化项目仓库

## 前情提要

项目选题为"7 Transformer Pretraining"，介绍如下

> 当今 Transformer 模型已成为众多大模型的首选架构，详细了解 Transformer 的结构细节有助于我们更好理解大模型的运作机理
> 该项目需要同学们从零搭建一个小规模 Transformer 模型，并训练其完成简单的任务。

任务要求：

- 了解基于传统 Transformer 架构的改进方案（如'非线性激活函数的选取'[R7-1](https://arxiv.org/abs/2002.05202)、'残差链接的位置'[R7-2](https://arxiv.org/abs/2002.04745)、相对位置编码[R7-3](https://arxiv.org/abs/1803.02155)、稀疏注意力机制[R7-4](https://arxiv.org/abs/1904.10509)等），并探究其改进的效果。

- 使用 pytorch 库搭建一个简易的 Transformer 模型（可根据自己的算力资源选择合适的模型大小），可考虑使用上述的改进方式。随后训练其完成简单的任务，以下是一个推荐的可能比较好实现的任务：使用编码器+解码器的 Transformer 架构，自行准备若干古诗句作为训练语料，训练的目标是使模型能够根据一句古诗已有的前半句自由书写出后半句。模型性能不作为主要评分依据，鼓励使用 pytorch 单元模块逐步搭建而非直接使用集成的 Transformer 模块。

- 如有余力，考虑将该模型封装为一个古诗写作助手。注意自回归生成的实现方式，以及如何做到古诗的押韵

## 代码介绍

1、项目树状图，其中有一些因为文件大小因素没有上传到仓库里面:

```
.
├── LICENSE
├── README.md
├── datasets
│   ├── test
│   └── train
│       └── poet.song.0.json
├── models
├── nets
│   ├── __pycache__
│   │   └── transformer.cpython-312.pyc
│   └── transformer.py
├── plugins
│   └── bert-base-chinese
│       ├── README.md
│       ├── config.json
│       ├── flax_model.msgpack
│       ├── model.safetensors
│       ├── pytorch_model.bin
│       ├── tf_model.h5
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── predict.py
├── tokenizerTest.py
├── train.py
└── utils
    ├── __pycache__
    │   └── dataloader.cpython-312.pyc
    ├── dataloader.py
    └── utils.py
```

2、主要文件夹：

- `datasets` 里面存放从仓库[chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)上下载的 `json` 文件。

- `models`里面存放模型的参数文件，使用`torch.save`储存。

- `nets`里面实现了一个从`pytorch`开始的`transformer`。

- `plugins`里面放着一个`Bert`的`tokenizer`，原仓库在[bert-base-chinese](https://hf-mirror.com/google-bert/bert-base-chinese/tree/main)

- `utils`里面主要实现了一个`pytorch`的`dataloader`，这部分与具体数据集对接，包含数据预处理操作。

- 根目录下，`train.py`用于训练，`predict.py`用于预测，`tokenizerTest.py`用于测试`tokenizer`

3、代码使用:

使用`python3 train.py`即可开始训练，具体的参数可以参考相应代码。

使用`python3 predict.py`开始预测模式，参数`--model`指定使用的模型参数，默认为测试时使用的参数。
