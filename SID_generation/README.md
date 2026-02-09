# RQVAE 项目

## 项目简介

本项目实现了RQVAE（Residual Quantized Variational Autoencoder）模型，用于图像和视频的特征编码与解码。模型支持多层残差量化，适用于大规模数据训练。

## 环境依赖

- Python 3.8+
- PyTorch 1.10+
- 其他依赖项请参考 `requirements.txt`

## 安装步骤

1. 克隆项目代码：
   ```bash
   git clone <项目地址>
   cd SID_generation
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 数据准备
除了使用 `wget` 命令下载，你也可以直接从 Hugging Face 上的 `AL-GR/Item-EMB` 数据集中获取这些文件。
数据集链接：[https://huggingface.co/datasets/AL-GR/Item-EMB](https://huggingface.co/datasets/AL-GR/Item-EMB)

   ```bash
   wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/reconstruct_data_mask.npz
   wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/contrastive_data_mask.npz
   ```


## 训练模型
使用以下命令启动分布式训练：
   ```bash
   python -m torch.distributed.launch --nnodes=2 --nproc_per_node=1 --master_port=27646 train.py --output_dir=/path/to/output --save_prefix=MODEL_NAME --cfg=configs/rqvae_i2v.yml
   torchrun --nnodes=1 --nproc_per_node=8 train.py --cfg=configs/rqvae_i2v.yml
   ```

## 参数说明

- `--cfg`：配置文件路径。
- `--output_dir`：模型输出目录。
- `--save_prefix`：模型保存前缀。

## 测试模型

使用以下命令启动测试：

```bash
python infer_SID.py
```

## 项目结构

```
open_RQVAE/
├── configs/            # 配置文件
├── datas/              # 数据处理模块
├── rqvae_embed/        # RQVAE模型核心代码
├── utils/              # 工具函数
├── train.py            # 训练脚本
└── requirements.txt    # 依赖列表
```
