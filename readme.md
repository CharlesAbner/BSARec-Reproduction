# BSARec Reproduction & Ablation Study based on ReChorus

本项目为课程大作业代码仓库。基于 [ReChorus](https://github.com/THUwangcy/ReChorus) 框架，复现了 **BSARec (Block Self-Attention for Sequential Recommendation)** 模型，并在 MovieLens-1M 和 Grocery 数据集上进行了复现，后续在MovieLens-1M上进行消融实验与超参数敏感性分析。

## 📌 项目简介

我们基于 ReChorus 的 `SequentialModel` 基类重新实现了 `BSARec.py`，并对框架进行了以下关键适配与改进：
1.  **接口对齐**：修改了部分接口以适配 ReChorus 框架的数据流。
2.  **逻辑修正**：修改了 `ML_1M.ipynb` 的预处理逻辑，以严格对齐 BSARec 原论文的数据处理标准。
3.  **损失函数优化创新**：原论文使用 Cross Entropy (CE) Loss，本项目将其调整为 **BPR Loss**。
    * *动机*：将模型优化目标从“全类目分类”转变为“正负样本对的相对排序”，在实验中表现出更稳定的收敛性。

## 🛠️ 运行环境 (Environment)

实验依托 **AutoDL 云算力平台** 进行，具体配置如下：

* **硬件配置**：
    * GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
    * CPU: Intel(R) Xeon(R) Platinum 8470Q (20 vCPU)
    * RAM: 90GB
* **软件环境**：
    * OS: Ubuntu 22.04
    * Python: 3.10
    * PyTorch: 2.1.0
    * CUDA: 12.1

## 📂 目录结构

```text
.
├── ReChorus/                  # 推荐系统基础框架 (含源码 src/)
├── run_bsarec_experiments.py  # [核心] BSARec 主实验脚本 (含 Baseline, Ablation, Sensitivity)
├── run_sasrec_search.py       # SASRec 对比实验脚本
├── run_gru4rec_search.py      # GRU4Rec 对比实验脚本
├── result_to_picture.py       # 实验结果可视化绘图脚本
├── *.csv                      # 实验结果汇总日志
├── requirements.txt           # 项目依赖
├── output_work.md             # 实验报告
└── README.md                  # 项目说明
````

## 🚀 快速开始 (Quick Start)

### 1\. 安装依赖

请确保安装了匹配 CUDA 版本的 PyTorch。

```bash
pip install -r requirements.txt
```

### 2\. 数据准备

本项目 **已集成** 对齐 BSARec 原论文标准的数据预处理逻辑（基于我们修改后的数据接口与 `ML_1M.ipynb`）。

* **无需手动清洗**：上传的代码已包含完整的数据处理流程。
* **数据集放置**：仅需确保原始的 MovieLens-1M 或 Grocery 数据集文件位于 `ReChorus/data/` 目录下。
* **自动加载**：运行实验脚本时，模型将自动调用适配后的接口读取并处理数据。


### 3\. 运行实验

本项目提供了一键复现脚本，支持直接运行：

**BSARec 完整实验 (复现 + 消融 + 敏感性分析):**

```bash
python run_bsarec_experiments.py
```

*该脚本将自动执行：Best Param Baseline, Ablation (w/o SA, w/o AIB), Sensitivity (Alpha, C).*

**运行对比模型 (SASRec / GRU4Rec):**

```bash
python run_sasrec_search.py
python run_gru4rec_search.py
```

### 4\. 结果可视化

实验结束后，运行绘图脚本即可在 `plots/` 目录下生成对比图表：

```bash
python result_to_picture.py
```

## ⚙️ 实验参数设置 (Hyperparameters)

基于论文最优配置及网格搜索，最终确定的核心参数如下：

### BSARec @ ML-1M 最佳参数配置

| 参数 (Parameter) | 值 (Value) | 说明 (Source/Note) |
| :--- | :--- | :--- |
| **Embedding Size ($D$)** | 64 | `emb_size`: 论文标准设置 |
| **Num Layers ($L$)** | 2 | `num_layers`: 堆叠层数 |
| **Num Heads ($H$)** | **4** | `num_heads`: **ML-1M 特有配置** (代码指定为4) |
| **Alpha ($\alpha$)** | **0.3** | `alpha`: BSARec 核心参数 (ML-1M 最优) |
| **Block Size ($c$)** | **9** | `c`: 上下文窗口大小 (ML-1M 最优) |
| **Learning Rate** | 0.0005 | `lr`: 5e-4 |
| **Batch Size** | 256 | `batch_size`: 训练批次大小 |
| **L2 Regularization** | 1e-6 | `l2`: 正则化系数 |
| **Loss Function** | **BPR** | 改进创新后的损失函数 (原论文为 CE) |

### SASRec @ ML-1M 最佳参数配置

| 参数 (Parameter) | 值 (Value) | 说明 (Source/Note) |
| :--- | :--- | :--- |
| **Embedding Size ($D$)** | 64 | `emb_size`: 嵌入向量维度 |
| **Num Layers ($L$)** | 2 | `num_layers`: Transformer 层数 |
| **Num Heads ($H$)** | **1** | `num_heads`: **网格搜索最优结果** |
| **Learning Rate** | **0.001** | `lr`: 1e-3 (**网格搜索最优结果**) |
| **Batch Size** | 256 | `batch_size`: 训练批次大小 |
| **History Max ($N$)** | 50 | `history_max`: 最大序列长度 |
| **L2 Regularization** | 1e-6 | `l2`: 正则化系数 |
| **Loss Function** | **BPR** | 强制使用 BPR (对齐 BSARec 设置) |

### GRU4Rec @ ML-1M 最佳参数配置

| 参数 (Parameter) | 值 (Value) | 说明 (Source/Note) |
| :--- | :--- | :--- |
| **Embedding Size ($D$)** | 64 | `emb_size`: 同时作为 GRU 隐层维度 (Hidden Size) |
| **Num Layers ($L$)** | 2 | `num_layers`: GRU 堆叠层数 |
| **Learning Rate** | **0.0005** | `lr`: 5e-4 (**网格搜索最优结果**) |
| **Batch Size** | 256 | `batch_size`: 训练批次大小 |
| **L2 Regularization** | 1e-6 | `l2`: 正则化系数 |
| **History Max ($N$)** | 50 | `history_max`: 最大序列长度 |
| **Loss Function** | **BPR** | 使用 BPR (为保证公平对比，对齐 BSARec 设置) |



## 📊 实验结果

代码运行后将自动生成以下日志文件：

  * `bsarec_experiment_results.csv`: 记录 BSARec 所有变体的详细指标。
  * `sasrec_tuning_results.csv`: SASRec 对照组结果。
  * `gru4rec_tuning_results.csv`: GRU4Rec 对照组结果。

## 🔗 参考引用

* **ReChorus Framework**: [Wang et al., "ReChorus: A Comprehensive Recommender System Framework"]
* **BSARec Paper**: [Ren et al., "Block Self-Attention for Sequential Recommendation", CIKM 2023]

---
**Author**: [李子康,鞠阳]

**Course**: [中山大学人工智能学院机器学习课程]


<!-- end list -->

