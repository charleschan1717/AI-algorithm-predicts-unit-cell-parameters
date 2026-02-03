<div align="center">

<img width="1036" height="358" alt="image" src="https://github.com/user-attachments/assets/4942fb82-f4fd-4ee4-814b-e60d8b9625e4" />


# CrystalNeXtT: AI-Driven XRD Analysis

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)](https://pytorch.org/)
[![Award](https://img.shields.io/badge/AICOMP_2025-National_2nd_Place-gold)](https://github.com/)


**2025 AICOMP 挑战赛"新材料赛道"全国第二名解决方案**
 

</div>

---
## 📖 项目任务 (Project Tasks)
<div align="center">
  <img width="457" height="422" alt="image" src="https://github.com/user-attachments/assets/5576f33d-5544-4da1-9247-629970576f16" />
</div>


## 📖 项目背景 (Project Background)

**新材料研发的难点：晶体结构解析**

新材料的发现是推动半导体、新能源及航空航天产业发展的核心动力。在材料科学中，X射线粉末衍射（XRD）是解析材料晶体结构最关键的技术手段，超过 90% 的新型功能材料依赖该技术进行结构确认 。

然而，XRD 数据分析的第一步——指标化（Indexing），即从一维的衍射峰序列反推三维晶胞参数并标定晶面指数，是解析过程中最具挑战性的瓶颈 。传统的指标化算法（如 ITO, TREOR 等）通常依赖于高质量的数据，在面对以下情况时往往失效：

* **低对称性晶体**：晶胞参数复杂，搜索空间巨大。
* **多相混合与杂质**：未知相中常伴随杂质峰，导致传统算法无法收敛。
* **实验非理想性**：仪器分辨率限制和制样误差导致的峰位漂移 。



本项目 **CrystalNeXtT** 旨在利用深度学习技术解决这一经典的晶体学难题，实现从原始图谱到晶体学常数的端到端预测。

## 🚀 项目意义 (Significance & Motivation)

本项目属于典型的 **AI for Science (AI4S)** 交叉学科应用，其科学意义与应用价值体现在以下三个维度：

1. **解决非线性反问题**：
晶胞参数与衍射角度之间存在高度非线性的映射关系，且存在复杂的峰重叠现象。本项目探索了深度神经网络在求解此类高维非线性反问题上的潜力，证明了数据驱动方法在晶体学领域的有效性 。


2. **鲁棒性与泛化能力**：
与仅适用于理想数据的传统算法不同，本项目针对真实实验中常见的零点漂移、样品偏移及背景噪声进行了专门的建模与训练 。这使得算法能够直接处理未经人工精修的原始数据，大幅降低了科研门槛。


3. **加速材料基因组工程**：
通过实现高精度的自动化指标化，本算法有望替代繁琐的人工解析流程，为高通量材料筛选（High-Throughput Screening）和大规模晶体数据库的构建提供核心算法支持，加速新材料的研发周期 。



## 🧗‍♂️ 核心难点 (Main Challenges)

本赛题的数据集由 *Crystallography Open Database* 真实数据模拟生成，高度还原了实验环境中的复杂干扰，对算法提出了极高的鲁棒性要求 。

### 1. 仪器系统误差的解耦 (Systematic Error Decoupling)

真实 XRD 图谱中普遍存在系统性偏差，模型必须学习在存在偏差的情况下推断真实的晶体结构：

* **零点漂移 (Zero Drift)**：随机偏移范围  。

* **样品偏移 (Sample Displacement)**：测角仪半径 200mm 下的偏移量  。

* *难点*：微小的角度偏移会导致高角度衍射峰的显著移动，严重干扰晶胞参数的回归精度。

### 2. 复杂噪声环境下的特征提取 (Feature Extraction in Noisy Environments)

数据信噪比极低，且包含随机干扰，模型需具备极强的抗干扰能力：

* **杂质干扰**：部分晶系随机添加 1-2 个杂质峰，且无标注，模型需自动识别并剔除“伪峰” 。

* **背景与噪声**：叠加了指数衰减背景及随机噪声，导致弱峰难以识别 。


* *难点*：如何在强噪声和杂峰干扰下，精准捕捉属于目标相的有效衍射特征。

### 3. 极高精度的评价体系 (High-Precision Metrics)

赛题对预测精度的要求极为严苛，迫使模型必须收敛至全局最优解而非局部最优：

* **评价公式**：<img width="368" height="35" alt="image" src="https://github.com/user-attachments/assets/cae530c9-bce8-4b70-84c9-ff15a1223fec" />


* **晶面匹配阈值**：仅允许的角度误差。即便晶胞参数预测略有偏差，也会导致高阶晶面指数匹配失败，从而导致 Accuracy 断崖式下跌 。
<img width="819" height="130" alt="image" src="https://github.com/user-attachments/assets/12039db3-b322-42b5-bef2-4b516ce9efb5" />



* *难点*：必须同时优化回归任务（晶胞参数 RMSE）和分类/匹配任务（晶面指数 Accuracy），平衡两者的损失函数权重是优化的关键。

### 4. 输出要求(Output Request)
<img width="448" height="41" alt="image" src="https://github.com/user-attachments/assets/2814418a-cd44-4f8d-9219-9351a3211cac" />

<img width="458" height="224" alt="image" src="https://github.com/user-attachments/assets/c50399d8-41a3-4a23-9f6b-500ac5cba799" />

---

## ⚙️ 数据预处理(Data Preprocessing)

针对 XRD 原始图谱中普遍存在的**非线性背景干扰**、**随机高频噪声**及**采样率不一致**问题，本项目设计了一套高效的并行信号处理流水线。流程首先采用 **Asymmetric Least Squares (AsLS)** 算法构建稀疏差分矩阵，自适应地拟合并不扣除指数衰减背景；随后应用 **Savitzky-Golay 滤波器** 进行保形平滑去噪；最后通过**统一网格重采样 (Uniform Grid Resampling)** 将非结构化数据线性映射为长度固定的标准化张量。该策略在保留晶体学关键特征（峰位、相对强度、FWHM）的同时，实现了数据的高效降维与归一化，显著加速了模型收敛。

<div align="center">
  <img width="3048" height="2374" alt="image" src="https://github.com/user-attachments/assets/91b7d256-491c-424d-b15b-6fa88feaa954" />

  <br>
  <div style="color: #555; font-size: 14px; width: 90%; text-align: left;">
    <strong>Figure 1. Visualization of the signal processing pipeline.</strong>
    <br>
    <strong>(a)</strong> 原始 XRD 图谱（灰色）与通过非对称最小二乘法（AsLS）拟合的自适应基线（红色虚线），直观展示了复杂背景的消除过程。
    <br>
    <strong>(b)</strong> 经过基线扣除、Savitzky-Golay 去噪、Min-Max 归一化及统一网格重采样后的最终模型输入信号，显著提升了信噪比 (SNR)。
  </div>
</div>

* **横坐标 ($2\theta$)**：代表衍射角。波峰在横轴上的**位置**由晶格常数 ($a, b, c$) 决定（遵循布拉格定律 $n\lambda = 2d\sin\theta$）。这是模型进行**晶胞参数回归**任务的核心依据。
* **纵坐标 (Intensity)**：代表衍射强度。波峰的相对**高度**包含原子排布信息，对于**晶面指数标定 ($hkl$)** 至关重要。
* **波峰 (Diffraction Peaks)**：每一个尖锐的波峰即布拉格反射对应晶体内部的一个特定晶面。**CrystalNeXtT** 的核心挑战在于：即便在仪器误差导致波峰发生偏移情况下，依然能精准地从这一维序列反推出三维晶体结构。

---
## 🧠 模型框架：The CrystalNeXtT Framework

本项目提出了 **CrystalNeXtT**，一种**物理感知的混合深度学习架构 (Physics-Aware Hybrid Architecture)**。针对 XRD 图谱数据的序列特性与晶体学约束，模型采用了 **"Local-to-Global"** 的设计范式，结合了ConvNeXT的局部特征提取能力与 Transformer 的全局上下文关联能力。

### Architecture Overview (架构总览)

模型处理流程分为三个核心阶段：**多尺度特征编码 (Multi-Scale Encoding)**、**全局上下文建模 (Global Context Modeling)** 以及 **物理约束解码 (Physics-Constrained Decoding)**。

<div align="center">
  <img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/8b210973-89a3-4eef-b83f-f9513bda1b32" />

  <br>
  <div style="color: #555; font-size: 14px; width: 90%; text-align: center;">
    <strong>Figure 2. The CrystalNeXtT Architecture.</strong> The pipeline consists of (1) a Multi-Scale CNN Encoder for local feature extraction, (2) a Transformer Encoder for global context modeling via the [CLS] token, and (3) Physics-Constrained Prediction Heads for decoupling crystal parameters from systematic errors.
  </div>
</div>

<br>

### Detailed Workflow (详细流程)

#### 🔹 Stage 1: Multi-Scale CNN Encoder (ConvNeXt-1D)
**目标：提取局部波形特征**
输入为经过预处理的 1D 张量 $(B, 1, 10000)$。由于 XRD 衍射峰的**半峰宽** 受晶粒尺寸和结晶度影响变化剧烈，单一尺度的卷积核难以同时捕捉尖锐的结晶峰和宽化的漫散射信号。
* **Stem Downsampling**: 通过 3 层大步长卷积 (Stride=2,2,4) 将高维输入快速映射至潜在特征空间。
* **Multi-Dilated ConvNeXt Block**: 每一个卷积块内部并行包含三个分支，膨胀率分别为 $d=[1, 3, 5]$。
    * *自适应感受野*：小膨胀率捕捉尖锐峰，大膨胀率提取背景与宽化峰。
    * *ECA Attention*：引入高效通道注意力(Efficient Channel Attention)，增强模型对特定衍射指纹通道的敏感度。

#### 🔹 Stage 2: Global Context Modeling (Transformer Encoder)
**目标：建立长距离依赖并且信息聚合**
XRD 图谱中低角度峰与高角度峰存在严格的几何拓扑关系（由同一套晶胞参数决定）。CNN 擅长局部特征，而 Transformer 擅长全局交互。
* **[CLS] Token 机制**: 我们在 CNN 输出的特征序列前拼接了一个可学习的 **[CLS] Token**。
* **Self-Attention**: 经过 4 层 Transformer Encoder 的自注意力机制交互，全图的晶体学信息最终被“压缩”并汇聚到这个 **[CLS] Token** 中。
* **Output**: 最终仅截取 [CLS] Token 的特征向量 $(B, C)$ 作为后续所有预测任务的输入，实现了高效的信息解耦。

#### 🔹 Stage 3: Multi-Task Prediction Heads (Physics-Constrained Decoding)
**目标：物理参数解耦与预测**
基于聚合了全局信息的 [CLS] Token，模型通过多个并行的 MLP 头输出物理参数，并引入几何门控机制。

**1. Lattice & Angle Gating (晶胞参数与角度门控)**
针对高对称性晶系（如立方、六方）的“伪对称”难题，我们设计了 **Geometric Angle Gating** 机制：
* **Angle Gate Head**: 将角度分类为 $\{90^\circ, 120^\circ, \text{Free}\}$ 三种概率 $P$。
* **Lattice Head**: 回归原始的角度数值 $\alpha_{raw}$。
* **物理约束公式**:

$$
\hat{\theta}_{final} = P_{90} \cdot 90^\circ + P_{120} \cdot 120^\circ + P_{free} \cdot \alpha_{raw}
$$
  
  该机制强制模型在面对正交或六方晶系时，精确收敛至理论值 ($90^\circ/120^\circ$)。

**2. Systematic Error Decoupling (系统误差显式解耦)**
为了应对真实实验中的干扰，**Error Head** 独立预测两项关键仪器误差，赋予模型“自动校准”能力：
* **Zero Shift ($\Delta 2\theta_0$)**: 预测检测器零点漂移。
* **Sample Displacement ($s$)**: 基于物理公式预测样品高度误差：
  
$$
\Delta 2\theta \approx -\frac{2s}{R}\cos\theta
$$

**3. HKL Prediction (晶面指数预测)**
**HKL Head** 输出多热编码 (Multi-hot) 向量，辅助模型理解衍射峰对应的晶面指数，进一步辅助晶胞参数的收敛。

### 以物理为导向的训练方法

#### 📉 Hybrid Loss Function (复合损失函数)
模型通过多任务学习(Multi-Task Learning)进行联合优化：

$$
\mathcal{L}_{total} = \lambda_{cell}\mathcal{L}_{MSE} + \lambda_{tth}\mathcal{L}_{Chamfer} + \lambda_{gate}\mathcal{L}_{CE} + \lambda_{error}\mathcal{L}_{L1}
$$

* **$\mathcal{L}_{Chamfer}$**: 在 $2\theta$ 空间计算预测谱图与真实谱图的倒角距离，直接优化物理观测量的吻合度。
* **$\mathcal{L}_{gate}$**: 监督角度门控分类器，确保晶系对称性的正确识别。

#### 🎲 Domain Randomization (域随机化)
为了弥补模拟数据与真实数据的分布差异，我们在训练循环中在线应用基于物理原理的数据增强：
* **Impurity Injection**: 随机生成高斯杂质峰，迫使模型学习区分“结构信号”与“杂质噪声”。
* **Peak Broadening**: 随机卷积模拟晶粒尺寸细化效应。
* **Elastic Displacement**: 动态注入随机的零点漂移 ($\pm 0.5^\circ$) 和样品偏移 ($\pm 0.2mm$)。


---




## 📂 数据集与环境

### 1. 数据集下载
本比赛使用的数据集包含 7000 条模拟 Cu 靶 XRD 图谱（`.xy` 格式）及对应的标注信息（`.json` 格式），源自 [Crystallography Open Database (COD)](https://www.crystallography.net/cod/)。

🔗 **下载链接:** [Google Drive Link](https://drive.google.com/file/d/1OqjqJCdUYJEuhitbAl7pUDtIDnGLZg7H/view?usp=drive_link)

下载后请解压至 `data/` 目录，目录结构如下（仅给出部分示例，可自行分配或在 COD 官网爬取）：

```text
data/
├── train/          # 训练集 .xy 和 .json 文件
└── test/           # 测试集 .xy 文件
preprocess.py
train.py
predict.py
score.py
