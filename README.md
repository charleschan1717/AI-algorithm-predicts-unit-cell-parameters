<div align="center">

<img width="1036" height="358" alt="image" src="https://github.com/user-attachments/assets/4942fb82-f4fd-4ee4-814b-e60d8b9625e4" />


# CrystalNeXtT: AI-Driven XRD Analysis

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)](https://pytorch.org/)
[![Award](https://img.shields.io/badge/AICOMP_2025-National_2nd_Place-gold)](https://github.com/)


**2025 AICOMP 挑战赛全国第二名解决方案**
 

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




没问题，这是为您精心打磨的 **README “数据预处理” 模块**。

这段内容采用了**“学术级摘要 + 顶级期刊风格图表”**的组合，既展示了您对信号处理算法的深刻理解，又通过可视化直观证明了数据质量的提升。您可以直接复制粘贴到您的 Markdown 文件中。

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
## 🧠 模型框架Framework: The CrystalNeXtT Framework

本项目提出了 **CrystalNeXtT**，一种**物理感知的混合深度学习架构 (Physics-Aware Hybrid Architecture)**。针对 XRD 图谱数据的序列特性与晶体学约束，模型采用了 **"Local-to-Global"** 的设计范式，结合了ConvNeXt的局部特征提取能力与Transformer的长程全局关联能力。

### 1 Overview

模型处理流程包含三个核心阶段：**多尺度特征编码 (Multi-Scale Encoding)**、**全局上下文建模 (Global Context Modeling)** 以及 **物理约束解码 (Physics-Constrained Decoding)**。
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/24c04b95-f7d2-4ffd-a00d-680116c8bf3e" />

### 2 核心技术创新

#### 🔹 1. Multi-Dilated ConvNeXt Encoder (多空洞卷积编码器)
针对 XRD 衍射峰**半峰宽 (FWHM)** 变化剧烈（受晶粒尺寸和结晶度影响）的特点，我们在 ConvNeXt 模块中引入了 **Multi-Dilated Depth-wise Convolution**。
* **Implementation**: 每一个卷积块并行包含膨胀率为 $d=[1, 3, 5]$ 的三个分支。
* **Significance**: 这使得单个神经元具备了**自适应感受野**，既能捕捉尖锐的结晶峰，也能有效提取宽化的漫散射信号。此外，引入 **ECA (Efficient Channel Attention)** 模块进一步增强了对特征指纹通道的敏感度，通过对不同位置来加入注意力模块来进行调优。

#### 🔹 2. Geometric Angle Gating (晶体几何门控机制)
这是本模型的**核心创新点**。在晶体学中，正交晶系 ($90^\circ$) 和六方晶系 ($120^\circ$) 具有严格的几何约束。普通的回归模型极易预测出 $89.9^\circ$ 或 $90.1^\circ$ 的数值误差，导致晶系判定错误。
* **Mechanism**: 我们设计了一个并行分支，同时进行**角度值回归 (Regression)** 和 **角度类别分类 (Classification)**。
* **Formulation**: 最终输出由门控概率加权决定：<img width="286" height="25" alt="image" src="https://github.com/user-attachments/assets/18c3ae88-51b5-43ce-9acf-9c50e5644f4c" />
* **Effect**: 这种设计作为一种**归纳偏置**，强制模型在面对高对称性晶体时精确收敛至理论值。

#### 🔹 3. Systematic Error Decoupling (系统误差显式解耦)
为了应对真实实验中的仪器误差，模型不仅仅预测晶胞参数，还设有独立的 **Error Estimator Head**。
* **Zero Shift ($\Delta 2\theta_0$)**: 预测由于检测器校准不准导致的整体偏移。
* **Sample Displacement ($s$)**: 基于物理公式预测样品高度误差：<img width="119" height="38" alt="image" src="https://github.com/user-attachments/assets/205e6478-31d8-4c45-87f7-741f76b74d41" />
* **Calibration**: 这使得模型具备了“自动校准”能力，将环境干扰从晶体结构特征中剥离出来。

### 3 以物理为导向的训练方法

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
