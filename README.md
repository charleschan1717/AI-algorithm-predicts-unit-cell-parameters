<div align="center">

<img width="1036" height="358" alt="image" src="https://github.com/user-attachments/assets/4942fb82-f4fd-4ee4-814b-e60d8b9625e4" />


# CrystalNeXtT: AI-Driven XRD Analysis

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)](https://pytorch.org/)
[![Award](https://img.shields.io/badge/AICOMP_2025-National_2nd_Place-gold)](https://github.com/)


**2025 AICOMP 挑战赛全国第二名解决方案**

[数据集下载] • [文档] 

</div>

---

## 📖 项目简介

本项目开源了 **CrystalNeXtT** 算法源码，这是 **2025 AICOMP 挑战赛的全国第二名解决方案**。

赛题旨在解决粉末 X 射线衍射（XRD）图谱分析中的“指标化”难题。算法需要从含有噪声、杂峰及仪器误差的原始图谱中，**直接预测晶胞参数并标定晶面指数**，以辅助新材料的结构解析。

## 🧗‍♂️ Main Challenges (核心难点)

赛题数据模拟了真实的实验环境，包含以下关键难点：

* **仪器系统误差**：存在随机的零点漂移和样品偏移。
* **复杂干扰**：数据包含低信噪比、背景衰减及随机添加的杂质峰。
* **高精度要求**：
    * 评分公式：`Score = 0.9 * Accuracy - 0.1 * RMSE`
    * 晶面匹配误差阈值仅为 $0.2^\circ$。

## 💡 CrystalNeXtT Solution

针对上述难点，我们提出了以下技术方案：

1.  **混合模型架构 (Hybrid Architecture)**
    * 采用 **ConvNeXt-1D** 提取局部峰形特征。
    * 结合 **Transformer Encoder** 捕捉全局衍射模式，有效建立了图谱到晶胞参数的非线性映射。

2.  **物理感知损失 (Physics-Aware Loss)**
    * 引入基于 **Chamfer Huber Loss** 的倒角距离损失。
    * 在训练中根据预测参数动态模拟理论，直接优化预测谱图与真实谱图的匹配度，而非仅依赖参数回归。

3.  **测试时增强 (TTA)**
    * 针对数据集中存在的仪器漂移，在推理阶段对输入谱图进行多尺度线性平移并取平均，显著降低了零漂和样偏的影响。

4.  **角度吸附 (Angle Snapping)**
    * 针对晶体对称性，设计分类头预测角度是否为 $90^\circ$ 或 $120^\circ$。
    * 高置信度下强制吸附至精确值，大幅提升了正交、四方及六方晶系的预测精度。

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
