<img width="1036" height="358" alt="image" src="https://github.com/user-attachments/assets/4942fb82-f4fd-4ee4-814b-e60d8b9625e4" />
<img width="241" height="110" alt="image" src="https://github.com/user-attachments/assets/7e7fdc91-244b-4e52-aeb2-ad296d9af2b0" />


#Introduction
本项目开源了 CrystalNeXtT 算法源码，这是 2025 AICOMP 挑战赛的全国第二名的解决方案。

赛题旨在解决粉末 X 射线衍射（XRD）图谱分析中的“指标化”难题。算法需要从含有噪声、杂峰及仪器误差的原始图谱中，直接预测晶胞参数并标定晶面指数，以辅助新材料的结构解析。

#Main Challenges
赛题数据模拟了真实的实验环境，包含以下关键难点：
仪器系统误差：存在随机的零点漂移和样品偏移
复杂干扰：数据包含低信噪比、背景衰减及随机添加的杂质峰。
高精度要求：评分公式为 Score = 0.9Accuracy - 0.1RMSE，且晶面匹配误差阈值仅为0.2°。

#CrystalNeXtT-Solution：
混合模型架构：采用 ConvNeXt-1D 提取局部峰形特征，结合 Transformer Encoder 捕捉全局衍射模式，有效建立了图谱到晶胞参数的非线性映射。
物理感知损失 (Physics-Aware Loss)：引入基于 Chamfer Huber Loss 的倒角距离损失。在训练中根据预测参数动态模拟理论，直接优化预测谱图与真实谱图的匹配度，而非仅依赖参数回归。
测试时增强 (TTA)：针对数据集中存在的仪器漂移，在推理阶段对输入谱图进行多尺度线性平移并取平均，显著降低了零漂和样偏的影响。
角度吸附 (Angle Snapping)：针对晶体对称性，设计分类头预测角度是否为 $90^\circ$ 或 $120^\circ$。高置信度下强制吸附至精确值，大幅提升了正交、四方及六方晶系的预测精度。

#数据集下载
本比赛使用的数据集包含 7000 条模拟 Cu 靶 XRD 图谱（.xy 格式）及对应的标注信息（.json 格式），源自 Crystallography Open Database (COD)。https://www.crystallography.net/cod/
下载链接: https://drive.google.com/file/d/1OqjqJCdUYJEuhitbAl7pUDtIDnGLZg7H/view?usp=drive_link
下载后请解压至 data/ 目录，结构如下(仅给出1000+数据集，可在COD官网自行爬取数据集，可自定义，可自行分配数据集)：

data/
├── train/          # 训练集 .xy 和 .json 文件
└── test/           # 测试集 .xy 文件

1. 环境依赖

pip install numpy torch scipy tqdm joblib

2. 数据预处理

使用 preprocess.py 将原始 .xy 文本数据转换为高效的 .npz 格式，并自动执行 AsLS 基线校正与 Savitzky-Golay 平滑。

  2.1处理训练集 (需包含 .json 标签)
python preprocess.py --input ./data/train --output data/train_processed.npz --mode train

  2.2处理测试集 (仅 .xy 文件)
python preprocess.py --input ./data/test --output data/test_processed.npz --mode test


3. 模型训练

运行 train.py 开始训练。代码会自动保存验证集分数（Score）最高的模型权重。
注：训练过程中会自动进行动态模拟以计算物理损失。

python train.py

4. 推理与打包

使用 predict.py 对测试集进行预测。该脚本包含 TTA 和 角度吸附 逻辑，并按赛题要求生成提交所需的 .json 结果文件。

python predict.py \
  --input data/test_processed.npz \
  --model_path unicrystal_best_by_score.pt \
  --output results/ \
  
5.打分
用score代码模拟平台打分。

Author: Chen Dapei
Affiliation: Hunan University of Technology and Business
