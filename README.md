# NBA 球员数据分析与机器学习项目 (nba-analytics)

本研究项目旨在对 NBA 历史与现役球员数据进行深度清洗、描述性统计分析、聚类挖掘及机器学习预测，并最终通过交互式 Web 看板（Streamlit）进行综合可视化呈现。

---

## 团队分工

本项目采用按功能模块与技术栈拆分的协作架构，5 位成员各司其职，通过标准的上下游数据与代码流紧密配合：

#### 1. 数据清洗
*   **核心职责**：负责多源数据的标准化清洗、跨表主键对齐、重名处理与主数据集（Master Dataset）构建。
*   **具体任务**：
    *   加载并校验 `Player.csv`、`Seasons_Stats.csv` 和 `player_data.csv` 原始数据。
    *   处理缺失值、异常极值、同名不同人及历史时代数据归一化。
    *   按照 `Player` + `Year` 等维度打通跨表关联，构建并派生真实命中率（TS%）、每36分钟折算等核心特征。
*   **关键交付**：`src/data_cleaning.py`（清洗脚本）与 `data/processed/master_data.csv`（统一标准数据集）。

#### 2. 描述性统计与 EDA 可视化分析 
*   **核心职责**：基于主数据集开展描述性统计计算，挖掘数据分布特征，产出多维度高质感静态/动态图表。
*   **具体任务**：
    *   计算各统计指标的均值、中位数、标准差及分位数，绘制描述性统计汇总表。
    *   探索得分分布、出场时间与效率值（PER）的 Pearson 相关性热力图。
    *   按场上位置（Pos）、薪资梯队或选秀时代进行对比分析（箱线图、小提琴图）。
*   **关键交付**：`notebooks/01_eda.ipynb`（分析过程）与 `src/visualization.py`（复用绘图函数）。

#### 3. 无监督学习与聚类挖掘
*   **核心职责**：打破传统 5 个标称位置的局限，运用降维与聚类算法挖掘现代化“无位置篮球”下球员的隐性战术定位。
*   **具体任务**：
    *   使用 PCA / t-SNE 对高维统计特征（如 3PAr, FTr, AST%, BLK% 等）进行降维与二维空间展示。
    *   构建 K-Means / 层次聚类模型，利用肘部法则（Elbow Method）与轮廓系数（Silhouette Score）确定最佳 K 值。
    *   对聚类结果进行战术语义画像（如定义“3D外线精英”、“组织型中锋”、“高使用率单打手”等）。
*   **关键交付**：`notebooks/02_clustering.ipynb`（聚类探索）与各簇球员特征对比雷达图。

#### 4. 监督学习与预测建模
*   **核心职责**：构建回归与分类机器学习模型，对球员的能力评估、未来技术表现或荣誉预测进行建模与解释。
*   **具体任务**：
    *   设定预测目标（如：基于历史数据预测球员下赛季 PER 效率值，或预测是否入选全明星 All-Star）。
    *   使用 Random Forest、XGBoost、LightGBM 等算法进行训练、交叉验证与超参数调优。
    *   引入 SHAP (SHapley Additive exPlanations) 值评估特征重要性，解释单一球员的能力归因。
*   **关键交付**：`notebooks/03_prediction.ipynb`（建模过程）与 `saved_models/*.pkl`（导出的模型权重）。

#### 5. Web 前端看板开发与项目整合
*   **核心职责**：搭建交互式 Web 动态看板，整合全队数据、图表与模型，负责最终成果的呈现与项目汇总。
*   **具体任务**：
    *   基于 Streamlit 框架搭建前端大屏，提供球员/赛季/球队的动态筛选功能。
    *   嵌合角色 B 的 EDA 图表、角色 C 的聚类雷达图与角色 D 的模型预测接口（实时输入参数输出评估）。
    *   统一规范全队代码仓库，汇总撰写最终项目报告（Report）与演示 PPT。
*   **关键交付**：`app/app.py`（交互大屏主程序）、最终项目总结报告与汇报 PPT。



## 项目目录

```text
nba-analytics/
├── .gitignore              # Git 忽略文件（已配置屏蔽大数据集与环境缓存）
├── README.md               # 本说明文档
├── requirements.txt        # 项目 Python 依赖包列表
├── data/
│   ├── raw/                # [本地] 原始数据集 (.csv)，不提交至 Git
│   └── processed/          # [本地] 清洗与 Merge 后的 Master 数据集
├── notebooks/              # 用于探索与分析的 Jupyter Notebooks
│   ├── 01_eda.ipynb
│   ├── 02_clustering.ipynb
│   └── 03_prediction.ipynb
├── src/                    # 项目核心可复用 Python 源码
│   ├── __init__.py
│   ├── data_cleaning.py    # 数据清洗与合并逻辑
│   ├── visualization.py    # 绘图函数封装
│   ├── models.py           # 算法模型封装
│   └── utils.py            # 通用工具小函数
├── saved_models/           # 保存训练好的模型权重 (如 .pkl)
└── app/                    # 交互大屏程序
    └── app.py              # Streamlit 主入口
```



## 快速开始与环境搭建 

### 1. 克隆仓库 
```bash
git clone https://github.com/Alexander-Hou/nba-analytics.git
cd nba-analytics
```

### 2. 配置 Python 虚拟环境
建议使用 `Python 3.10+` 环境：
```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate

# 激活虚拟环境 (Windows Cmd)
.venv\Scripts\activate.bat
```

### 3. 安装项目依赖
```bash
pip install -r requirements.txt
```

### 4. 获取数据集 
由于原始 `.csv` 数据集体积较大，不直接存储于 Git 仓库中。
1. 请在本地找到以下 3 个原始文件：
   - `Players.csv`
   - `Seasons_Stats.csv`
   - `player_data.csv`
2. 下载后请手动移动至本地项目的 `data/raw/` 目录下。



## Git 协作与分支规范

1. **主分支限制**：禁止直接在 `main` 分支上提交代码
2. **分支开发**：每个人在各自的功能分支上独立开发：
   - 角色 A: `git checkout -b feat/data-cleaning`
   - 角色 B: `git checkout -b feat/eda-vis`
   - 角色 C: `git checkout -b feat/clustering`
   - 角色 D: `git checkout -b feat/ml-prediction`
   - 角色 E: `git checkout -b feat/dashboard`
3. **提交与合并**：
   - 提交前请清空 Notebook 输出：`Kernel -> Restart & Clear Output`。
   - 开发完成后推送到远程，并发起 Pull Request (PR) 申请合并至 `main`。