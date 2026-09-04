# -*- coding: utf-8 -*-
"""
NBA 球员数据分析 —— 可视化模块（模块 2）

职责:
    封装可复用的描述性统计与 EDA 绘图函数，供 notebooks/01_eda.ipynb 使用，
    同时供角色 E 的 Streamlit 看板直接 import 复用。

调用约定:
    - 所有 plot_* 函数返回 matplotlib Figure 对象，内部不调用 plt.show()，
      由调用方决定展示或保存；
    - 所有函数只读输入 DataFrame，绝不就地修改。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = PROJECT_ROOT / "data" / "processed" / "master_data.csv"

# 需要做「场均」折算的赛季总量列
PER_GAME_COLS = ["MP", "PTS", "TRB", "AST", "STL", "BLK", "TOV", "FGA", "3PA"]

# 相关性热力图默认考察的核心指标（统一用场均/率值口径，避免上场时间主导相关系数）
CORE_METRICS = [
    "Age", "MP_per_game", "PTS_per_game", "PER", "TS%", "USG%", "3PAr",
    "AST%", "TRB%", "STL%", "BLK%", "TOV%", "eFG%", "WS/48", "BPM",
]

# 描述性统计汇总表默认考察的指标
DEFAULT_DESCRIBE_COLS = [
    "Age", "G", "MP", "MP_per_game", "PTS", "PTS_per_game", "TRB_per_game",
    "AST_per_game", "PER", "TS%", "USG%", "3PAr", "WS/48", "BPM", "VORP", "PTS_per36",
]

_POS_ORDER = ["PG", "SG", "G", "SF", "F", "PF", "C"]


# ----------------------------------------------------------------------------
# 样式与数据读取
# ----------------------------------------------------------------------------
def setup_style(font_size: int = 11) -> None:
    """统一图表风格并配置中文字体。可重复调用（幂等）。

    顺序很重要：sns.set_theme() 会重置 font.family，必须先调用它再覆盖字体。
    """
    sns.set_theme(style="whitegrid")

    from matplotlib import font_manager

    candidates = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Hiragino Sans GB",
        "Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in candidates if name in available]
    plt.rcParams["font.sans-serif"] = chosen + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号在中文字体下会显示成方块
    plt.rcParams["font.size"] = font_size
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.bbox"] = "tight"


def load_master(path: Path | str | None = None) -> pd.DataFrame:
    """读取角色 A 交付的 master_data.csv，并补齐 EDA 需要的派生分析列。

    必须用 utf-8-sig 读取：该文件带 BOM，用默认编码读会让首列名变成 "\\ufeffYear"。
    """
    csv_path = Path(path) if path is not None else MASTER_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到主数据集: {csv_path}")

    return add_derived_columns(pd.read_csv(csv_path, encoding="utf-8-sig"))


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """追加 EDA 需要的派生列（不修改传入对象）。

    *_per_game  : 场均值。主数据集中的 MP/PTS/TRB 等是**赛季总量**（已用
                  Harden 2010 赛季 G=76、PTS=753 验证 = 场均 9.9 分），
                  直接拿去画分布或算相关会把「上场多少场」混进「打得有多好」。
    primary_pos : 主位置。原始 Pos 含 "G-F" 等组合，直接分组会得到稀疏碎类别。
    """
    out = df.copy()

    if "G" in out.columns:
        games = pd.to_numeric(out["G"], errors="coerce").replace(0, np.nan)
        for col in PER_GAME_COLS:
            if col in out.columns:
                out[f"{col}_per_game"] = (
                    pd.to_numeric(out[col], errors="coerce") / games
                ).round(2)

    if "Pos" in out.columns:
        out["primary_pos"] = out["Pos"].astype("string").str.split("-").str[0].str.strip()

    return out


def add_value_tier(df: pd.DataFrame, by: str = "BPM") -> pd.DataFrame:
    """按综合影响力指标划分价值梯队，作为「薪资梯队」的代理变量。

    三份原始数据都不含合同/薪资信息，因此主数据集没有薪资列。这里用 BPM
    （Box Plus-Minus，每百回合相对联盟平均的净贡献）分档近似球员档次，
    分档边界参照 NBA 轮换惯例：顶星 / 主力 / 轮换 / 边缘。
    拿到真实薪资数据后应改用真实薪资分档并废弃本代理。
    """
    out = df.copy()
    labels = ("顶星级", "主力级", "轮换级", "边缘级")

    values = pd.to_numeric(out[by], errors="coerce")
    top, middle, bottom = values.quantile([0.90, 0.50, 0.20]).to_list()

    def _classify(value: float) -> str:
        if pd.isna(value):
            return "未评档"
        if value >= top:
            return labels[0]
        if value >= middle:
            return labels[1]
        if value >= bottom:
            return labels[2]
        return labels[3]

    out["value_tier"] = pd.Categorical(
        out[by].map(_classify), categories=list(labels) + ["未评档"], ordered=True
    )
    return out


def filter_qualified(df: pd.DataFrame, min_mp: float = 500,
                     min_games: float = 5) -> pd.DataFrame:
    """筛选合格样本赛季行，剔除极小样本造成的统计假象。

    主数据集含大量 `G=1、MP 仅几分钟` 的短工合同行。这类行的比率型指标会取到
    无意义的极值（实测：Chuck Nevitt 1994 赛季 1 场 1 分钟 3 分 → `USG%=100`、
    `PTS_per36=108`；Gheorghe Muresan 1 场 1 分钟 → `PER=-90.6`）。
    直接对全量数据做分布分析、相关性、聚类或回归都会被这些行主导，
    因此角色 B/C/D/E 应统一使用本函数给出的合格样本口径。
    """
    mask = pd.Series(True, index=df.index)
    if "MP" in df.columns:
        mask &= pd.to_numeric(df["MP"], errors="coerce") >= min_mp
    if "G" in df.columns:
        mask &= pd.to_numeric(df["G"], errors="coerce") >= min_games
    return df[mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 描述性统计
# ----------------------------------------------------------------------------
def describe_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """产出描述性统计汇总表：计数、均值、中位数、标准差、分位数、偏度与缺失率。"""
    cols = [c for c in (columns or DEFAULT_DESCRIBE_COLS) if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")

    summary = pd.DataFrame({
        "样本数": numeric.count(),
        "缺失率%": numeric.isna().mean() * 100,
        "均值": numeric.mean(),
        "中位数": numeric.median(),
        "标准差": numeric.std(),
        "最小值": numeric.min(),
        "P25": numeric.quantile(0.25),
        "P75": numeric.quantile(0.75),
        "最大值": numeric.max(),
        "偏度": numeric.skew(),
    })
    return summary.round(3)


def plot_describe_heatmap(summary: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """把汇总表中的统计量做列内 Z 标准化后画热力图，直观对比各指标量纲与形态差异。"""
    setup_style()
    cols = [c for c in ["均值", "中位数", "标准差", "偏度"] if c in summary.columns]
    matrix = summary[cols].head(top_n)
    z = matrix.sub(matrix.mean()).div(matrix.std(ddof=0).replace(0, np.nan))

    fig, ax = plt.subplots(figsize=(7.2, max(4.0, len(z) * 0.34)))
    sns.heatmap(z, cmap="RdBu_r", center=0, annot=True, fmt=".1f",
                cbar_kws={"label": "Z 分数"}, ax=ax)
    ax.set_title("描述性统计量对比（列内 Z 标准化）")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0)
    return fig


# ----------------------------------------------------------------------------
# 分布分析
# ----------------------------------------------------------------------------
def plot_score_distribution(df: pd.DataFrame, metric: str = "PTS_per_game",
                           hue: str | None = None) -> plt.Figure:
    """绘制单项指标的分布直方图 + 核密度曲线，可按类别变量叠加对比。"""
    setup_style()
    keep = [metric] + ([hue] if hue else [])
    data = df[keep].dropna().copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    hue_order = None
    if hue:
        counts = data[hue].value_counts()
        hue_order = ([p for p in _POS_ORDER if p in counts.index]
                     if hue == "primary_pos" else counts.index.tolist())
    sns.histplot(
        data=data, x=metric, hue=hue, hue_order=hue_order, kde=True, bins=45,
        element="step" if hue else "bars",
        fill=hue is None,
        legend=hue is not None,
        ax=ax,
    )
    y_top = ax.get_ylim()[1]
    for stat, y_frac in (("mean", 0.95), ("median", 0.86)):
        value = data[metric].agg(stat)
        ax.axvline(value, linestyle="--", linewidth=1.2, alpha=0.75, color="black")
        ax.annotate(f"{stat}={value:.1f}", xy=(value, y_top * y_frac),
                    xytext=(4, 0), textcoords="offset points", fontsize=9)

    ax.set_title(f"{metric} 分布（n={len(data)}" + (f"，分组: {hue}" if hue else "") + "）")
    ax.set_xlabel(metric)
    ax.set_ylabel("球员-赛季 记录数")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title(hue)
        for text in legend.get_texts():
            text.set_fontsize(8)
    return fig


def plot_metric_trend(df: pd.DataFrame, metric: str = "3PAr", agg: str = "median") -> plt.Figure:
    """绘制指标随赛季的演化趋势，用于观察打法的时代迁移。"""
    setup_style()
    tmp = df[["Year", metric]].dropna()
    series = (
        pd.to_numeric(tmp[metric], errors="coerce")
        .groupby(tmp["Year"], observed=True)
        .agg(agg)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax, marker="o", linewidth=1.8)
    ax.set_title(f"{metric} 的赛季趋势（逐年取 {agg}）")
    ax.set_xlabel("赛季年份")
    ax.set_ylabel(metric)
    return fig


# ----------------------------------------------------------------------------
# 相关性分析
# ----------------------------------------------------------------------------
def plot_corr_heatmap(df: pd.DataFrame, columns: list[str] | None = None,
                     method: str = "pearson") -> plt.Figure:
    """绘制指标间的 Pearson（或 Spearman）相关系数热力图。"""
    setup_style()
    cols = [c for c in (columns or CORE_METRICS) if c in df.columns]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method=method)

    fig, ax = plt.subplots(figsize=(max(9, len(cols) * 0.8), max(7, len(cols) * 0.68)))
    sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.6,
                cbar_kws={"label": f"{method} 相关系数"}, ax=ax)
    ax.set_title(f"核心指标 {method.capitalize()} 相关性矩阵")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return fig


def plot_scatter_with_trend(df: pd.DataFrame, x: str = "MP_per_game",
                           y: str = "PER") -> plt.Figure:
    """绘制双变量散点图并叠加回归线，标题报告 Pearson r 与样本量。"""
    setup_style()
    data = df[[x, y]].dropna()
    r = pd.to_numeric(data[x], errors="coerce").corr(
        pd.to_numeric(data[y], errors="coerce"), method="pearson")

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    sns.regplot(data=data, x=x, y=y, scatter_kws={"s": 12, "alpha": 0.35}, ax=ax)
    ax.set_title(f"{y} vs {x}（Pearson r = {r:.3f}，n = {len(data)}）")
    return fig


# ----------------------------------------------------------------------------
# 分组对比（箱线图 / 小提琴图 / 中位数排名）
# ----------------------------------------------------------------------------
def _prepare_groups(df: pd.DataFrame, metric: str, group: str,
                    top_groups: int) -> tuple[pd.DataFrame, list[str]]:
    """清洗分组数据，并按样本量保留前 top_groups 个组别、排列坐标轴顺序。"""
    data = df[[metric, group]].dropna().copy()
    counts = data[group].value_counts()
    keep = counts.head(top_groups).index.tolist()
    data = data[data[group].isin(keep)]

    if group == "primary_pos":
        ordered = [p for p in _POS_ORDER if p in keep]
        ordered += [p for p in sorted(keep) if p not in ordered]
    else:
        # 列本身已是有序分类（如 value_tier）时沿用其语义顺序；
        # 按字面排序会把「顶星级/主力级/轮换级/边缘级」打散，并把「未评档」挤到轴中间。
        existing = getattr(df[group].dtype, "categories", None)
        if existing is not None:
            ordered = [c for c in existing if c in keep]
            ordered += [c for c in sorted(keep) if c not in ordered]
        else:
            ordered = sorted(keep)
    data[group] = pd.Categorical(data[group], categories=ordered, ordered=True)
    return data, ordered


def plot_group_box(df: pd.DataFrame, metric: str, group: str = "primary_pos",
                   top_groups: int = 8) -> plt.Figure:
    """箱线图：对比不同组别（位置 / 时代 / 价值梯队）的指标分布。"""
    setup_style()
    data, ordered = _prepare_groups(df, metric, group, top_groups)

    fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 1.2), 5.2))
    sns.boxplot(data=data, x=group, y=metric, hue=group, palette="deep",
                legend=False, showfliers=False, ax=ax)
    ax.set_title(f"{metric} 按 {group} 对比（箱线图，不显示离群点）")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    return fig


def plot_group_violin(df: pd.DataFrame, metric: str, group: str = "era",
                      top_groups: int = 8) -> plt.Figure:
    """小提琴图：在分组对比之外额外呈现分布形态（双峰、长尾）。"""
    setup_style()
    data, ordered = _prepare_groups(df, metric, group, top_groups)

    fig, ax = plt.subplots(figsize=(max(8, len(ordered) * 1.2), 5.2))
    sns.violinplot(data=data, x=group, y=metric, hue=group, palette="muted",
                   legend=False, inner="box", cut=0, ax=ax)
    ax.set_title(f"{metric} 按 {group} 对比（小提琴图）")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    return fig


def plot_group_median_bar(df: pd.DataFrame, metric: str,
                          group: str = "primary_pos") -> plt.Figure:
    """分组中位数条形图：比箱线图更适合横向比较组间差距的大小。"""
    setup_style()
    data = df[[metric, group]].dropna().copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    medians = (
        data.groupby(group, observed=True)[metric].median()
        .sort_values(ascending=False)
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(9, max(3.6, len(medians) * 0.44)))
    medians.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", len(medians)))
    ax.invert_yaxis()
    ax.set_title(f"{metric} 中位数排名 by {group}")
    ax.set_xlabel(f"{metric}（中位数）")
    ax.set_ylabel("")
    return fig
