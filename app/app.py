"""NBA 球员数据分析 Streamlit 看板入口。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import visualization as viz  # noqa: E402
from src import formula_model  # noqa: E402


MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "master_data.csv"
MODEL_DIR = PROJECT_ROOT / "saved_models"
FIGURE_DIR = PROJECT_ROOT / "figures"

st.set_page_config(page_title="NBA 球员数据分析", page_icon="🏀", layout="wide")


@st.cache_data(show_spinner="正在加载主数据集…")
def load_data(path: str) -> pd.DataFrame:
    """读取主数据并补齐 EDA 所需的派生列。"""
    return viz.load_master(path)


def apply_filters(
    data: pd.DataFrame,
    years: tuple[int, int],
    positions: list[str],
    teams: list[str],
    min_minutes: int,
) -> pd.DataFrame:
    """按侧边栏条件筛选数据，不修改缓存中的原始 DataFrame。"""
    filtered = data.loc[data["Year"].between(*years)].copy()
    if positions:
        filtered = filtered[filtered["primary_pos"].isin(positions)]
    if teams:
        filtered = filtered[filtered["Tm"].isin(teams)]
    return viz.filter_qualified(filtered, min_mp=min_minutes)


def render_overview(data: pd.DataFrame) -> None:
    st.header("项目概览")
    if data.empty:
        st.info("当前筛选条件没有符合最低出场时间要求的记录。")
        return

    cols = st.columns(5)
    cols[0].metric("球员赛季记录", f"{len(data):,}")
    cols[1].metric("球员数量", f"{data['Player'].nunique():,}")
    cols[2].metric("覆盖赛季", f"{int(data['Year'].min())}–{int(data['Year'].max())}")
    cols[3].metric("场均得分中位数", f"{data['PTS_per_game'].median():.1f}")
    cols[4].metric("PER 中位数", f"{data['PER'].median():.1f}")

    left, right = st.columns((1.25, 1))
    with left:
        trend = (
            data.groupby("Year", as_index=False)
            .agg(场均得分=("PTS_per_game", "median"), 真实命中率=("TS%", "median"))
            .dropna()
        )
        figure = px.line(
            trend,
            x="Year",
            y=["场均得分", "真实命中率"],
            title="球员赛季表现的中位数趋势",
            labels={"value": "数值", "variable": "指标", "Year": "赛季"},
        )
        st.plotly_chart(figure, use_container_width=True)
    with right:
        position_counts = (
            data["primary_pos"].value_counts().rename_axis("位置").reset_index(name="记录数")
        )
        figure = px.bar(position_counts, x="位置", y="记录数", title="合格样本的位置分布")
        st.plotly_chart(figure, use_container_width=True)


def render_player_explorer(data: pd.DataFrame) -> None:
    st.header("球员与赛季探索")
    if data.empty:
        st.info("当前筛选条件没有数据。")
        return

    players = sorted(data["Player"].dropna().unique().tolist())
    selected = st.multiselect("选择球员（最多 5 人）", players, max_selections=5)
    display_columns = [
        col for col in ["Player", "Year", "Tm", "Pos", "G", "MP", "PTS", "PTS_per_game", "PER", "TS%", "AST", "TRB"] if col in data
    ]
    if selected:
        player_data = data[data["Player"].isin(selected)].sort_values(["Player", "Year"])
        figure = px.line(
            player_data,
            x="Year",
            y="PER",
            color="Player",
            markers=True,
            title="所选球员的 PER 赛季趋势",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(player_data[display_columns], use_container_width=True, hide_index=True)
    else:
        ranking = data.sort_values("PER", ascending=False).head(50)
        st.caption("未选择球员时展示当前筛选范围内 PER 最高的 50 条赛季记录。")
        st.dataframe(ranking[display_columns], use_container_width=True, hide_index=True)


def render_eda(data: pd.DataFrame) -> None:
    st.header("效率与分布")
    if data.empty:
        st.info("当前筛选条件没有数据。")
        return

    metric = st.selectbox("分布指标", ["PTS_per_game", "PER", "TS%", "MP_per_game", "PTS_per36"])
    left, right = st.columns(2)
    with left:
        st.pyplot(viz.plot_score_distribution(data, metric=metric), clear_figure=True)
    with right:
        st.pyplot(viz.plot_group_box(data, metric=metric, group="primary_pos"), clear_figure=True)

    st.subheader("核心指标相关性")
    st.pyplot(viz.plot_corr_heatmap(data), clear_figure=True)


def render_comparison(data: pd.DataFrame) -> None:
    st.header("球队与位置对比")
    if data.empty:
        st.info("当前筛选条件没有数据。")
        return

    metric = st.selectbox("比较指标", ["PTS_per_game", "PER", "TS%", "AST_per_game", "TRB_per_game"], key="comparison_metric")
    group = st.radio("比较维度", ["球队", "位置"], horizontal=True)
    column = "Tm" if group == "球队" else "primary_pos"
    summary = (
        data.groupby(column, as_index=False)
        .agg(中位数=(metric, "median"), 样本数=(metric, "count"))
        .query("样本数 >= 10")
        .sort_values("中位数", ascending=False)
        .head(20)
    )
    figure = px.bar(
        summary.sort_values("中位数"), x="中位数", y=column, orientation="h",
        color="样本数", color_continuous_scale="Blues", title=f"{group}{metric} 中位数排名",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)


def render_clustering() -> None:
    st.header("聚类挖掘成果")
    st.caption("本页直接展示角色 C 已产出的分时代聚类图。后续若提供聚类结果 CSV，可在此扩展为逐球员交互筛选。")
    eras = ["1980s", "1990s", "2000s", "2010s"]
    era = st.selectbox("选择时代", eras)
    pca_path = FIGURE_DIR / f"clustering_pca_feature_{era}.png"
    radar_path = FIGURE_DIR / f"clustering_radar_manual_{era}.png"
    left, right = st.columns(2)
    with left:
        if pca_path.exists():
            st.image(str(pca_path), caption=f"{era} PCA 聚类分布")
        else:
            st.warning("未找到 PCA 聚类图。")
    with right:
        if radar_path.exists():
            st.image(str(radar_path), caption=f"{era} 聚类特征雷达图")
        else:
            st.warning("未找到聚类雷达图。")
    evolution_path = FIGURE_DIR / "clustering_era_evolution_two_stage.png"
    if evolution_path.exists():
        st.image(str(evolution_path), caption="跨时代聚类演化")


def render_prediction() -> None:
    st.header("预测模型")
    st.caption("输入当季基础统计，前端直接套用 notebook Baseline 公式计算下赛季 PER 与全明星级概率，无需后端服务。")
    with st.expander("模型口径与公式", expanded=False):
        st.write(formula_model.formula_text())
        st.markdown(
            "- 回归：`LinearRegression`，严格按 `Year ≤ 2010 / Year > 2010` 时序验证，测试集 **R²=0.3441，RMSE=4.7785**。\n"
            "- 分类：`LogisticRegression(class_weight='balanced')`，测试集 **ROC-AUC=0.9539，F1=0.5732**。\n"
            "- 位置字段按 notebook 的 `Pos` 独热编码处理；未填写项使用 0。"
        )
    with st.form("prediction_form"):
        cols = st.columns(3)
        values = {
            "Age": cols[0].number_input("年龄", min_value=18, max_value=45, value=27),
            "G": cols[1].number_input("出场数", min_value=1, max_value=82, value=72),
            "MP": cols[2].number_input("总出场时间", min_value=1, max_value=4000, value=2400),
            "TS%": cols[0].number_input("TS%", min_value=0.0, max_value=1.0, value=0.56, step=0.01),
            "3PAr": cols[1].number_input("3PAr", min_value=0.0, max_value=1.0, value=0.30, step=0.01),
            "FTr": cols[2].number_input("FTr", min_value=0.0, max_value=2.0, value=0.25, step=0.01),
            "USG%": cols[0].number_input("USG%", min_value=0.0, max_value=100.0, value=22.0),
            "PTS_per36": cols[1].number_input("每 36 分钟得分", min_value=0.0, max_value=80.0, value=18.0),
            "AST%": cols[2].number_input("AST%", min_value=0.0, max_value=100.0, value=15.0),
            "TRB%": cols[0].number_input("TRB%", min_value=0.0, max_value=100.0, value=10.0),
            "WS": cols[1].number_input("WS", min_value=-10.0, max_value=30.0, value=5.0),
            "Pos": cols[2].selectbox("位置", ["PG", "SG", "SF", "PF", "C"]),
        }
        submitted = st.form_submit_button("运行预测")
    if submitted:
        try:
            per = formula_model.predict_per(values)
            probability, label = formula_model.predict_allstar_proba(values)
            left, right = st.columns(2)
            left.metric("预测下赛季 PER", f"{per:.2f}")
            right.metric("全明星级概率", f"{probability:.1%}", "核心级" if label else "非核心级")
            st.caption("提示：该概率对应 notebook 中“下一季 PER ≥ 20 且 WS ≥ 6”的全明星级定义。")
        except (TypeError, ValueError, OverflowError) as exc:
            st.error(f"输入数据无法计算：{exc}")


def main() -> None:
    st.title("🏀 NBA 球员数据分析与机器学习看板")
    st.caption("覆盖 1950–2017 赛季数据。默认仅展示出场时间不少于 500 分钟的合格样本，以降低极小样本对比率指标的干扰。")
    if not MASTER_PATH.exists():
        st.error("未找到 `data/processed/master_data.csv`。请先运行 `python src/data_cleaning.py` 生成主数据集。")
        st.stop()
    data = load_data(str(MASTER_PATH))

    st.sidebar.header("筛选条件")
    years = st.sidebar.slider("赛季范围", int(data["Year"].min()), int(data["Year"].max()), (int(data["Year"].min()), int(data["Year"].max())))
    positions = st.sidebar.multiselect("位置", sorted(data["primary_pos"].dropna().unique()))
    teams = st.sidebar.multiselect("球队", sorted(data["Tm"].dropna().unique()))
    min_minutes = st.sidebar.slider("最低总出场时间", 0, int(data["MP"].max()), 500, step=100)
    filtered = apply_filters(data, years, positions, teams, min_minutes)

    page = st.sidebar.radio("页面", ["项目概览", "球员与赛季探索", "效率与分布", "球队与位置对比", "聚类挖掘成果", "预测模型"])
    if page == "项目概览":
        render_overview(filtered)
    elif page == "球员与赛季探索":
        render_player_explorer(filtered)
    elif page == "效率与分布":
        render_eda(filtered)
    elif page == "球队与位置对比":
        render_comparison(filtered)
    elif page == "聚类挖掘成果":
        render_clustering()
    else:
        render_prediction()


if __name__ == "__main__":
    main()
