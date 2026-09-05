# -*- coding: utf-8 -*-
"""
NBA 球员数据分析与预测 —— 监督学习模型封装模块（模块 4 收尾交付）

职责:
    1. 复现 `notebooks/03_prediction_final.ipynb` 中回归与分类任务的核心建模流程；
    2. 提供可复用的特征工程、样本构造、时间切分、模型训练与评估函数；
    3. 支持 SHAP 可解释性计算，以及最终回归 Pipeline 的 joblib 持久化与加载推理；
    4. 保持与 `models/nba_per_xgb_pipeline.pkl` 导出产物一致的路径约定。

关键交付:
    - 本脚本: src/models.py
    - 导出的最终回归模型: models/nba_per_xgb_pipeline.pkl
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor


# ----------------------------------------------------------------------------
# 控制台编码自愈（保持与 data_cleaning.py 一致，避免 Windows 中文输出报错）
# ----------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# 路径与常量配置
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MASTER_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "master_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "nba_per_xgb_pipeline.pkl"

# 分类任务使用的原统计特征（与 03_prediction_advance.ipynb / final 一致）
FEATURE_COLS_CLS = [
    "Age",
    "G",
    "MP",
    "TS%",
    "3PAr",
    "FTr",
    "USG%",
    "PTS_per36",
    "AST%",
    "TRB%",
    "WS",
]

# 回归任务额外追加的时序 / 生涯特征
REGRESSION_EXTRA_FEATURES = ["PER_diff", "Career_Year", "PER_3yr_avg"]

# 回归任务使用的完整特征（与 03_prediction_advance_plus_2.ipynb 一致）
FEATURE_COLS_REG = FEATURE_COLS_CLS + REGRESSION_EXTRA_FEATURES

# 与建模无关 / 可能造成泄露的字段
DROP_COLS = [
    "Player",
    "Tm",
    "college",
    "birth_year",
    "birth_city",
    "birth_state",
    "birth_date",
    "year_start",
    "year_end",
    "position_career",
]

# 最终回归模型超参数
DEFAULT_REG_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# 分类模型超参数（scale_pos_weight 应对约 5.9% 的正类比例）
DEFAULT_CLS_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.05,
    "scale_pos_weight": 15.6,
    "random_state": 42,
}


# ----------------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------------
def _pos_columns(df: pd.DataFrame) -> list[str]:
    """返回 DataFrame 中由 Pos 独热编码生成的哑变量列名。"""
    return [col for col in df.columns if col.startswith("Pos_")]


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    """校验 DataFrame 是否包含指定列，缺失时给出明确错误。"""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{context} 缺少必需列: {missing}")


# ----------------------------------------------------------------------------
# 1. 特征工程
# ----------------------------------------------------------------------------
def add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构造回归标签 `PER_next` 与三个时序 / 生涯特征。

    规则（与最终 notebook 一致）:
        - PER_next:      下赛季 PER（shift(-1)）；
        - PER_diff:      当季 PER - 上一赛季 PER，仅当相邻赛季年份差 == 1 时计算，
                         断档复出置为 NaN，新秀首季显式设为 0；
        - Career_Year:   按 Player 组内 Year 的最小值计算生涯第几年；
        - PER_3yr_avg:   近三年 PER 移动平均（min_periods=1）。
    """
    _require_columns(df, ["Player", "Year", "PER"], "add_time_series_features")

    result = df.sort_values(["Player", "Year"]).reset_index(drop=True).copy()

    # 回归标签：下一赛季 PER
    result["PER_next"] = result.groupby("Player")["PER"].shift(-1)

    # PER_diff：仅相邻赛季（Year 差 == 1）才计算环比变化
    prev_year = result.groupby("Player")["Year"].shift(1)
    prev_per = result.groupby("Player")["PER"].shift(1)
    result["PER_diff"] = np.where(
        result["Year"] - prev_year == 1,
        result["PER"] - prev_per,
        np.nan,
    )

    # Career_Year：该球员最早数据年份记为第 1 年
    result["Career_Year"] = (
        result["Year"] - result.groupby("Player")["Year"].transform("min") + 1
    )

    # 新秀首季没有“上一赛季”，从可解释性角度 PER_diff 应为 0
    result.loc[result["Career_Year"] == 1, "PER_diff"] = 0.0

    # 近三年 PER 移动平均
    result["PER_3yr_avg"] = result.groupby("Player")["PER"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    return result


def drop_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """剔除身份标识、球队环境、出生信息等无关 / 易泄露字段。"""
    return df.drop(columns=[col for col in DROP_COLS if col in df.columns]).copy()


def encode_position(df: pd.DataFrame) -> pd.DataFrame:
    """将位置字段 Pos 进行 One-Hot Encoding。"""
    _require_columns(df, ["Pos"], "encode_position")
    dummies = pd.get_dummies(df["Pos"], prefix="Pos").astype(int)
    return pd.concat([df.drop(columns=["Pos"]), dummies], axis=1)


def prepare_common_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    完成两任务共用的数据准备：时序特征 + 剔除无关字段 + Pos 独热编码。

    若 df 为 None，则从默认路径 data/processed/master_data.csv 读取。
    """
    if df is None:
        df = pd.read_csv(MASTER_DATA_PATH)

    prepared = add_time_series_features(df)
    prepared = drop_identity_columns(prepared)
    prepared = encode_position(prepared)
    return prepared


def make_feature_matrix(data: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """按指定特征列构造特征矩阵（统计/时序特征 + Pos 哑变量）。"""
    _require_columns(data, feature_cols, "make_feature_matrix")
    return data[feature_cols + _pos_columns(data)].copy()


# ----------------------------------------------------------------------------
# 2. 样本构造与时间切分
# ----------------------------------------------------------------------------
def prepare_regression_samples(
    model_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    构造回归任务样本。

    Returns:
        (X_reg, y_reg, year_reg, sample_weight)
        sample_weight 为当季累计出场时间 MP，用于训练时给主力球员更高权重。
    """
    _require_columns(model_df, ["PER_next", "Year", "MP"], "prepare_regression_samples")
    reg_df = model_df[model_df["PER_next"].notna()].copy()

    X_reg = make_feature_matrix(reg_df, FEATURE_COLS_REG)
    y_reg = reg_df["PER_next"]
    year_reg = reg_df["Year"]
    sample_weight = reg_df["MP"].astype(float)
    return X_reg, y_reg, year_reg, sample_weight


def prepare_classification_samples(
    model_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    构造分类任务样本（当季是否达到全明星实力门槛）。

    Returns:
        (X_cls, y_cls, year_cls)
    """
    _require_columns(model_df, ["PER", "WS", "Year"], "prepare_classification_samples")
    cls_df = model_df.dropna(subset=["PER", "WS"]).copy()
    cls_df["Is_AllStar_Caliber"] = (
        (cls_df["PER"] >= 20.0) & (cls_df["WS"] >= 6.0)
    ).astype(int)

    X_cls = make_feature_matrix(cls_df, FEATURE_COLS_CLS)
    y_cls = cls_df["Is_AllStar_Caliber"]
    year_cls = cls_df["Year"]
    return X_cls, y_cls, year_cls


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    years: pd.Series,
    sample_weight: pd.Series | None = None,
    train_max: int = 2010,
) -> tuple:
    """
    按赛季年份切分训练集 / 测试集，防止未来数据泄露。

    默认切分: Train `Year <= 2010`，Test `Year > 2010`。
    若传入 sample_weight，会同步返回 train/test 权重。
    """
    train_mask = years <= train_max
    test_mask = ~train_mask

    result = (
        X.loc[train_mask],
        X.loc[test_mask],
        y.loc[train_mask],
        y.loc[test_mask],
    )
    if sample_weight is not None:
        result += (
            sample_weight.loc[train_mask],
            sample_weight.loc[test_mask],
        )
    return result


# ----------------------------------------------------------------------------
# 3. 模型 Pipeline 构建、训练与评估
# ----------------------------------------------------------------------------
def build_regression_pipeline(
    model_params: dict | None = None,
    imputer_strategy: str = "median",
) -> Pipeline:
    """构建回归 Pipeline：中位数填补 + XGBRegressor。"""
    params = DEFAULT_REG_PARAMS.copy()
    if model_params:
        params.update(model_params)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("model", XGBRegressor(**params)),
        ]
    )


def build_classification_pipeline(
    model_params: dict | None = None,
    imputer_strategy: str = "median",
) -> Pipeline:
    """构建分类 Pipeline：中位数填补 + XGBClassifier。"""
    params = DEFAULT_CLS_PARAMS.copy()
    if model_params:
        params.update(model_params)

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("model", XGBClassifier(**params)),
        ]
    )


def train_regression_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | None = None,
    pipeline: Pipeline | None = None,
    model_params: dict | None = None,
) -> Pipeline:
    """
    训练回归 Pipeline。

    使用 sample_weight（通常为 MP）时，通过 `model__sample_weight`
    将权重仅传递给 XGBRegressor，SimpleImputer 不接收样本权重。
    """
    if pipeline is None:
        pipeline = build_regression_pipeline(model_params=model_params)

    if sample_weight is not None:
        pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train)
    return pipeline


def train_classification_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline: Pipeline | None = None,
    model_params: dict | None = None,
) -> Pipeline:
    """训练分类 Pipeline。"""
    if pipeline is None:
        pipeline = build_classification_pipeline(model_params=model_params)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_regression(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """计算回归测试集上的 R² 与 RMSE。"""
    y_pred = pipeline.predict(X_test)
    return {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    }


def evaluate_classification(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """计算分类测试集上的 Accuracy / Precision / Recall / F1。"""
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }


# ----------------------------------------------------------------------------
# 4. SHAP 可解释性计算
# ----------------------------------------------------------------------------
def explain_regression(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
) -> tuple:
    """
    使用 shap.TreeExplainer 计算回归模型在测试集上的 SHAP 值。

    Returns:
        (shap_values, X_test_clean, importance_df)
        - X_test_clean: 经与训练一致的中位数填补后的测试特征；
        - importance_df: 平均绝对 SHAP 重要性表。
    """
    import shap  # 延迟导入，避免模块在非解释性场景下依赖 shap

    model = pipeline.named_steps["model"]
    imputer = pipeline.named_steps["imputer"]

    # 测试集先按训练口径填补，保证 SHAP 输入与模型训练一致
    X_test_clean = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    explainer = shap.TreeExplainer(model, feature_names=list(X_test_clean.columns))
    shap_values = explainer(X_test_clean)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = (
        pd.DataFrame(
            {
                "Feature": X_test_clean.columns,
                "Mean |SHAP|": mean_abs_shap,
            }
        )
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )
    return shap_values, X_test_clean, importance_df


# ----------------------------------------------------------------------------
# 5. 模型持久化与推理
# ----------------------------------------------------------------------------
def save_pipeline(
    pipeline: Pipeline,
    path: str | Path = DEFAULT_MODEL_PATH,
) -> Path:
    """将 Pipeline 使用 joblib 导出到指定路径（默认 models/nba_per_xgb_pipeline.pkl）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline 已导出: {path.resolve()}")
    return path


def load_pipeline(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """加载已导出的 Pipeline。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    return joblib.load(path)


def predict_next_per(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> float | np.ndarray:
    """
    使用回归 Pipeline 预测下赛季 PER。

    单条样本返回 float，多条样本返回 ndarray。
    """
    y_pred = pipeline.predict(X)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1 and y_pred.size == 1:
        return float(y_pred[0])
    return y_pred


# ----------------------------------------------------------------------------
# 主流程：快速复现 final notebook 的训练 / 评估 / 导出
# ----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("NBA 球员模型封装 —— 回归 + 分类 + 导出")
    print("=" * 70)

    model_df = prepare_common_dataset()
    print(f"模型数据集规模: {model_df.shape[0]} 行 x {model_df.shape[1]} 列")

    # ---- 回归：评估 + 全量重训导出 ----
    X_reg, y_reg, year_reg, sample_weight = prepare_regression_samples(model_df)
    (
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
        sample_weight_train,
        _,
    ) = time_split(X_reg, y_reg, year_reg, sample_weight=sample_weight)

    eval_pipeline = train_regression_model(
        X_train_reg, y_train_reg, sample_weight=sample_weight_train
    )
    reg_metrics = evaluate_regression(eval_pipeline, X_test_reg, y_test_reg)
    print(
        f"[回归测试集] R2 = {reg_metrics['r2']:.4f}, "
        f"RMSE = {reg_metrics['rmse']:.4f}"
    )

    # 部署模型：在全部带标签回归样本上重训
    final_pipeline = train_regression_model(X_reg, y_reg, sample_weight=sample_weight)
    save_pipeline(final_pipeline)

    # 加载 + 推理测试
    loaded_pipeline = load_pipeline()
    sample_row = X_test_reg.iloc[[0]]
    pred = predict_next_per(loaded_pipeline, sample_row)
    print(f"[模型加载推理测试] 样本真实 PER_next = {y_test_reg.iloc[0]:.4f}, "
          f"预测 PER_next = {pred:.4f}")

    # ---- 分类：评估回顾 ----
    X_cls, y_cls, year_cls = prepare_classification_samples(model_df)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = time_split(
        X_cls, y_cls, year_cls
    )
    cls_pipeline = train_classification_model(X_train_cls, y_train_cls)
    cls_metrics = evaluate_classification(cls_pipeline, X_test_cls, y_test_cls)
    print(
        f"[分类测试集] Accuracy = {cls_metrics['accuracy']:.4f}, "
        f"Precision = {cls_metrics['precision']:.4f}, "
        f"Recall = {cls_metrics['recall']:.4f}, "
        f"F1 = {cls_metrics['f1']:.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())