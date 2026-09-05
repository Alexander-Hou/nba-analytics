"""NBA 预测模型推理与 SHAP 可解释性服务封装。

该模块面向 Streamlit 前端提供三个标准接口：

1. ``predict_per`` —— 预测球员下赛季 PER（效率值）；
2. ``predict_allstar_proba`` —— 输出入选全明星/核心级的概率与类别标签；
3. ``explain_prediction_waterfall`` —— 对单条样本计算 SHAP 并返回瀑布图 Figure。

所有 Pipeline 均以 ``joblib`` 序列化到 ``saved_models/`` 目录，
模型内部通过 ``feature_names_in_`` 记录完整特征列，从而实现
前端输入缺列时的自动补齐与容错。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


logger = logging.getLogger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = _PROJECT_ROOT / "saved_models"

DEFAULT_PER_MODEL_PATH = MODEL_DIR / "nba_per_xgb_pipeline.pkl"
DEFAULT_ALLSTAR_MODEL_PATH = MODEL_DIR / "nba_allstar_xgb_pipeline.pkl"

_POSITION_PREFIX = "Pos_"


class NBAPredictor:
    """NBA 模型预测与解释的统一服务入口。

    默认通过 :meth:`get_instance` 获取进程级单例；若需要在同一进程内
    同时使用多组模型文件，也可以直接实例化该类。
    """

    _instance: Optional[NBAPredictor] = None

    def __init__(
        self,
        per_model_path: Optional[Union[str, Path]] = None,
        allstar_model_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """初始化推理器（模型采用惰性加载）。

        Parameters
        ----------
        per_model_path : 回归 PER Pipeline 文件路径，缺省使用项目默认位置。
        allstar_model_path : 分类全明星 Pipeline 文件路径，缺省使用项目默认位置。
        """
        self.per_model_path = Path(
            per_model_path or DEFAULT_PER_MODEL_PATH
        ).resolve()
        self.allstar_model_path = Path(
            allstar_model_path or DEFAULT_ALLSTAR_MODEL_PATH
        ).resolve()

        self._per_pipeline: Any = None
        self._allstar_pipeline: Any = None
        self._per_features: Optional[list] = None
        self._allstar_features: Optional[list] = None

    # ------------------------------------------------------------------
    # 单例 / 模型加载
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(
        cls,
        per_model_path: Optional[Union[str, Path]] = None,
        allstar_model_path: Optional[Union[str, Path]] = None,
    ) -> NBAPredictor:
        """返回进程级单例实例。"""
        if cls._instance is None:
            cls._instance = cls(
                per_model_path=per_model_path,
                allstar_model_path=allstar_model_path,
            )
        return cls._instance

    def _ensure_per_pipeline(self) -> Any:
        """惰性加载 PER 回归 Pipeline，并缓存期望特征列。"""
        if self._per_pipeline is None:
            self._per_pipeline = self._load_pipeline(
                self.per_model_path, "PER 回归"
            )
            self._per_features = list(self._per_pipeline.feature_names_in_)
        return self._per_pipeline

    def _ensure_allstar_pipeline(self) -> Any:
        """惰性加载全明星分类 Pipeline，并缓存期望特征列。"""
        if self._allstar_pipeline is None:
            self._allstar_pipeline = self._load_pipeline(
                self.allstar_model_path, "全明星分类"
            )
            self._allstar_features = list(
                self._allstar_pipeline.feature_names_in_
            )
        return self._allstar_pipeline

    @staticmethod
    def _load_pipeline(path: Path, role: str) -> Any:
        """带清晰异常信息的模型加载函数。"""
        if not path.is_file():
            raise FileNotFoundError(
                f"{role}模型不存在：{path}。\n"
                "请先在 notebooks/03_prediction.ipynb 中运行模型导出单元格，"
                "或通过构造函数传入正确的模型路径。"
            )
        try:
            pipeline = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 - 对外给出统一错误信息
            raise RuntimeError(f"{role}模型加载失败：{path}") from exc

        if not hasattr(pipeline, "predict"):
            raise RuntimeError(f"{role}模型缺少 predict 方法：{path}")
        if not hasattr(pipeline, "feature_names_in_"):
            raise RuntimeError(
                f"{role}模型未记录 feature_names_in_，无法自动对齐输入列：{path}"
            )
        return pipeline

    # ------------------------------------------------------------------
    # 输入列对齐
    # ------------------------------------------------------------------
    @staticmethod
    def _to_single_row_frame(
        input_data: Union[Mapping[str, Any], pd.DataFrame, pd.Series]
    ) -> pd.DataFrame:
        """将字典 / DataFrame / Series 统一转换为仅含单行的 DataFrame。"""
        if isinstance(input_data, pd.DataFrame):
            frame = input_data.iloc[:1].copy()
        elif isinstance(input_data, pd.Series):
            frame = pd.DataFrame([input_data.to_dict()])
        elif isinstance(input_data, Mapping):
            frame = pd.DataFrame([dict(input_data)])
        else:
            raise TypeError(
                "input_data 必须是 dict / pd.Series / pd.DataFrame"
            )
        return frame.reset_index(drop=True)

    @classmethod
    def _prepare_features(
        cls,
        input_data: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
        expected_columns: Sequence[str],
    ) -> pd.DataFrame:
        """对齐并补齐模型期望的特征列。

        处理策略：

        - 若输入包含原始位置字段 ``Pos``，自动展开为独热列并置 1；
        - 若输入缺少模型期望列，统一以 ``NaN`` 补齐，由 Pipeline 的
          ``SimpleImputer`` 使用训练期 median 填充；
        - 额外传入的非模型列会被安全丢弃。
        """
        frame = cls._to_single_row_frame(input_data)
        expected = list(expected_columns)
        expected_pos_cols = [
            col for col in expected if col.startswith(_POSITION_PREFIX)
        ]

        position_value: Optional[str] = None
        if "Pos" in frame.columns and not frame["Pos"].isna().all():
            position_value = str(frame.loc[0, "Pos"]).strip()

        # 缺失的位置独热列先统一置 0，再由原始 Pos 字段激活对应列。
        for col in expected_pos_cols:
            if col not in frame.columns:
                frame[col] = 0

        if position_value:
            canonical_pos = (
                position_value
                if position_value.startswith(_POSITION_PREFIX)
                else f"{_POSITION_PREFIX}{position_value}"
            )
            for col in expected_pos_cols:
                frame[col] = 1 if col == canonical_pos else 0

        # 缺失的数值型 / 动态特征列统一补 NaN，等待 imputer 中位数填充。
        for col in expected:
            if col not in frame.columns:
                frame[col] = np.nan

        # 严格按模型训练时的列顺序重排，并转为数值类型。
        aligned = frame[expected].copy()
        for col in expected:
            aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        return aligned.astype(float)

    # ------------------------------------------------------------------
    # 公共推理接口
    # ------------------------------------------------------------------
    def predict_per(
        self,
        input_dict: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
    ) -> float:
        """预测球员下赛季 PER。

        Parameters
        ----------
        input_dict : 单一样本特征字典/数据行，允许缺列与附带无关列。

        Returns
        -------
        float
            预测的下赛季 PER，保留 2 位小数。
        """
        pipeline = self._ensure_per_pipeline()
        frame = self._prepare_features(input_dict, self._per_features)
        prediction = float(pipeline.predict(frame)[0])
        return round(prediction, 2)

    def predict_allstar_proba(
        self,
        input_dict: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
    ) -> Tuple[float, int]:
        """预测球员入选全明星/核心级（PER ≥ 20 且 WS ≥ 6）的概率与类别。

        Parameters
        ----------
        input_dict : 当季技术统计字典/数据行，允许缺列。

        Returns
        -------
        Tuple[float, int]
            ``(probability, label)``，其中 probability ∈ [0, 1] 保留 4 位小数，
            label 为 0（非核心级）或 1（核心级）。
        """
        pipeline = self._ensure_allstar_pipeline()
        model = pipeline.named_steps["model"]

        classes = np.asarray(model.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if positive_indices.size == 0:
            raise RuntimeError("分类模型训练标签中不存在正类 1")
        positive_index = int(positive_indices[0])

        frame = self._prepare_features(input_dict, self._allstar_features)
        probability = float(pipeline.predict_proba(frame)[0, positive_index])
        label = int(pipeline.predict(frame)[0])
        return round(probability, 4), label

    # ------------------------------------------------------------------
    # SHAP 瀑布图解释
    # ------------------------------------------------------------------
    def explain_prediction_waterfall(
        self,
        input_dict: Union[Mapping[str, Any], pd.DataFrame, pd.Series],
        max_display: int = 12,
    ) -> plt.Figure:
        """对单条 PER 预测生成 SHAP 瀑布图。

        Parameters
        ----------
        input_dict : 单一样本特征字典/数据行。
        max_display : 瀑布图中展示的最大特征数量。

        Returns
        -------
        matplotlib.figure.Figure
            可直接交由 Streamlit ``st.pyplot(fig)`` 渲染的 Figure。
        """
        pipeline = self._ensure_per_pipeline()
        imputer = pipeline.named_steps.get("imputer")
        model = pipeline.named_steps.get("model")
        if imputer is None or model is None:
            raise RuntimeError(
                "PER Pipeline 缺少 imputer 或 model 步骤，无法执行 SHAP 解释"
            )

        prepared = self._prepare_features(input_dict, self._per_features)
        imputed = np.asarray(imputer.transform(prepared))
        single_x = imputed[:1]

        explainer = shap.TreeExplainer(model)
        raw_shap_values = np.asarray(explainer.shap_values(single_x))
        if raw_shap_values.ndim != 2 or raw_shap_values.shape[0] != 1:
            raise RuntimeError(
                "TreeExplainer 输出维度异常，无法构建单样本 Explanation"
            )

        base_value = explainer.expected_value
        try:
            base_value = float(np.asarray(base_value).reshape(-1)[0])
        except (TypeError, ValueError):
            base_value = 0.0

        explanation = shap.Explanation(
            values=raw_shap_values[0],
            base_values=base_value,
            data=single_x[0],
            feature_names=self._per_features,
        )

        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6.5)
        fig.tight_layout()
        return fig


# 模块级便捷函数：直接复用进程级单例。
def get_predictor(
    per_model_path: Optional[Union[str, Path]] = None,
    allstar_model_path: Optional[Union[str, Path]] = None,
) -> NBAPredictor:
    """返回 NBAPredictor 单例（等价于 NBAPredictor.get_instance）。"""
    return NBAPredictor.get_instance(
        per_model_path=per_model_path,
        allstar_model_path=allstar_model_path,
    )


def _build_demo_input() -> Dict[str, Any]:
    """构建一组贴近真实球员的演示输入（含可选 Pos 与动态特征）。"""
    return {
        "Age": 27,
        "G": 75,
        "MP": 2900,
        "TS%": 0.598,
        "3PAr": 0.38,
        "FTr": 0.32,
        "USG%": 27.5,
        "PTS_per36": 25.4,
        "AST%": 24.1,
        "TRB%": 11.6,
        "WS": 9.2,
        "PER_diff": 1.8,
        "Career_Year": 8,
        "PER_3yr_avg": 20.6,
        "Pos": "PG",
    }


def main() -> None:
    """模块自测：预测 / 概率评估 / SHAP 瀑布图导出。"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    predictor = get_predictor()

    demo = _build_demo_input()

    per = predictor.predict_per(demo)
    print(f"预测下赛季 PER：{per:.2f}")

    probability, label = predictor.predict_allstar_proba(demo)
    print(
        "全明星/核心级概率："
        f"{probability:.4f}（类别：{'1 入选' if label == 1 else '0 未入选'}）"
    )

    figure = predictor.explain_prediction_waterfall(demo, max_display=12)
    demo_output = Path("/tmp/nba_shap_waterfall_demo.png")
    figure.savefig(demo_output, dpi=120, bbox_inches="tight")
    print(f"SHAP 瀑布图已保存：{demo_output}")


if __name__ == "__main__":
    main()
