"""基于 03_prediction.ipynb Baseline 的前端直算模型。

模型已经在主数据集上离线拟合，并将标准化后的系数还原为原始特征空间。
因此看板只需执行点积和 sigmoid，不依赖后端服务、joblib 或 XGBoost。
"""

from __future__ import annotations

import math
from typing import Mapping


FEATURES = [
    "Age", "G", "MP", "TS%", "3PAr", "FTr", "USG%", "PTS_per36",
    "AST%", "TRB%", "WS",
]
POSITION_COLUMNS = [
    "Pos_C-F", "Pos_C-PF", "Pos_C-SF", "Pos_F", "Pos_F-C", "Pos_F-G",
    "Pos_G", "Pos_G-F", "Pos_PF", "Pos_PF-C", "Pos_PF-SF", "Pos_PG",
    "Pos_PG-SF", "Pos_PG-SG", "Pos_SF", "Pos_SF-PF", "Pos_SF-PG",
    "Pos_SF-SG", "Pos_SG", "Pos_SG-PF", "Pos_SG-PG", "Pos_SG-SF",
]

# 由 notebook 中 LinearRegression / LogisticRegression 的 StandardScaler
# 系数还原得到，训练口径为 Player 分组后预测下一季 PER。
REGRESSION_INTERCEPT = 8.599278131400165
REGRESSION_COEF = {
    "Age": -0.16790307773239133, "G": -0.005702079489860931,
    "MP": 5.9067628488222906e-05, "TS%": 0.1777319754598474,
    "3PAr": -0.3480437317818415, "FTr": 0.7051119566731235,
    "USG%": 0.07092398599301868, "PTS_per36": 0.22014987773146583,
    "AST%": 0.09224867087272506, "TRB%": 0.13192246575244207,
    "WS": 0.6870283806850944, "Pos_C-F": -2.251357372436054,
    "Pos_C-PF": -0.6143705562550241, "Pos_C-SF": -1.1751408821926346,
    "Pos_F": -5.433946562617026, "Pos_F-C": -1.5786605012640358,
    "Pos_F-G": -6.240412720821869, "Pos_G": -4.836451736204838,
    "Pos_G-F": -0.7648294756898333, "Pos_PF": 0.012956546588391881,
    "Pos_PF-C": 1.3440177829582567, "Pos_PF-SF": 0.951461337856493,
    "Pos_PG": -0.4183087906492354, "Pos_PG-SF": 4.111119233079601,
    "Pos_PG-SG": -0.7037737509556069, "Pos_SF": -0.32681124337076384,
    "Pos_SF-PF": 1.320445610925982, "Pos_SF-PG": -1.2995303948165802,
    "Pos_SF-SG": -0.3563717990732909, "Pos_SG": -0.42881112498614865,
    "Pos_SG-PF": 0.15974663684383103, "Pos_SG-PG": 0.05888599393615314,
    "Pos_SG-SF": -2.1890587502737593,
}

CLASSIFICATION_INTERCEPT = -3.849267825889004
CLASSIFICATION_COEF = {
    "Age": -0.1838284721310216, "G": -0.024022192319276364,
    "MP": 0.00018822754015128412, "TS%": -0.5292326170618973,
    "3PAr": 0.2000625038680961, "FTr": 1.2211925802737946,
    "USG%": 0.06456502572985216, "PTS_per36": 0.20952074180199326,
    "AST%": 0.08410773989491292, "TRB%": 0.10186846542423489,
    "WS": 0.5202812837798179, "Pos_C-F": 2.4972879017258327,
    "Pos_C-PF": -4.991111048630997, "Pos_C-SF": -3.565031937907015,
    "Pos_F": -12.220050488280167, "Pos_F-C": 0.380036079504274,
    "Pos_F-G": -15.322488109222833, "Pos_G": -5.808972566946647,
    "Pos_G-F": -5.460680507314708, "Pos_PF": -0.6741150833392933,
    "Pos_PF-C": 0.7022606215260842, "Pos_PF-SF": -5.8872369255814885,
    "Pos_PG": -0.9627410645621228, "Pos_PG-SF": -4.53892049935476,
    "Pos_PG-SG": -6.776665256949912, "Pos_SF": -0.9237285433199984,
    "Pos_SF-PF": -5.873834341376225, "Pos_SF-PG": -3.6229237446398894,
    "Pos_SF-SG": -1.5840482325580836, "Pos_SG": -1.3265653894317255,
    "Pos_SG-PF": -6.2355693303075945, "Pos_SG-PG": -0.34108728742360417,
    "Pos_SG-SF": -5.6754506069461135,
}


def _vector(values: Mapping[str, float | str]) -> dict[str, float]:
    """将表单输入映射为训练时的 11 项指标及位置独热列。"""
    row = {name: float(values.get(name, 0.0) or 0.0) for name in FEATURES}
    position = str(values.get("Pos", "")).strip()
    for name in POSITION_COLUMNS:
        row[name] = 1.0 if name == f"Pos_{position}" else 0.0
    return row


def predict_per(values: Mapping[str, float | str]) -> float:
    row = _vector(values)
    return round(REGRESSION_INTERCEPT + sum(REGRESSION_COEF[k] * row[k] for k in REGRESSION_COEF), 2)


def predict_allstar_proba(values: Mapping[str, float | str]) -> tuple[float, int]:
    row = _vector(values)
    score = CLASSIFICATION_INTERCEPT + sum(CLASSIFICATION_COEF[k] * row[k] for k in CLASSIFICATION_COEF)
    probability = 1.0 / (1.0 + math.exp(-max(min(score, 60.0), -60.0)))
    probability = round(probability, 4)
    return probability, int(probability >= 0.5)


def formula_text() -> str:
    """返回给看板展示的模型说明。"""
    return "PER_next = 8.5993 + Σ(特征值 × 回归系数)；全明星概率 = sigmoid(-3.8493 + Σ(特征值 × 分类系数))"
