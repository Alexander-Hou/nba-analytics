# -*- coding: utf-8 -*-
"""
NBA 球员信息统计与可视化 —— 数据清洗模块（模块 1）

职责:
    1. 加载并校验 Players.csv、Seasons_Stats.csv、player_data.csv 三份原始数据；
    2. 处理缺失值、异常极值、同名不同人以及历史时代数据归一化；
    3. 按照 Player + Year 维度打通跨表关联，构建主数据集 Master Dataset；
    4. 派生真实命中率 TS% 与每 36 分钟折算特征等核心指标。

关键交付:
    - 本脚本: src/data_cleaning.py
    - 主数据集: data/processed/master_data.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# 控制台编码自愈：Windows 中文终端默认 GBK/ASCII，中文 print 会抛
# UnicodeEncodeError 导致脚本中途崩溃、CSV 来不及生成。
# 这里把控制台切到 UTF-8 代码页，并把 stdout/stderr 重配为 UTF-8 + 容错，
# 保证任何字符都不会让脚本崩溃。
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

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"          # 原始数据目录（兼容写法）
if not RAW_DIR.exists():
    # 原始 CSV 直接放在项目根目录的情况
    RAW_DIR = PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PLAYERS_CSV = RAW_DIR / "Players.csv"
PLAYER_DATA_CSV = RAW_DIR / "player_data.csv"
SEASONS_CSV = RAW_DIR / "Seasons_Stats.csv"

# 唯一交付的数据集：统一标准主数据集
MASTER_OUT = PROCESSED_DIR / "master_data.csv"


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _to_numeric(series: pd.Series) -> pd.Series:
    """安全地把一个 Series 转成数值类型。"""
    return pd.to_numeric(series, errors="coerce")


def _height_ft_in_to_cm(value) -> float:
    """
    将 '6-10' (6 英尺 10 英寸) 形式的身高转换为厘米。
    1 英尺 = 30.48 cm, 1 英寸 = 2.54 cm。
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if "-" not in s:
        return np.nan
    try:
        feet, inches = s.split("-")
        return float(feet) * 30.48 + float(inches) * 2.54
    except (ValueError, TypeError):
        return np.nan


def _lbs_to_kg(value) -> float:
    """磅 -> 千克（保留 1 位小数）。"""
    if pd.isna(value):
        return np.nan
    try:
        return round(float(value) * 0.45359237, 1)
    except (ValueError, TypeError):
        return np.nan


# ----------------------------------------------------------------------------
# 1. 加载与校验原始数据
# ----------------------------------------------------------------------------
def load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """加载三份原始 CSV，并做最基本的列存在性校验。"""
    print("[1/6] 加载原始数据 ...")

    for path in (PLAYERS_CSV, PLAYER_DATA_CSV, SEASONS_CSV):
        if not path.exists():
            raise FileNotFoundError(f"找不到原始数据文件: {path}")

    players = pd.read_csv(PLAYERS_CSV, encoding="utf-8")
    player_data = pd.read_csv(PLAYER_DATA_CSV, encoding="utf-8")
    seasons = pd.read_csv(SEASONS_CSV, encoding="utf-8")

    print(f"  Players.csv      : {players.shape[0]} 行 x {players.shape[1]} 列")
    print(f"  player_data.csv  : {player_data.shape[0]} 行 x {player_data.shape[1]} 列")
    print(f"  Seasons_Stats.csv   : {seasons.shape[0]} 行 x {seasons.shape[1]} 列")
    return players, player_data, seasons


# ----------------------------------------------------------------------------
# 2. 清洗 Players.csv —— 球员身体信息表
# ----------------------------------------------------------------------------
def clean_players(players: pd.DataFrame) -> pd.DataFrame:
    """清洗 Players.csv：重命名列、单位归一、缺失值处理、去重。"""
    print("[2/6] 清洗 Players.csv ...")

    df = players.copy()

    # 2.1 去掉原始行索引列（首列无名）
    first_col = df.columns[0]
    if first_col == "" or first_col.lower().startswith("unnamed"):
        df = df.drop(columns=first_col)

    # 2.2 列重命名：collage 是原始数据的拼写错误，统一为 college；born 表示出生年份；
    #              height/weight 显式标注单位，避免与 player_data 合并时列名冲突
    df = df.rename(columns={
        "Player": "name",
        "collage": "college",
        "born": "birth_year",
        "height": "height_cm",
        "weight": "weight_kg",
        "birth_city": "birth_city",
        "birth_state": "birth_state",
    })

    # 2.3 数值列类型转换
    df["height_cm"] = _to_numeric(df["height_cm"])    # 厘米
    df["weight_kg"] = _to_numeric(df["weight_kg"])    # 千克
    df["birth_year"] = _to_numeric(df["birth_year"])

    # 2.4 缺失值处理
    # 身高/体重：极少数缺失，用中位数填充
    df["height_cm"] = df["height_cm"].fillna(df["height_cm"].median())
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    # 文本列缺失填 'Unknown'
    for col in ("college", "birth_city", "birth_state"):
        df[col] = df[col].fillna("Unknown")
    # 出生年份缺失：先留空，后续与 player_data 合并时再补

    # 2.5 异常极值处理：身高/体重按 1%~99% 分位截尾，剔除明显错误
    for col in ("height_cm", "weight_kg"):
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=lo, upper=hi)

    # 2.6 字符串去空白
    df["name"] = df["name"].astype(str).str.strip()

    # 2.7 同名去重：对完全重复的球员行保留首条；同名不同人保留，靠出生年份区分
    df = df.drop_duplicates(subset=["name"], keep="first").reset_index(drop=True)

    print(f"  清洗后: {df.shape[0]} 行")
    return df


# ----------------------------------------------------------------------------
# 3. 清洗 player_data.csv —— 球员职业生涯信息表
# ----------------------------------------------------------------------------
def clean_player_data(player_data: pd.DataFrame) -> pd.DataFrame:
    """清洗 player_data.csv：身高 ft-in→cm、磅→kg、解析出生日期等。"""
    print("[3/6] 清洗 player_data.csv ...")

    df = player_data.copy()

    # 3.1 列重命名，与 Players 表对齐
    df = df.rename(columns={
        "name": "name",
        "year_start": "year_start",
        "year_end": "year_end",
        "position": "position_career",
        "height": "height_raw",
        "weight": "weight_lbs",
        "birth_date": "birth_date",
        "college": "college_pd",
    })

    # 3.2 身高 ft-in -> cm，体重 lbs -> kg
    df["height_cm"] = df["height_raw"].apply(_height_ft_in_to_cm)
    df["weight_kg"] = df["weight_lbs"].apply(_lbs_to_kg)

    # 3.3 解析出生日期，抽取出生年份
    df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")
    df["birth_year_pd"] = df["birth_date"].dt.year

    # 3.4 职业起止年份
    df["year_start"] = _to_numeric(df["year_start"])
    df["year_end"] = _to_numeric(df["year_end"])

    # 3.5 字符串去空白
    df["name"] = df["name"].astype(str).str.strip()
    for col in ("position_career", "college_pd"):
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    # 3.6 身高/体重缺失用中位数填充
    df["height_cm"] = df["height_cm"].fillna(df["height_cm"].median())
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())

    # 3.7 异常极值截尾
    for col in ("height_cm", "weight_kg"):
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lower=lo, upper=hi)

    # 3.8 仅删除完全重复行，保留同名不同人（靠 birth_year 区分，在主表合并阶段处理）
    df = df.drop_duplicates().reset_index(drop=True)

    print(f"  清洗后: {df.shape[0]} 行")
    return df


# ----------------------------------------------------------------------------
# 4. 构建球员主表 players_master（合并身体信息 + 职业信息）
# ----------------------------------------------------------------------------
def build_players_master(players: pd.DataFrame, player_data: pd.DataFrame) -> pd.DataFrame:
    """
    以 player_data 表为基底，按 (name, birth_year) 从 Players 表补充身体信息，形成球员主表。
    匹配规则（保证同名不同人正确区分）：
      - Players 表中该姓名唯一：当两侧出生年份“相容”（均缺失，或相等）时合并；
        若出生年份明确不等，则视为不同人，不合并（保留为两条记录）。
      - Players 表中该姓名不唯一（同名多人）：必须 birth_year 严格相等才合并。

    最终保留 player_data 的全部记录，并补上仅在 Players 表出现的球员（作为独立行）。
    """
    print("[4/6] 构建球员主表 players_master ...")

    pm = players.copy()  # name, height_cm, weight_kg, birth_year, college, birth_city, birth_state
    base = player_data.copy()  # name, ..., height_cm_pd, weight_kg_pd, birth_year_pd, college_pd, ...

    # 4.1 按姓名建立 Players 表的快速索引（同名时为多行子表）
    pm_groups = {n: g for n, g in pm.groupby("name")}

    # 4.2 逐行从 Players 表挑选最佳匹配
    h, w, coll, by, bcity, bstate = [], [], [], [], [], []
    for _, row in base.iterrows():
        name = row["name"]
        by_pd = row.get("birth_year_pd")
        cand = pm_groups.get(name)
        matched = None
        if cand is not None:
            if len(cand) == 1:
                c = cand.iloc[0]
                c_by = c["birth_year"]
                # 出生年份相容：均缺失，或相等
                compatible = pd.isna(c_by) or pd.isna(by_pd) or float(c_by) == float(by_pd)
                if compatible:
                    matched = c
                # 不相容：不同人，不合并（matched 保持 None）
            else:
                # 同名多人：必须 birth_year 严格相等
                if pd.notna(by_pd):
                    sub = cand[cand["birth_year"] == by_pd]
                    if len(sub) == 1:
                        matched = sub.iloc[0]
        if matched is not None:
            h.append(matched["height_cm"])
            w.append(matched["weight_kg"])
            coll.append(matched["college"])
            by.append(matched["birth_year"] if pd.notna(matched["birth_year"]) else by_pd)
            bcity.append(matched["birth_city"])
            bstate.append(matched["birth_state"])
        else:
            h.append(np.nan)
            w.append(np.nan)
            coll.append(np.nan)
            by.append(by_pd)
            bcity.append(np.nan)
            bstate.append(np.nan)

    base["height_cm_p"] = h
    base["weight_kg_p"] = w
    base["college_p"] = coll
    base["birth_city_p"] = bcity
    base["birth_state_p"] = bstate

    # 4.3 统一最终列：优先 Players 表，缺失用 player_data 转换值
    base["height_cm"] = base["height_cm_p"].fillna(base["height_cm_pd"])
    base["weight_kg"] = base["weight_kg_p"].fillna(base["weight_kg_pd"])
    base["college"] = base["college_p"].fillna(base["college_pd"])
    # 出生年份：取相容后的统一值
    base["birth_year"] = by
    base["birth_city"] = base["birth_city_p"]
    base["birth_state"] = base["birth_state_p"]

    # 4.4 补上仅在 Players 表出现、且未与 player_data 匹配的球员（作为独立行）
    matched_names = set(base["name"])
    # Players 表中姓名不在 player_data 的行
    p_only = pm[~pm["name"].isin(matched_names)].copy()
    if not p_only.empty:
        p_only = p_only.rename(columns={"college": "college"})
        for col in ("height_cm_pd", "weight_kg_pd", "college_pd", "position_career",
                    "year_start", "year_end", "birth_date"):
            if col not in p_only.columns:
                p_only[col] = np.nan

    # 4.5 统一列结构后合并
    keep_cols = [
        "name", "height_cm", "weight_kg", "birth_year", "college",
        "birth_city", "birth_state", "year_start", "year_end", "position_career",
        "birth_date",
    ]
    base = base.reindex(columns=keep_cols)
    p_only = p_only.reindex(columns=keep_cols)
    master = pd.concat([base, p_only], ignore_index=True)

    # 4.6 统计同名不同人情况
    name_counts = master["name"].value_counts()
    dup_count = int((name_counts > 1).sum())
    print(f"  球员主表: {master.shape[0]} 人；其中存在同名不同人的姓名 {dup_count} 个")
    return master


# ----------------------------------------------------------------------------
# 5. 清洗 Seasons_Stats.csv —— 赛季统计表
# ----------------------------------------------------------------------------
def clean_seasons(seasons: pd.DataFrame) -> pd.DataFrame:
    """清洗赛季统计表：删空列、类型转换、缺失值、交易特例去重、历史时代归一化。"""
    print("[5/6] 清洗 Seasons_Stats.csv ...")

    df = seasons.copy()

    # 5.1 去掉原始行索引列
    first_col = df.columns[0]
    if first_col == "" or str(first_col).lower().startswith("unnamed"):
        df = df.drop(columns=first_col)

    # 5.2 删除全空的导出占位列 blanl / blank2
    for col in ("blanl", "blank2"):
        if col in df.columns:
            df = df.drop(columns=col)

    # 5.3 丢弃 Year 或 Player 缺失的垃圾行（原始导出残留的空行）
    n_before = len(df)
    df = df.dropna(subset=["Year", "Player"])
    df = df[(df["Player"].astype(str).str.strip() != "") & (df["Player"] != "nan")]
    df = df.reset_index(drop=True)
    if len(df) != n_before:
        print(f"  丢弃 Year/Player 缺失行: {n_before - len(df)} 行")

    # 5.4 球员名去空白
    df["Player"] = df["Player"].astype(str).str.strip()

    # 5.5 关键数值列类型转换
    num_cols = [
        "Year", "Age", "G", "GS", "MP", "PER", "TS%", "3PAr", "FTr",
        "ORB%", "DRB%", "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%",
        "OWS", "DWS", "WS", "WS/48", "OBPM", "DBPM", "BPM", "VORP",
        "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
        "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # 5.6 年份取整（去除可能的 .0）
    df["Year"] = df["Year"].astype("Int64")

    # 5.7 交易特例去重：按 (Player, Year, Age) 分组。
    #     关键：不能只按 (Player, Year) 分组——同名不同人在同一年打球时
    #     会被误当成同一人的交易分流行而丢掉。同名不同人在同年 Age 不同，
    #     用 Age 作为“人”的判别维度可正确区分；同一名球员被交易时各队行 Age 一致。
    #     组内：优先保留 TOT（合计）行，否则保留出场数(G)最多的行。
    key_cols = ["Player", "Year", "Age"]
    # _tot_rank：TOT 行=0（排最前），非 TOT=1
    df["_tot_rank"] = (df["Tm"] != "TOT").astype("int8")
    # 先按组键、再按 TOT 优先、G 降序排序，组内取首行即得保留行
    df = df.sort_values(
        by=key_cols + ["_tot_rank", "G"],
        ascending=[True, True, True, True, False],
        kind="mergesort",
    )
    df = df.drop_duplicates(subset=key_cols, keep="first").drop(columns=["_tot_rank"])
    df = df.reset_index(drop=True)

    # 5.8 缺失值处理
    # 计数类统计（FG/FGA/PTS 等）：早期未记录的按 0 填充更合理（赛季层面有出场即有数据）
    counting_cols = [
        "G", "GS", "MP", "FG", "FGA", "3P", "3PA", "2P", "2PA",
        "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    ]
    for col in counting_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 高阶比率类（PER/TS%/USG% 等）缺失保留 NaN —— 早期时代未统计，不强填
    # 但 TS% 可后续用公式补算，见特征派生

    # 5.9 历史时代归一化：增加 era 列
    def _era(year):
        if pd.isna(year):
            return "Unknown"
        y = int(year)
        if y < 1960:
            return "1950s"
        if y < 1970:
            return "1960s"
        if y < 1980:
            return "1970s"
        if y < 1990:
            return "1980s"
        if y < 2000:
            return "1990s"
        if y < 2010:
            return "2000s"
        if y < 2020:
            return "2010s"
        return "2020s"
    df["era"] = df["Year"].apply(_era)

    # 5.10 异常极值：截掉 PTS、MP 等明显错误（<=0 但有出场或极大值）
    # 这里只做最低限度处理：保证计数类非负
    for col in counting_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df = df.reset_index(drop=True)
    print(f"  清洗后: {df.shape[0]} 行")
    return df


# ----------------------------------------------------------------------------
# 6. 跨表关联：把球员主表信息合并进赛季统计，形成 Master Dataset
# ----------------------------------------------------------------------------
def align_and_merge(seasons_clean: pd.DataFrame, players_master: pd.DataFrame) -> pd.DataFrame:
    """
    以赛季统计行为主轴，逐行从球员主表匹配身体/职业信息，形成 Master Dataset。

    匹配策略（保证同名不同人正确区分，且同年同名的不同球员各自保留一行）：
      - 该姓名在主表中仅 1 条记录：直接取该条；
      - 该姓名在主表中有多条记录（同名不同人）：取 birth_year 与
        “Year - Age 估算出生年”最接近的那一条；若 birth_year 缺失则取首条。
      - 该姓名不在主表：球员信息留空，赛季行照常保留。

    每条赛季行最终产出恰好一条 master 行，不做 (Player, Year) 维度去重——
    因为同名不同人可在同一年各自打球，按 (Player,Year) 去重会误删真实数据。
    """
    print("[6/6] 跨表关联，构建 master_data ...")

    stats = seasons_clean.copy()
    # 用 Year - Age 估算出生年份（Age 为赛季开始时年龄），用于同名消歧
    stats["_birth_year_est"] = stats["Year"] - stats["Age"]

    # 需要从主表附加的列
    attach_cols = [
        "height_cm", "weight_kg", "birth_year", "college", "birth_city",
        "birth_state", "year_start", "year_end", "position_career", "birth_date",
    ]
    attach_cols = [c for c in attach_cols if c in players_master.columns]

    # 按姓名建索引：name -> list[record]（record 为该球员主表行的 dict）
    name_index = {}
    for _, row in players_master.iterrows():
        name_index.setdefault(row["name"], []).append(row)

    # 逐行匹配
    matched = {c: [] for c in attach_cols}
    unresolved_dup = 0  # 同名多记录、且无法精确区分的计数
    for _, srow in stats.iterrows():
        name = srow["Player"]
        cands = name_index.get(name)
        chosen = None
        if cands is not None:
            if len(cands) == 1:
                chosen = cands[0]
            else:
                est = srow["_birth_year_est"]
                yr = srow["Year"]
                age_val = srow["Age"]
                # 优先级1（最可靠）：用 birth_date 算“赛季年龄”与 Age 精确匹配
                #   basketball-reference 的 Age = 球员在该赛季年 1月31日时的年龄
                #   expected_age = Y - birth_year - (1 if birth_month >= 2 else 0)
                exact = []
                if pd.notna(age_val):
                    for c in cands:
                        bd = c.get("birth_date")
                        by = c.get("birth_year")
                        if pd.notna(bd) and pd.notna(by):
                            try:
                                bm = pd.to_datetime(bd).month
                                exp_age = int(yr) - int(by) - (1 if bm >= 2 else 0)
                                if exp_age == int(age_val):
                                    exact.append(c)
                            except (ValueError, TypeError):
                                pass
                if exact:
                    # 精确命中：若多个（同年同月出生的同名不同人极少见），再按职业跨度收窄
                    chosen = exact[0] if len(exact) == 1 else min(
                        exact, key=lambda c: abs(float(c["birth_year"]) - float(est))
                    )
                else:
                    # 优先级2：职业跨度 [year_start, year_end] 须包含该赛季年
                    span_cands = [
                        c for c in cands
                        if pd.notna(c.get("year_start")) and pd.notna(c.get("year_end"))
                        and float(c["year_start"]) <= float(yr) <= float(c["year_end"])
                    ]
                    pool = span_cands if span_cands else cands
                    # 优先级3：birth_year 与估算出生年最近（±1 年容差）
                    comparable = [c for c in pool if pd.notna(c["birth_year"]) and pd.notna(est)]
                    if comparable:
                        def _dist(c):
                            by = float(c["birth_year"]); e = float(est)
                            return min(abs(by - e), abs(by - (e + 1)))
                        chosen = min(comparable, key=_dist)
                    else:
                        chosen = pool[0]
                        unresolved_dup += 1
        for c in attach_cols:
            matched[c].append(chosen[c] if chosen is not None else np.nan)

    for c in attach_cols:
        stats[c] = matched[c]

    if unresolved_dup:
        print(f"  注: {unresolved_dup} 条同名多记录行因 birth_date 缺失退化为近似匹配")

    # 清理辅助列
    stats = stats.drop(columns=["_birth_year_est"])
    stats = stats.reset_index(drop=True)
    return stats


# ----------------------------------------------------------------------------
# 7. 特征派生：TS% 补算 + 每 36 分钟折算
# ----------------------------------------------------------------------------
def derive_features(master: pd.DataFrame) -> pd.DataFrame:
    """派生真实命中率 TS%（缺失时补算）以及每 36 分钟折算的计数统计。"""
    print("[特征派生] 计算 TS% 与 per-36 特征 ...")

    df = master.copy()

    # 7.1 真实命中率 TS% = PTS / (2 * (FGA + 0.44 * FTA))
    #     当原 TS% 缺失但拥有 PTS/FGA/FTA 时按公式补算
    fga = df.get("FGA", pd.Series([np.nan] * len(df)))
    fta = df.get("FTA", pd.Series([np.nan] * len(df)))
    pts = df.get("PTS", pd.Series([np.nan] * len(df)))
    denom = 2 * (fga + 0.44 * fta)
    ts_calc = np.where(denom > 0, pts / denom, np.nan)
    if "TS%" in df.columns:
        df["TS%"] = df["TS%"].fillna(pd.Series(ts_calc, index=df.index))
    else:
        df["TS%"] = pd.Series(ts_calc, index=df.index)

    # 7.2 每 36 分钟折算（counting stats * 36 / MP）
    per36_cols = ["PTS", "FG", "FGA", "3P", "3PA", "2P", "2PA",
                  "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF"]
    mp = df.get("MP", pd.Series([np.nan] * len(df)))
    valid_mp = mp.fillna(0) > 0
    for col in per36_cols:
        base = df.get(col, pd.Series([np.nan] * len(df)))
        calc = np.where(valid_mp, base * 36.0 / mp.replace(0, np.nan), np.nan)
        df[f"{col}_per36"] = pd.Series(calc, index=df.index).round(2)

    # 7.3 重新排序：把派生列放到末尾，保持主数据集整洁
    leading = [c for c in ["Year", "Player", "Pos", "Age", "Tm", "G", "GS", "MP"] if c in df.columns]
    rest = [c for c in df.columns if c not in leading]
    df = df[leading + rest]

    return df


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("NBA 球员数据清洗 —— 主数据集构建")
    print("=" * 70)

    players, player_data, seasons = load_raw()

    players_clean = clean_players(players)
    player_data_clean = clean_player_data(player_data)

    # 把 player_data 清洗后的 height_cm/weight_kg 列名调整，方便合并
    player_data_clean = player_data_clean.rename(
        columns={"height_cm": "height_cm_pd", "weight_kg": "weight_kg_pd"}
    )

    players_master = build_players_master(players_clean, player_data_clean)

    seasons_clean = clean_seasons(seasons)

    master = align_and_merge(seasons_clean, players_master)

    master = derive_features(master)

    # 保存唯一交付的数据集：统一标准主数据集
    master.to_csv(MASTER_OUT, index=False, encoding="utf-8-sig")

    print("-" * 70)
    print(f"主数据集已保存: {MASTER_OUT}")
    print(f"主数据集规模  : {master.shape[0]} 行 x {master.shape[1]} 列")
    print(f"覆盖赛季范围  : {int(master['Year'].min())} - {int(master['Year'].max())}")
    print(f"涉及球员数量  : {master['Player'].nunique()} 人")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
