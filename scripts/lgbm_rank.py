#!/usr/bin/env python3
"""量化黑盒: LightGBM 排序模型 (LGBMRanker / lambdarank) —— 候选股买入优先级评分。

方案: 腾讯文档《基于 LightGBM 排序模型的 A 股短期选股方案》(docs.qq.com/doc/DWHpySUZTa0dsa0ti)
与 quality_kdj5 的关系: 选股条件完全一致(同一条动量轮动过滤), 本模块只替换
「同日候选谁先买」的打分排队 —— 排序学习直接优化相对次序, 更契合每日取前 3 名的组合构建。

防泄漏三条线:
  1) 特征只用 T 日及以前收盘数据 (与动量策略同口径: T 收盘打分 -> T+1 开盘执行);
  2) 训练样本的标签(未来 5 日收益)必须在预测块开始前 ≥6 个交易日完成 (label purge);
  3) 全历史分数一律 walk-forward 产出: 每个预测块用其之前的数据训练, 无样本内分数。

产出: data/meta/lgbm_scores.parquet (code,date,score, float32) + lgbm_model.txt + lgbm_model_meta.json
每日增量: 新日期用最近模型打分; 模型落后超过 REFIT_EVERY 个交易日自动重训 (walk-forward 语义不变)。
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")
import signals as S  # noqa: E402  (load_klines / build_indicators / groll / gshift 复用)

SCORES_FILE = f"{S.BASE}/data/meta/lgbm_scores.parquet"
MODEL_FILE = f"{S.BASE}/data/meta/lgbm_model.txt"
MODEL_META = f"{S.BASE}/data/meta/lgbm_model_meta.json"
REFIT_EVERY = 42          # 每隔 N 个交易日重训一次 (walk-forward 滚动步长)
TRAIN_FROM = "2023-01-01"  # 训练数据起点 (K线库回补起点)
LABEL_N = 5               # 标签: 未来 5 日收益分位
LABEL_PURGE = 6           # 预测块开始前 N 个交易日内的样本不进训练集 (标签未完成)

FEATURES = [
    # 价量 (hfq 收盘口径)
    "ret1", "ret3", "ret5", "ret10", "ret20", "ret60",
    # 波动率
    "vol5", "vol20", "vol60", "dvol20", "skew20", "kurt20",
    # 量能
    "vol_ratio", "vma_ratio", "amt_ratio",
    # 日内结构
    "amp", "amp20", "intraday", "overnight",
    # 均线/技术形态
    "bias20", "bias60", "rsi14", "macd_hist", "win20", "win60",
    "sharpe20", "trend_streak", "kdj_j", "j_dev", "illiq20",
    # 横截面排名特征 (文档: cross_sectional_rank 是最重要特征)
    "rk_ret5", "rk_ret20", "rk_vol20", "rk_amt20", "rk_sharpe20",
]


def build_features() -> pd.DataFrame:
    """在 (code,date) MultiIndex 上构建全部特征。复用 signals.build_indicators 的基础指标。"""
    hfq = S.load_klines("hfq")
    raw = S.load_klines("raw")
    df = S.build_indicators(hfq, raw)
    g, gs = S.groll, S.gshift
    df["ret1"] = df["close"] / gs(df["close"], 1) - 1
    df["ret3"] = df["close"] / gs(df["close"], 3) - 1
    df["ret5"] = df["close"] / gs(df["close"], 5) - 1
    df["ret10"] = df["close"] / gs(df["close"], 10) - 1
    df["ret60"] = df["close"] / gs(df["close"], 60) - 1
    df["vol5"] = g(df["ret"], 5, "std")
    df["vol60"] = g(df["ret"], 60, "std")
    neg = df["ret"].where(df["ret"] < 0)
    df["dvol20"] = g(neg, 20, "std")
    df["skew20"] = g(df["ret"], 20, "skew")
    df["kurt20"] = g(df["ret"], 20, "kurt")
    df["vol_ratio"] = df["volume"] / df["vol_ma5"].replace(0, np.nan)
    df["vma_ratio"] = df["vol_ma5"] / df["vol_ma20"].replace(0, np.nan)
    df["amt_ratio"] = df["amt"] / df["amt20"].replace(0, np.nan)
    df["amp"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["amp20"] = g(df["amp"], 20, "mean")
    df["intraday"] = df["close"] / df["open"] - 1
    df["overnight"] = df["open"] / gs(df["close"], 1) - 1
    df["bias20"] = df["close"] / df["ma20"] - 1
    df["bias60"] = df["close"] / df["ma60"] - 1
    # RSI14 (Wilder 平滑) / MACD 柱归一化
    up = df["ret"].clip(lower=0)
    dn = (-df["ret"]).clip(lower=0)
    au = up.groupby(level=0, sort=False).transform(lambda x: x.ewm(alpha=1 / 14, adjust=False).mean())
    ad = dn.groupby(level=0, sort=False).transform(lambda x: x.ewm(alpha=1 / 14, adjust=False).mean())
    df["rsi14"] = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    e12 = df["close"].groupby(level=0, sort=False).transform(lambda x: x.ewm(span=12, adjust=False).mean())
    e26 = df["close"].groupby(level=0, sort=False).transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = e12 - e26
    sig9 = macd.groupby(level=0, sort=False).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = (macd - sig9) / df["close"] * 100
    df["win20"] = g((df["ret"] > 0).astype(float), 20, "mean")
    rk = lambda s: s.groupby(level=1).rank(pct=True)
    df["rk_ret5"] = rk(df["ret5"])
    df["rk_ret20"] = rk(df["ret20"])
    df["rk_vol20"] = rk(df["vol20"])
    df["rk_amt20"] = rk(df["amt20"])
    df["rk_sharpe20"] = rk(df["sharpe20"])
    return df[FEATURES]


def tradable_mask(index: pd.MultiIndex) -> np.ndarray:
    """与 run_scan 同口径的可交易宇宙: 剔 ST 日行 / 退市 [out_date-30天,∞) 行 / 无 out_date 的"退"股。"""
    basic = pd.read_parquet(f"{S.BASE}/data/meta/stock_basic.parquet")
    st = pd.read_parquet(f"{S.BASE}/data/meta/st_history.parquet")
    if "out_date" in basic.columns:
        _cut = pd.to_datetime(basic["out_date"], errors="coerce") - pd.Timedelta(days=30)
        cut_map = pd.Series(_cut.to_numpy(), index=basic["secid"]).dropna()
    else:
        cut_map = pd.Series(dtype="datetime64[ns]")
    retire = set(basic.loc[~basic["secid"].isin(cut_map.index)
                           & basic["name"].str.contains("退", na=False), "secid"])
    idx_df = index.to_frame(index=False)
    idx_df = idx_df.merge(st.loc[st["is_st"] == 1, ["code", "date"]].assign(_bad=True),
                          on=["code", "date"], how="left")
    bad = idx_df["_bad"].fillna(False).astype(bool).to_numpy()
    dates = index.get_level_values(1).to_numpy()
    cut_for = pd.Series(index.get_level_values(0)).map(cut_map).to_numpy()
    bad_retire = pd.notna(cut_for) & (dates >= cut_for)
    in_retire = index.get_level_values(0).isin(retire)
    return ~(bad | bad_retire | in_retire)


def _params():
    import lightgbm as lgb  # noqa: F401
    return dict(objective="lambdarank", metric="ndcg", boosting_type="gbdt",
                n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15,
                min_child_samples=50, subsample=0.8, bagging_freq=1,
                colsample_bytree=0.8,
                label_gain=[0, 1, 3, 7, 15], lambdarank_truncation_level=30,
                random_state=42, n_jobs=-1, verbosity=-1)


def _make_labels(feats: pd.DataFrame, close: pd.Series) -> pd.Series:
    """标签: 未来 LABEL_N 日收益按日分位 5 档 (0..4, 越大越强)。fut 收益用后复权收盘。"""
    fut = close.groupby(level=0).shift(-LABEL_N) / close - 1
    lab = fut.groupby(level=1).transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    return lab


def _fit_model(train_X, train_y, train_g, valid_X, valid_y, valid_g):
    import lightgbm as lgb
    m = lgb.LGBMRanker(**_params())
    m.fit(train_X, train_y, group=train_g,
          eval_set=[(valid_X, valid_y)], eval_group=[valid_g], eval_at=[3],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m


def _rank_data(sub: pd.DataFrame, lab: pd.Series):
    """按日分组: group 数组 + 5 档标签; 标签缺失行剔除。"""
    df = sub.copy()
    df["_lab"] = lab.reindex(df.index)
    df = df[df["_lab"].notna()]
    df["_lab"] = df["_lab"].astype(int)
    groups = df.groupby(level=1, sort=False).size().to_numpy()
    return df[FEATURES], df["_lab"], groups


def walk_forward_scores(feats: pd.DataFrame, close: pd.Series,
                        block_starts: list[pd.Timestamp],
                        train_from: str = TRAIN_FROM) -> tuple[pd.DataFrame, object]:
    """对每个预测块: 用块开始日前 LABEL_PURGE 个交易日之前、标签已完成的样本训练,
    预测块内全部 (code,date)。返回 (scores_df, 最后一个模型)。"""
    lab = _make_labels(feats, close)
    all_dates = feats.index.get_level_values(1).unique().sort_values()
    train_pool = all_dates[(all_dates >= pd.Timestamp(train_from))]
    out = []
    model = None
    for bi, t in enumerate(block_starts):
        # 训练集: [train_from, t 前第 LABEL_PURGE 个交易日]
        prior = train_pool[train_pool < t]
        if len(prior) <= LABEL_PURGE + 40:
            print(f"[lgbm] 块 {t:%Y-%m-%d} 训练样本不足, 跳过 (保持无分数)", flush=True)
            continue
        train_dates = prior[:len(prior) - LABEL_PURGE]
        tr = feats[feats.index.get_level_values(1).isin(train_dates)]
        tr_X, tr_y, tr_g = _rank_data(tr, lab)
        # 早停验证集: 训练期最后 15% 的日期, 拟合集为前 85%
        cut = max(int(len(train_dates) * 0.85), 1)
        fX, fy, fg = _rank_data(
            feats[feats.index.get_level_values(1).isin(train_dates[:cut])], lab)
        vX, vy, vg = _rank_data(
            feats[feats.index.get_level_values(1).isin(train_dates[cut:])], lab)
        model = _fit_model(fX, fy, fg, vX, vy, vg)
        blk = feats[feats.index.get_level_values(1).isin(
            all_dates[(all_dates >= t) & (all_dates < (block_starts[bi + 1] if bi + 1 < len(block_starts) else pd.Timestamp.max))])]
        pred = model.predict(blk[FEATURES])
        out.append(pd.DataFrame({
            "code": blk.index.get_level_values(0),
            "date": blk.index.get_level_values(1),
            "score": pred.astype(np.float32)}))
        print(f"[lgbm] 块 {t:%Y-%m-%d}: 训练 {len(fX):,} 行/{len(train_dates)} 天 "
              f"(best_iter={model.best_iteration_}) -> 预测 {len(blk):,} 行", flush=True)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["code", "date", "score"]), model


def _full_walk_forward(feats: pd.DataFrame, close: pd.Series) -> tuple[pd.DataFrame, object]:
    """全历史 OOS: 从 START 起, 每 REFIT_EVERY 个交易日一个块。"""
    all_dates = feats.index.get_level_values(1).unique().sort_values()
    starts = all_dates[(all_dates >= pd.Timestamp(S.START))][::REFIT_EVERY].tolist()
    if not starts or starts[0] > pd.Timestamp(S.START):
        starts = [pd.Timestamp(S.START)] + starts
    return walk_forward_scores(feats, close, starts)


def update_scores() -> pd.DataFrame:
    """入口: 无分数文件 -> 全历史 walk-forward; 有 -> 增量为新日期打分 (必要时重训)。"""
    t0 = time.time()
    feats = build_features()
    mask = tradable_mask(feats.index)
    feats = feats[mask]
    # 标签基准: 后复权收盘 (build_indicators 的 close 已被裁掉, 从源数据重建并重对齐)
    hfq = S.load_klines("hfq")
    hfq["date"] = pd.to_datetime(hfq["date"])
    hfq_close = hfq.set_index(["code", "date"]).sort_index()["close"].reindex(feats.index)

    if not os.path.exists(SCORES_FILE):
        print("[lgbm] 无历史分数文件, 执行全历史 walk-forward (一次性, 较慢)...", flush=True)
        scores, model = _full_walk_forward(feats, hfq_close)
        scores = scores.drop_duplicates(["code", "date"]).sort_values(["date", "score"], ascending=[True, False])
        scores.to_parquet(SCORES_FILE, index=False)
        if model is not None:
            model.booster_.save_model(MODEL_FILE)
            _save_meta(model, feats)
        _summary(scores, feats, hfq_close)
        print(f"[lgbm] 全历史分数完成: {len(scores):,} 行, 耗时 {time.time()-t0:.0f}s", flush=True)
        return scores

    old = pd.read_parquet(SCORES_FILE)
    old["date"] = pd.to_datetime(old["date"])
    last_scored = old["date"].max()
    all_dates = feats.index.get_level_values(1).unique().sort_values()
    missing = all_dates[all_dates > last_scored]
    if len(missing) == 0:
        print(f"[lgbm] 分数已是最新 ({last_scored:%Y-%m-%d}), 无需更新", flush=True)
        return old
    # 增量块: 从 last_scored 后一天起按 REFIT_EVERY 切块, 每块独立训练 (walk-forward 语义)
    starts = missing[::REFIT_EVERY].tolist()
    if starts[0] != missing[0]:
        starts = [missing[0]] + starts
    inc, model = walk_forward_scores(feats, hfq_close, starts)
    scores = pd.concat([old, inc], ignore_index=True) if len(inc) else old
    scores = scores.drop_duplicates(["code", "date"], keep="last").sort_values(["date", "score"], ascending=[True, False])
    scores.to_parquet(SCORES_FILE, index=False)
    if model is not None:
        model.booster_.save_model(MODEL_FILE)
        _save_meta(model, feats)
    _summary(scores, feats, hfq_close)
    print(f"[lgbm] 增量打分 {len(inc):,} 行 ({missing[0]:%Y-%m-%d} ~ {missing[-1]:%Y-%m-%d}), "
          f"耗时 {time.time()-t0:.0f}s", flush=True)
    return scores


def _save_meta(model, feats):
    with open(MODEL_META, "w") as f:
        json.dump({"features": FEATURES, "n_features": len(FEATURES),
                   "best_iteration": int(model.best_iteration_ or 0),
                   "updated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}, f, ensure_ascii=False, indent=1)


def _summary(scores: pd.DataFrame, feats: pd.DataFrame, close: pd.Series):
    """Rank IC 摘要 (评估排序能力; 写 CI 日志用)。"""
    from scipy.stats import spearmanr
    m = scores.merge(close.rename("close").reset_index(), on=["code", "date"], how="inner")
    m = m.sort_values(["code", "date"])
    m["fut5"] = m.groupby("code")["close"].shift(-5) / m["close"] - 1
    m = m[m["fut5"].notna()]
    ics = m.groupby("date").apply(lambda g: spearmanr(g["score"], g["fut5"]).correlation
                                  if len(g) > 10 else np.nan).dropna()
    if len(ics):
        print(f"[lgbm] Rank IC: mean={ics.mean():.4f} ICIR={ics.mean()/ics.std():.2f} "
              f"({len(ics)} 天, 全期含训练边界块)", flush=True)


if __name__ == "__main__":
    update_scores()
