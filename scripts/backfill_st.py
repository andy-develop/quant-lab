#!/usr/bin/env python3
"""baostock 逐日 ST 状态(isST)回补 —— 消除 ST 过滤的未来函数。

背景: 用"当前股票名称含 ST"过滤整个回测窗口是未来函数 —— 2024 年戴帽的股票
会被错误地从 2023 年的信号里剔除, 摘帽股会被错误地保留。
baostock query_history_k_data_plus 的 isST 字段提供逐日历史状态, 与 K 线同源。

用法:
  python backfill_st.py <worker_id> <total_workers>   # 分片回补, 输出 st_part_{wid}.parquet
  python backfill_st.py merge                          # 合并分片 -> data/meta/st_history.parquet
  python backfill_st.py incw <wid> <total>             # 增量worker: 从最新日期往前10天起补自己的分片
  python backfill_st.py incmerge                       # 合并增量 -> st_history.parquet (删除增量分片)

输出: data/meta/st_history.parquet (code=1.600000 腾讯格式, date, is_st int8)
断点: data/meta/st_progress_{wid}.json
"""
import datetime
import glob
import json
import os
import sys
import time

import baostock as bs
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META = f"{BASE}/data/meta"
OUT = f"{META}/st_history.parquet"
START = "2023-01-01"
END = datetime.date.today().strftime("%Y-%m-%d")  # 动态截止
FIELDS_Q = "date,code,isST"


def norm_code(s: pd.Series) -> pd.Series:
    return s.str.replace("sh.", "1.", regex=False).str.replace("sz.", "0.", regex=False)


def query(bs_mod, code: str, start: str, end: str) -> pd.DataFrame | None:
    rs = bs_mod.query_history_k_data_plus(code, FIELDS_Q, start_date=start, end_date=end, frequency="d")
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=FIELDS_Q.split(","))
    df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype("int8")
    return df[["date", "isST"]]


def all_store_codes() -> list[str]:
    """K线库中出现过的全部代码(归一为 baostock 格式) —— 含已退市股, 天然是完整回测宇宙"""
    codes: set[str] = set()
    for f in glob.glob(f"{BASE}/data/kline/hfq_*.parquet") + glob.glob(f"{BASE}/data/kline/incremental/hfq_*.parquet"):
        codes.update(pd.read_parquet(f, columns=["code"])["code"].unique())
    out = []
    for c in codes:
        if c.startswith(("sh.", "sz.")):
            out.append(c)
        elif c.startswith("1."):
            out.append("sh." + c[2:])
        elif c.startswith("0."):
            out.append("sz." + c[2:])
    return sorted(set(out))


def backfill_worker(wid: int, total: int) -> None:
    codes = all_store_codes()
    mine = codes[wid::total]
    part_path = f"{META}/st_part_{wid}.parquet"
    prog_path = f"{META}/st_progress_{wid}.json"

    done: set[str] = set()
    frames: list[pd.DataFrame] = []
    if os.path.exists(prog_path) and os.path.exists(part_path):
        with open(prog_path) as f:
            done = set(json.load(f))
        old = pd.read_parquet(part_path)
        frames.append(old)  # 已完成的直接续用
    todo = [c for c in mine if c not in done]
    print(f"[st-{wid}] 分到 {len(mine)}, 已完成 {len(done)}, 待拉 {len(todo)}", flush=True)

    bs.login()
    t0 = time.time()
    n_ok = n_fail = 0
    for k, code in enumerate(todo):
        df = query(bs, code, START, END)
        if df is None:
            n_fail += 1  # 无数据(未上市/退市早于窗口)也记入 done, 避免重复拉
        else:
            df.insert(0, "code", code)
            frames.append(df)
            n_ok += 1
        done.add(code)
        if (k + 1) % 200 == 0 or k + 1 == len(todo):
            if frames:  # 全部 None 时(如代码格式错误)跳过写盘, 不让 concat 崩掉
                pd.concat(frames, ignore_index=True).to_parquet(part_path, index=False)
            with open(prog_path, "w") as f:
                json.dump(sorted(done), f)
            el = time.time() - t0
            eta = el / (k + 1) * (len(todo) - k - 1)
            print(f"[st-{wid}] {k+1}/{len(todo)} ok={n_ok} fail={n_fail} "
                  f"平均 {el/(k+1):.2f}s/只 ETA {eta/60:.0f}min", flush=True)
        if (k + 1) % 1000 == 0:
            time.sleep(5)
    bs.logout()
    print(f"[st-{wid}] 完成: ok={n_ok} fail={n_fail} -> {part_path}", flush=True)


def merge_parts() -> None:
    parts = sorted(glob.glob(f"{META}/st_part_*.parquet"))
    if not parts:
        raise SystemExit("没有 st_part_*.parquet, 先分片回补")
    df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
    _save(df, "merge")


def incremental_worker(wid: int, total: int) -> None:
    """增量分片: 各 worker 只拉 [最新日期-10天, 今天] 的自己那份"""
    hist = pd.read_parquet(OUT)
    last = pd.to_datetime(hist["date"]).max()
    start = (last - pd.Timedelta(days=10)).strftime("%Y-%m-%d")  # 10天重叠, 覆盖节假日+baostock延迟
    mine = sorted(hist["code"].unique())[wid::total]
    print(f"[st-inc-{wid}] {len(mine)} 只, 起点 {start}", flush=True)
    bs.login()
    frames: list[pd.DataFrame] = []
    for k, code in enumerate(mine):
        df = query(bs, code, start, END)
        if df is not None:
            df.insert(0, "code", code)
            frames.append(df)
        if (k + 1) % 500 == 0:
            print(f"[st-inc-{wid}] {k+1}/{len(mine)}", flush=True)
    bs.logout()
    if frames:
        pd.concat(frames, ignore_index=True).to_parquet(f"{META}/st_inc_part_{wid}.parquet", index=False)
    print(f"[st-inc-{wid}] 完成 {len(frames)} 只有数据", flush=True)


def inc_merge() -> None:
    parts = sorted(glob.glob(f"{META}/st_inc_part_*.parquet"))
    hist = pd.read_parquet(OUT)
    frames = [hist] + [pd.read_parquet(f) for f in parts]
    _save(pd.concat(frames, ignore_index=True), "inc-merge")
    for f in parts:
        os.remove(f)


def _save(df: pd.DataFrame, tag: str) -> None:
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = norm_code(df["code"])
    if "isST" in df.columns:  # 兼容增量件(isST)与历史件(is_st)混合
        if "is_st" in df.columns:
            df["isST"] = df["isST"].fillna(df["is_st"])
        df["is_st"] = df["isST"].astype("int8")
    df = df.drop_duplicates(["code", "date"], keep="last").sort_values(["code", "date"]).reset_index(drop=True)
    df[["code", "date", "is_st"]].to_parquet(OUT, index=False)
    n_st = int((df["is_st"] == 1).sum())
    print(f"[st-{tag}] 保存 {OUT}: {len(df):,} 行 / {df['code'].nunique()} 只 / "
          f"ST日 {n_st:,} / {df['code'].loc[df['is_st']==1].nunique()} 只有过ST / 最新 {df['date'].max():%Y-%m-%d}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge_parts()
    elif sys.argv[1] == "incw":
        incremental_worker(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "incmerge":
        inc_merge()
    else:
        backfill_worker(int(sys.argv[1]), int(sys.argv[2]))
