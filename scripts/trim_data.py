#!/usr/bin/env python3
"""数据滚动清理, 保证仓库不无限增大。

默认清理(每次运行):
  - data/pool/zt_*.parquet / zb_*.parquet: 删除滚动窗口外的日期文件
  - data/kline/incremental/: 删除窗口外的单日增量文件
可选(仅当 TRIM_KLINE=1 时, 会截断回测历史):
  - data/kline 主分片/fixup: 保留最近 KEEP_DAYS 个交易日的行, 空分片删除

说明: data/meta 下 flags_long/signals/回测结果均为派生数据(不入库, 每日全量重算),
因此入库体量只由 K 线原料决定: 增量文件 ~0.2MB/交易日, 主分片固定。
"""
import glob
import os
import sys

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META, KDIR, POOL = f"{BASE}/data/meta", f"{BASE}/data/kline", f"{BASE}/data/pool"
KEEP_DAYS = int(os.environ.get("KEEP_DAYS", "500"))       # 滚动窗口(交易日)
TRIM_KLINE = os.environ.get("TRIM_KLINE", "0") == "1"     # 是否裁剪主分片(截断回测历史, 默认关)


def cutoff_date():
    idx = pd.read_parquet(f"{META}/index_daily.parquet")
    return idx["date"].iloc[-KEEP_DAYS]


def trim_pool(cut):
    if not os.path.isdir(POOL):
        return 0
    n = 0
    for f in glob.glob(f"{POOL}/*.parquet"):
        d = os.path.basename(f).split("_")[1][:8]
        if d < cut.strftime("%Y%m%d"):
            os.remove(f)
            n += 1
    return n


def trim_incremental(cut):
    n = 0
    for f in glob.glob(f"{KDIR}/incremental/*.parquet"):
        d = os.path.basename(f).split("_")[1][:8]
        if d < cut.strftime("%Y%m%d"):
            os.remove(f)
            n += 1
    return n


def trim_shards(cut):
    """裁剪主分片与 fixup 到滚动窗口 (TRIM_KLINE=1 时才调用)"""
    n = 0
    for f in sorted(glob.glob(f"{KDIR}/qfq_*.parquet") + glob.glob(f"{KDIR}/raw_*.parquet")
                    + glob.glob(f"{KDIR}/hfq_*.parquet") + glob.glob(f"{KDIR}/fixup/*.parquet")):
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        kept = df[df["date"] >= cut]
        if kept.empty:
            os.remove(f)
        elif len(kept) < len(df):
            kept.to_parquet(f, index=False)
        n += 1
    return n


if __name__ == "__main__":
    cut = cutoff_date()
    print(f"滚动窗口: 保留 {cut:%Y-%m-%d} 之后 (最近 {KEEP_DAYS} 个交易日)", flush=True)
    n = trim_pool(cut) + trim_incremental(cut)
    print(f"已清理窗口外 pool/增量文件: {n} 个", flush=True)
    if TRIM_KLINE:
        n = trim_shards(cut)
        print(f"已裁剪主分片/fixup: {n} 个", flush=True)
    size = sum(os.path.getsize(p) for p in glob.glob(f"{BASE}/data/**/*", recursive=True) if os.path.isfile(p))
    print(f"当前 data/ 总体量: {size/1e6:.0f} MB", flush=True)
