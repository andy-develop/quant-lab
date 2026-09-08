import os
#!/usr/bin/env python3
"""baostock 后复权(hfq) K线回补 —— 信号口径切换专用。
hfq 历史值永久冻结(不像 qfq 随最新价整体缩放), 保证信号可复现。
用法: python backfill_hfq.py <worker_id> <total_workers>
输出: data/kline/hfq_b{i}_{n}.parquet (每200只一个分片, 保留 sh./sz. 前缀)
断点: data/meta/hfq_progress_{i}.json
"""
import datetime
import json, os, sys, time
import pandas as pd
import baostock as bs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDIR, META = f"{BASE}/data/kline", f"{BASE}/data/meta"
START = "2023-01-01"
END = datetime.date.today().strftime("%Y-%m-%d")  # 动态截止, 避免硬编码过期后回补出静默缺口
FIELDS_Q = "date,open,high,low,close"


def query(bs_mod, code):
    rs = bs_mod.query_history_k_data_plus(code, FIELDS_Q, start_date=START, end_date=END,
                                          frequency="d", adjustflag="1")  # 1=后复权
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=FIELDS_Q.split(","))
    for c in FIELDS_Q.split(",")[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def main():
    wid, total = int(sys.argv[1]), int(sys.argv[2])
    basic = pd.read_parquet(f"{META}/stock_basic.parquet")
    basic = basic[~basic["name"].str.contains("ST|退", na=False)]
    basic = basic[(basic["status"] == "1") | (basic["out_date"] >= pd.Timestamp(START))]
    codes = sorted(basic["code"].tolist())          # baostock 格式 sh.600000
    mine = codes[wid::total]

    done = set()
    prog_path = f"{META}/hfq_progress_{wid}.json"
    if os.path.exists(prog_path):
        done = set(json.load(open(prog_path)))
    todo = [c for c in mine if c not in done]

    print(f"[hfq-{wid}] 分到 {len(mine)}, 已完成 {len(done & set(mine))}, 待拉 {len(todo)}", flush=True)
    bs.login()
    shard_no = len([f for f in os.listdir(KDIR) if f.startswith(f"hfq_b{wid}_")])
    buf = []
    t_start, n_ok, n_fail = time.time(), 0, 0

    def flush(force=False):
        nonlocal shard_no
        if buf and (len(buf) >= 200 or force):
            pd.concat(buf).to_parquet(f"{KDIR}/hfq_b{wid}_{shard_no:02d}.parquet", index=False)
            buf.clear()
            shard_no += 1

    for k, code in enumerate(todo):
        df = query(bs, code)
        if df is None:
            n_fail += 1
        else:
            df.insert(0, "code", code)
            buf.append(df)
            n_ok += 1
            done.add(code)
        flush()
        if (k + 1) % 100 == 0:
            json.dump(sorted(done), open(prog_path, "w"))
            el = time.time() - t_start
            eta = el / (k + 1) * (len(todo) - k - 1)
            print(f"[hfq-{wid}] {k+1}/{len(todo)} ok={n_ok} fail={n_fail} "
                  f"平均 {el/(k+1):.2f}s/只 ETA {eta/60:.0f}min", flush=True)
        if (k + 1) % 1000 == 0:
            time.sleep(5)
    flush(force=True)
    json.dump(sorted(done), open(prog_path, "w"))
    print(f"[hfq-{wid}] 完成: ok={n_ok} fail={n_fail} 分片={shard_no}", flush=True)
    bs.logout()


if __name__ == "__main__":
    main()
