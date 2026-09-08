import os
#!/usr/bin/env python3
"""baostock 历史K线回补 (4进程并行, 每进程独立TCP连接)。
用法: python backfill_baostock.py <worker_id> <total_workers>
输出: data/kline/raw_b{i}_{n}.parquet / qfq_b{i}_{n}.parquet (每200只一个分片)
断点: data/meta/bs_progress_{i}.json
"""
import datetime
import json, os, sys, time
import pandas as pd
import baostock as bs

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
KDIR, META = f"{BASE}/data/kline", f"{BASE}/data/meta"
START = "2023-09-01"
END = datetime.date.today().strftime("%Y-%m-%d")  # 动态截止, 避免硬编码过期后回补出静默缺口
FIELDS = "date,open,high,low,close,volume,amount"
FIELDS_Q = "date,open,high,low,close"


def query(bs_mod, code, adjust):
    flds = FIELDS if adjust == "3" else FIELDS_Q
    rs = bs_mod.query_history_k_data_plus(code, flds, start_date=START, end_date=END,
                                          frequency="d", adjustflag=adjust)
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=flds.split(","))
    for c in flds.split(",")[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).reset_index(drop=True)


def main():
    wid, total = int(sys.argv[1]), int(sys.argv[2])
    basic = pd.read_parquet(f"{META}/stock_basic.parquet")
    # 过滤 ST/退整 (与 signals 保持一致); 2023-09 前已退市且无窗口内数据的跳过
    basic = basic[~basic["name"].str.contains("ST|退", na=False)]
    basic = basic[(basic["status"] == "1") | (basic["out_date"] >= pd.Timestamp(START))]
    codes = sorted(basic["code"].tolist())
    mine = codes[wid::total]

    done = set()
    prog_path = f"{META}/bs_progress_{wid}.json"
    if os.path.exists(prog_path):
        done = set(json.load(open(prog_path)))
    # 兼容: 腾讯源已完成的也跳过
    tp = f"{META}/backfill_progress.json"
    tencent_done = set(json.load(open(tp))) if os.path.exists(tp) else set()
    tc2bs = {}
    for c in tencent_done:
        mkt, num = c.split(".")
        tc2bs[("sh." if mkt == "1" else "sz.") + num] = c
    todo = [c for c in mine if c not in done and c not in tc2bs]

    print(f"[worker-{wid}] 分到 {len(mine)}, 已完成 {len(done & set(mine))}, 腾讯已拉 {len(set(mine)&set(tc2bs))}, 待拉 {len(todo)}", flush=True)
    bs.login()
    shard_no = len([f for f in os.listdir(KDIR) if f.startswith(f"raw_b{wid}_")])
    buf_r, buf_q = [], []
    t_start, n_ok, n_fail = time.time(), 0, 0

    def flush(force=False):
        nonlocal shard_no
        if buf_r and (len(buf_r) >= 200 or force):
            pd.concat(buf_r).to_parquet(f"{KDIR}/raw_b{wid}_{shard_no:02d}.parquet", index=False)
            pd.concat(buf_q).to_parquet(f"{KDIR}/qfq_b{wid}_{shard_no:02d}.parquet", index=False)
            buf_r.clear()
            buf_q.clear()
            shard_no += 1

    for k, code in enumerate(todo):
        raw = query(bs, code, "3")
        qfq = query(bs, code, "2")
        if raw is None or qfq is None:
            n_fail += 1
        else:
            raw.insert(0, "code", code)
            qfq.insert(0, "code", code)
            buf_r.append(raw)
            buf_q.append(qfq)
            n_ok += 1
            done.add(code)
        flush()
        if (k + 1) % 100 == 0:
            json.dump(sorted(done), open(prog_path, "w"))
            el = time.time() - t_start
            eta = el / (k + 1) * (len(todo) - k - 1)
            print(f"[worker-{wid}] {k+1}/{len(todo)} ok={n_ok} fail={n_fail} "
                  f"平均 {(el/(k+1)):.2f}s/只 ETA {eta/60:.0f}min", flush=True)
        if (k + 1) % 1000 == 0:
            time.sleep(5)  # 防护性间歇
    flush(force=True)
    json.dump(sorted(done), open(prog_path, "w"))
    print(f"[worker-{wid}] 完成: ok={n_ok} fail={n_fail} 分片={shard_no}", flush=True)
    bs.logout()


if __name__ == "__main__":
    main()
