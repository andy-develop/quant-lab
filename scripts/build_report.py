#!/usr/bin/env python3
"""生成静态回测报告网页 report/index.html
数据: 引擎输出的权益曲线/交易/持仓 + 信号 + 涨停池汇总, 全部内嵌 JSON, 纯前端交互。
"""
import json
import glob
import os
import sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
META, POOL = f"{BASE}/data/meta", f"{BASE}/data/pool"
sys.path.insert(0, f"{BASE}/scripts")
import engine                                    # noqa: E402  (MAX_HOLD 权威来源)
from refresh_docs import compute_kpis, check_doc_consistency, DOC_SCORE_MODE  # noqa: E402

MAX_HOLD = engine.MAX_HOLD                       # 文案随引擎参数化, 杜绝 "满10日" 硬编码撒谎
STRAT_CN = {"momentum": "动量轮动"}
REASON_CN = {"stop_loss": "止损", "vol_shrink": "放量滞涨",
             "expired": "持有到期", "quick_fail": "快速认错", "market_exit": "逃顶清仓"}
REASON_DESC = {"stop_loss": "收盘价 ≤ 买入价×92% (硬止损)",
               "vol_shrink": "成交量≥2×5日均量且收阴",
               "expired": f"持有满 {MAX_HOLD} 个交易日",
               "quick_fail": "首日收盘浮亏超阈值",
               "market_exit": "大盘触发 Z0 空仓信号, 全线清仓"}
# 净值 payload: 单一全期 (回测起点起, 起点归一 100 万); 区间切换由前端切片+切片起点重归一, 见 main() 与 slice()


def _shared():
    """跨模式共享: 基准序列 / 信号 / 股票名"""
    bench = pd.read_parquet(f"{META}/bench_daily.parquet").set_index("date")["close"]
    # 中证1000 基准 (缺失时退化为沪深300)
    p1000 = f"{META}/csi1000_daily.parquet"
    if os.path.exists(p1000):
        _c = pd.read_parquet(p1000)
        _c["date"] = pd.to_datetime(_c["date"])
        bench1000 = _c.set_index("date")["close"]
    else:
        bench1000 = bench
    sigs = pd.read_parquet(f"{META}/signals.parquet")
    sigs["date"] = pd.to_datetime(sigs["date"])
    return bench, bench1000, sigs


def build_payload(meta_dir, mode, bench, bench1000, sigs, window_years: int | None = None):
    """window_years=None -> 回测全期; 数字 -> 近N年窗口。
    2026-09-09 起单一全期 payload: 区间切换全部由前端切片完成, 切片起点重归一 ¥100万
    (消除此前 y1/y3 双 payload 归一常数不同导致的基准对比细微差异与语义模糊)"""
    eq = pd.read_csv(f"{meta_dir}/equity.csv", parse_dates=["date"]).set_index("date")["equity"]
    bench = bench.reindex(eq.index).ffill()
    bench1000 = bench1000.reindex(eq.index).ffill()
    trades = pd.read_parquet(f"{meta_dir}/trades.parquet")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    hold = pd.read_parquet(f"{meta_dir}/holdings.parquet")
    hold["date"] = pd.to_datetime(hold["date"])
    hold["entry_date"] = pd.to_datetime(hold["entry_date"])

    last_day = eq.index[-1]
    if window_years is None:          # 全期: 不截断
        win_start = eq.index[0]
    else:
        # ---- 报告窗口: 去年同期(自然日)起, 如今天 2026-09-08 -> 2025-09-08 ----
        win_start = last_day - pd.DateOffset(years=window_years)
        # 回测起点 2023-09-01: 近3年窗口可能超出回测范围, 落到实际可用首日
        win_start = max(win_start, eq.index[0])
    if eq.index[0] < win_start:
        eq = eq[eq.index >= win_start]
        bench = bench.reindex(eq.index).ffill()
        bench1000 = bench1000.reindex(eq.index).ffill()   # 必须与 eq 同步截断, 否则前端窗口切片错位
    start_day = eq.index[0]
    trades = trades[trades["exit_date"] >= start_day].reset_index(drop=True)
    # ---- 起点归一: 窗口首日净值 = ¥100万 (两种模式口径一致, 基准同基准化) ----
    raw_last_eq = float(eq.iloc[-1])          # 归一前留存, 用于真实仓位比例
    eq = eq / eq.iloc[0] * 1_000_000.0

    # ---- 待买清单: 最后交易日的信号 ----
    last_sigs = sigs[sigs["date"] == last_day].sort_values("score", ascending=False).reset_index(drop=True)
    last_sigs["rank"] = last_sigs.index + 1          # 当日候选排名 (评分第1名起)
    pending = last_sigs.head(3).to_dict("records")   # 每日最多买 3 只
    for x in pending:
        x["code"] = x["code"][-6:]
        x["strategy_cn"] = STRAT_CN[x["strategy"]]
        x["date"] = str(pd.Timestamp(x["date"]).date())

    # ---- 下个交易日卖出计划: 收盘已挂单的持仓 ----
    sell_plan = []
    hold_last = hold[hold["date"] == last_day].set_index("code")
    ps_file = f"{meta_dir}/pending_sells.parquet"
    ps = pd.DataFrame(columns=["code", "reason"])   # 显式初始化: 文件缺失时 risk 段不依赖短路逻辑保命
    if os.path.exists(ps_file):
        ps = pd.read_parquet(ps_file)
        for _, r in ps.iterrows():
            if r["code"] not in hold_last.index:
                continue
            h = hold_last.loc[r["code"]]
            if isinstance(h, pd.DataFrame):     # 同code同日理论唯一, 防御
                h = h.iloc[0]
            sell_plan.append({
                "code": r["code"][-6:], "name": h["name"], "strategy_cn": STRAT_CN.get(h["strategy"], h["strategy"]),
                "entry_date": str(pd.Timestamp(h["entry_date"]).date()),
                "hold_days": int(h["hold_days"]), "pnl_pct": float(h["pnl_pct"]),
                "reason": REASON_CN.get(r["reason"], r["reason"]),
                "condition": REASON_DESC.get(r["reason"], r["reason"])})

    # ---- 卖出风险预警: 未挂单但距离触发条件很近的持仓 ----
    risk_list = []
    ps_codes_all = set(ps["code"]) if "code" in ps.columns else set()
    pending_codes = {c for c in ps_codes_all if c in hold_last.index}
    try:
        codes = [c for c in hold_last.index if c not in pending_codes]
        if codes:
            fl = pd.read_parquet(f"{META}/flags_long.parquet", filters=[("code", "in", codes)],
                                 columns=["code", "date", "close"])
            fl["date"] = pd.to_datetime(fl["date"])
            fl = fl.sort_values(["code", "date"])
            fl["ma20"] = fl.groupby("code")["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
            last_row = fl.groupby("code").tail(1).set_index("code")
            for code in codes:
                h = hold_last.loc[code]
                if isinstance(h, pd.DataFrame):
                    h = h.iloc[0]
                risks = []
                if code in last_row.index:
                    cl = float(last_row.at[code, "close"])
                    # cl/entry 由浮盈亏反推 (cost 含佣金+滑点系数≈1.0011)
                    ratio = (1 + float(h["pnl_pct"])) * 1.0011
                    if ratio > 0:
                        drop = 1 - 0.92 / ratio          # 再跌多少触发 -8% 硬止损
                        if 0 <= drop <= 0.02:
                            risks.append(f"再跌 {drop:.1%} 触发硬止损(-8%)")
                if h["hold_days"] == MAX_HOLD - 1:
                    risks.append(f"下个交易日满 {MAX_HOLD} 日, 强制持有到期退出")
                elif h["hold_days"] == MAX_HOLD - 2:
                    risks.append(f"2 个交易日后满 {MAX_HOLD} 日强制退出")
                if risks:
                    risk_list.append({
                        "code": code[-6:], "name": h["name"],
                        "entry_date": str(pd.Timestamp(h["entry_date"]).date()),
                        "hold_days": int(h["hold_days"]), "pnl_pct": float(h["pnl_pct"]),
                        "risks": risks})
    except Exception as e:
        print(f"[WARN] 卖出风险预警计算失败: {e}", flush=True)

    # ---- 持仓合计 ----
    hold_list = [
        {"code": r["code"][-6:], "name": r["name"], "strategy_cn": STRAT_CN[r["strategy"]],
         "entry_date": str(r["entry_date"].date()), "entry_px": float(r["entry_px"]),
         "shares": int(r["shares"]), "value": float(r["value"]),
         "pnl_pct": float(r["pnl_pct"]), "hold_days": int(r["hold_days"])}
        for _, r in hold[hold["date"] == last_day].sort_values("strategy").iterrows()
    ]
    hold_value = sum(x["value"] for x in hold_list)
    hold_ratio = hold_value / raw_last_eq if raw_last_eq > 0 else 0.0

    # ---- JSON 载荷 ----
    return {
        "mode": mode,
        "win_label": "回测全期" if window_years is None else f"近{window_years}年",
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "last_day": str(last_day.date()),
        "start_day": str(start_day.date()),
        "dates": [str(d.date()) for d in eq.index],
        "equity": [round(float(v), 2) for v in eq.values],
        "bench": [round(float(v / bench.iloc[0] * eq.iloc[0]), 2) for v in bench.values],
        "bench1000": [round(float(v / bench1000.iloc[0] * eq.iloc[0]), 2) for v in bench1000.values],
        "start_cap": 1000000,
        "trades": [
            {"code": r["code"][-6:], "name": r["name"], "strategy": r["strategy"],
             "strategy_cn": STRAT_CN[r["strategy"]],
             "entry_date": str(r["entry_date"].date()), "exit_date": str(r["exit_date"].date()),
             "entry_px": float(r["entry_px"]), "exit_px": float(r["exit_px"]),
             "shares": int(r["shares"]),
             "pnl_pct": float(r["pnl_pct"]), "pnl_cny": float(r["pnl_cny"]),
             "reason": REASON_CN.get(r["reason"], r["reason"]), "hold_days": int(r["hold_days"]),
             "buy_rank": int(r["buy_rank"]) if "buy_rank" in r else 0}
            for _, r in trades.sort_values("exit_date", ascending=False).iterrows()
        ],
        "holdings": hold_list,
        "hold_value": round(hold_value, 0),
        "hold_ratio": round(hold_ratio, 4),
        "pending": pending,
        "sell_plan": sell_plan,
        "risk_list": risk_list,
    }


def main():
    # ---- 口径一致性门禁: 评分模式/持有期与文档声明不符时直接失败 (CI fatal) ----
    check_doc_consistency()
    bench, bench1000, sigs = _shared()
    off_dir = f"{BASE}/data/meta_no"
    has_off = os.path.exists(f"{off_dir}/equity.csv")
    # 单一全期 payload × 双仓位口径: MODES["y3"][on/off]。
    # 区间切换(近一周~近三年)全部由前端从全期切片, 切片起点重归一 ¥100万 —— 归一语义唯一
    modes = {"y3": {}}
    p_on = build_payload(META, "on", bench, bench1000, sigs, window_years=None)
    if has_off:
        p_off = build_payload(off_dir, "off", bench, bench1000, sigs, window_years=None)
    else:
        p_off = dict(p_on, mode="off")   # 兜底: 无关模式数据时复用开模式
    modes["y3"] = {"on": p_on, "off": p_off}

    html = render_html(modes)
    os.makedirs(f"{BASE}/report", exist_ok=True)
    out = f"{BASE}/report/index.html"
    with open(out, "w") as f:
        f.write(html)
    n_on = len(p_on["trades"])
    n_off = len(p_off["trades"])
    print(f"报告已生成: {out}  ({len(html)/1024:.0f} KB, 全期 开:{n_on}笔/关:{n_off}笔; "
          f"区间切换由前端切片)", flush=True)
    return modes


def _kpi_block_html() -> str:
    """报告内的全期 A/B 数据块 (数字来自 refresh_docs.compute_kpis, 自动生成)。"""
    k = compute_kpis()

    def row(label, m):
        return (f"<tr><td>{label}</td><td class=\"{'up' if m['ret']>=0 else 'down'}\"><b>{m['ret']:+.1%}</b></td>"
                f"<td>{m['maxdd']:.1%}</td><td>{m['sharpe']:.2f}</td>"
                f"<td>{m['trades']}</td><td>{m['winrate']:.1%}</td></tr>")

    return (f"<table><tr><th>口径</th><th>收益</th><th>最大回撤</th><th>夏普</th><th>交易</th><th>胜率</th></tr>"
            f"{row('开 (默认)', k['on'])}{row('关', k['off'])}</table>"
            f"<p class=\"note\">数据区间 {k['on']['start']} ~ {k['on']['end']} · 由 refresh_docs.py 自动生成 · "
            f"评分模式 {DOC_SCORE_MODE} · 夏普口径 √244</p>")


def _st_banner_html() -> str:
    """ST 数据新鲜度横幅: 落后 >7 天在报告顶部醒目展示 (防 baostock 静默失败链无人察觉)。
    注: signals.run_scan 在 lag>20 时已 fatal, 此横幅覆盖 8~20 天的中间地带 + 报告存档可读性。"""
    f = f"{BASE}/data/meta/st_history.parquet"
    try:
        if not os.path.exists(f):
            return ('<div style="background:#FDE8E8;border:1px solid #C0392B;border-radius:10px;'
                    'padding:10px 14px;font-size:13px;color:#7B241C;margin-bottom:14px">'
                    '<b>⚠ ST 数据文件缺失</b> —— 新戴帽个股可能未被过滤, 回测结果不可信。</div>')
        st_last = pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).max()
        lag = (pd.Timestamp.today().normalize() - st_last).days
        if lag > 7:
            return (f'<div style="background:#FDE8E8;border:1px solid #C0392B;border-radius:10px;'
                    f'padding:10px 14px;font-size:13px;color:#7B241C;margin-bottom:14px">'
                    f'<b>⚠ ST 数据截至 {st_last:%Y-%m-%d} (已落后 {lag} 天)</b> —— '
                    f'期间新戴帽个股仍按正常股参与回测, 结果可能失真; '
                    f'请先运行 backfill_st.py incw/incmerge 刷新。</div>')
    except Exception as e:
        print(f"[WARN] ST 横幅生成失败: {e}", flush=True)
    return ""


def render_html(P):
    data_json = json.dumps(P, ensure_ascii=False)
    doc = (STRAT_DOC
           .replace("__MAX_HOLD__", str(MAX_HOLD))
           .replace("__KPI_BLOCK__", _kpi_block_html()))
    return (HTML_TEMPLATE
            .replace("__DATA__", data_json)
            .replace("__STRAT_DOC__", doc)
            .replace("__MAX_HOLD__", str(MAX_HOLD))
            .replace("__ST_BANNER__", _st_banner_html()))


STRAT_DOC = """
<style>
.doc h3{font-size:13.5px;font-weight:600;margin:14px 0 6px;color:#26251F;}
.doc h3:first-child{margin-top:0;}
.doc p,.doc li{font-size:12.5px;color:#44423B;line-height:1.75;}
.doc ul{padding-left:18px;margin:4px 0 8px;}
.doc code{background:#F0EFE9;border-radius:4px;padding:1px 5px;font-size:11.5px;font-family:ui-monospace,Menlo,monospace;}
.doc table{margin:6px 0 10px;}
.doc th,.doc td{font-size:12px;}
</style>
<div class="doc">

<h3>一、这个策略在做什么（一句话）</h3>
<p>每个交易日收盘后，从全市场五千多只股票里，筛出<b>「最近涨得最猛、趋势最健康」</b>的少数几只，按综合评分买最好的 3 只，<b>第二天开盘价买入</b>；每只最多拿 __MAX_HOLD__ 个交易日，赚的是强势股「惯性还在」这段钱——就像接力赛，只在趋势还在加速的路段接棒，不对抗趋势、不抄底、不追妖股。</p>

<h3>二、买什么？（四个门槛 + 一场排队）</h3>
<p>一只股票必须<b>同时</b>过下面四道门槛，才有资格进入买入候选：</p>
<ul>
<li><b>① 涨得最猛</b>：最近 20 天的涨幅排进全市场前 5%——只做最强的一批；</li>
<li><b>② 趋势健康</b>：股价站在 20 日线和 60 日线之上，且两条均线都朝上——涨得有序，不是乱拉；</li>
<li><b>③ 不追妖股</b>：最近 20 天涨幅不超过 60%，且过去 120 天整体是涨的——排除连续暴炒的鱼尾行情；</li>
<li><b>④ 买得进卖得掉</b>：日均成交额至少 3000 万元，且<b>绝不买 ST / 退市风险股</b>。</li>
</ul>
<p>同一天通过的股票往往不止 3 只，谁先买？按一张<b>「综合体检表」</b>打分排队，六项指标加权：</p>
<ul>
<li>涨势强不强（30%）＋ 涨得稳不稳（25%）＋ 过去 60 天上涨天数占比（20%）＋ 多头排列持续多久（15%）＋ 波动小不小（5%）＋ 短线回调是否到位（5%）。</li>
</ul>
<p>分数高的先买，买满当天 3 个名额为止。<b>这只影响排队顺序，不影响选股条件</b>；权重来自一轮轮对比实测（不同权重各跑一遍三年回测，选四项指标都占优的那组），当前这组在收益、回撤、夏普、胜率上全面领先。</p>

<h3>三、买多少？（大盘好才重仓，默认开启）</h3>
<p>页面的默认口径是<b>跟着大盘控制总仓位</b>，右上角按钮可切换：</p>
<ul>
<li><b>开（默认）</b>：盯上证指数给大盘「量体温」——均线空头或单日急跌 → <b>清仓避险</b>；多头排列 → <b>满仓</b>；中间状态轻仓 30% / 半仓 50% 过渡。大盘危险时第一优先级是先跑，而不是扛；</li>
<li><b>关</b>：不管大盘，始终满仓，最多同时持有 10 只、每只预算约总资金的 1/10。</li>
</ul>
<p><b>为什么默认开</b>：三年回测对比，开仓位控制在<b>收益、回撤、夏普、胜率四个指标上全部占优</b>——收益约为关的 2.2 倍，最大回撤只有关的约 1/3。机制也很直白：清仓空出来的名额，能让位给评分更高的新股票，「换仓周转」本身就在赚钱。下方数据块即为最新对比：</p>
__KPI_BLOCK__

<h3>四、什么时候卖？（四条规则，优先级从高到低）</h3>
<ul>
<li><b>① 大盘亮红灯</b>：大盘触发清仓条件 → 全部持仓无条件卖掉，最高优先级；</li>
<li><b>② 止损</b>：单只浮亏到 <b>-8%</b> → 第二天开盘卖出，认错要快；</li>
<li><b>③ 到期</b>：拿满 __MAX_HOLD__ 个交易日 → 强制换仓，把名额让给更强的信号；</li>
<li><b>④ 主力出货嫌疑</b>：持有至少 2 天后，成交量突然放大到前一日的 2 倍以上、当天却收阴线 → 像是有人在高位出货，卖出。</li>
</ul>
<p>曾经还有一条「跌破 20 日线就卖」，对比实测发现它<b>专门把正在赚钱的强势股提前砍掉</b>（74 笔合计亏 17.7 万，胜率仅 16%），已删除——这正说明了每条规则都要用数据验证，而不是凭感觉加。</p>

<h3>五、怎么买怎么算成本？（执行细节）</h3>
<ul>
<li><b>时序</b>：今天收盘后做决定，<b>明天开盘价</b>成交，先卖后买——决策用的全是当天已收盘的数据，不存在「偷看未来」；</li>
<li><b>A 股规则内建</b>：今天买的明天才能卖（T+1）；开盘就涨停的<b>放弃追入</b>，开盘跌停卖不掉的顺延到下一个能卖的开盘；停牌顺延；</li>
<li><b>整手买入</b>：股数按 100 股整数倍，单只预算约总资金 1/10；</li>
<li><b>成本从宽计提</b>：佣金万分之一（单笔最低 5 元）＋ 卖出印花税 0.05% ＋ 滑点 0.1%——回测里的收益是扣完这些之后的。</li>
</ul>

<h3>六、看这份报告要注意什么</h3>
<ul>
<li><b>数据底子</b>：回测从 2023 年 9 月起，股票池<b>包含后来退市的股票</b>（不用「只活下来」的股票自欺欺人），价格已消除分红送股的影响、任意日期重跑结果一致；</li>
<li><b>诚实的局限</b>：按开盘价全额成交（实盘大资金未必）、滑点按固定 0.1%（极端行情可能更高）、开盘涨停的机会直接放弃（实盘可能忍不住追高）、<b>历史表现不代表未来</b>。</li>
</ul>
<p style="color:#88867E">仅供研究学习，不构成投资建议。</p>
</div>
"""


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>A股短线策略实验室 · 回测报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
<style>
:root{--bg:#F6F6F4;--card:#FFFFFF;--ink:#26251F;--muted:#88867E;--line:#E4E3DC;
--up:#D5423E;--down:#1D9E75;--blue:#185FA5;--amber:#854F0B;--purple:#534AB7;}
*{margin:0;padding:0;box-sizing:border-box;}
body{background:var(--bg);color:var(--ink);font:14px/1.6 -apple-system,"PingFang SC","Helvetica Neue",sans-serif;padding:28px 32px;}
.wrap{max-width:1180px;margin:0 auto;}
h1{font-size:22px;font-weight:600;letter-spacing:.5px;}
.sub{color:var(--muted);font-size:12.5px;margin-top:4px;}
.tabs{display:flex;gap:8px;margin:20px 0 14px;flex-wrap:wrap;}
.tab{padding:6px 16px;border-radius:16px;background:var(--card);border:1px solid var(--line);cursor:pointer;font-size:13px;color:var(--muted);}
.tab.on{background:var(--ink);color:#fff;border-color:var(--ink);}
.kpis{margin-bottom:14px;}
.up{color:var(--up);} .down{color:var(--down);}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;margin-bottom:14px;}
.card h2{font-size:15px;font-weight:600;margin-bottom:10px;}
.card .note{font-size:12px;color:var(--muted);}
.chart{width:100%;height:340px;}
.chart2{width:100%;height:300px;}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
th{text-align:left;color:var(--muted);font-weight:500;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap;}
td{padding:6px 8px;border-bottom:1px solid #F0EFE9;font-variant-numeric:tabular-nums;white-space:nowrap;}
tr:hover td{background:#FAFAF7;}
.grid2{display:grid;grid-template-columns:1.2fr 1fr;gap:14px;}
.sub-h{font-size:13px;font-weight:600;margin:16px 0 6px;color:#26251F;border-left:3px solid #534AB7;padding-left:8px;}
.sub-h:first-of-type{margin-top:4px;}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;}
.fold-btn{float:right;background:var(--card);border:1px solid var(--line);border-radius:16px;padding:3px 14px;font-size:12px;color:var(--muted);cursor:pointer;font-family:inherit;}
.fold-btn:hover{color:var(--ink);border-color:var(--muted);}
.hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;}
.mode-sw{display:flex;border:1px solid var(--line);border-radius:16px;overflow:hidden;flex-shrink:0;margin-top:2px;}
.mode-btn{padding:6px 14px;font-size:12.5px;color:var(--muted);cursor:pointer;background:var(--card);border:0;font-family:inherit;white-space:nowrap;}
.mode-btn.on{background:var(--ink);color:#fff;}
.mode-btn:not(.on):hover{color:var(--ink);}
.mode-note{font-size:11px;color:var(--muted);margin-top:3px;text-align:right;}
.tg-a{background:#E6F1FB;color:#185FA5;} .tg-b{background:#E1F5EE;color:#0F6E56;} .tg-c{background:#EEEDFE;color:#534AB7;}
.warn{background:#FAEEDA;border:1px solid #EF9F27;border-radius:10px;padding:10px 14px;font-size:12.5px;color:#633806;margin-bottom:14px;}
footer{color:var(--muted);font-size:11.5px;margin-top:20px;line-height:1.8;}
</style>
</head>
<body>
<div class="wrap">
<div class="hdr">
  <div>
    <h1>A股短线策略实验室</h1>
    <div class="sub" id="sub"></div>
  </div>
  <div>
    <div class="mode-sw" id="modeSw">
      <button class="mode-btn on" data-m="on" onclick="setMode('on')">仓位控制 开</button>
      <button class="mode-btn" data-m="off" onclick="setMode('off')">关 (满仓无择时)</button>
    </div>
    <div class="mode-note">大盘状态机 Z0–Z3 · 锚定上证</div>
  </div>
</div>
__ST_BANNER__<div class="warn" style="margin-top:14px" id="warnBar"></div>

<div class="tabs" id="rangeTabs">
  <div class="tab" data-n="5">近一周</div>
  <div class="tab" data-n="21">近一月</div>
  <div class="tab" data-n="63">近三月</div>
  <div class="tab" data-n="122">近六月</div>
  <div class="tab on" data-n="244">近一年</div>
  <div class="tab" data-n="0">近三年</div>
</div>
<div class="kpis card"><h2>核心指标对比 <span class="note">(所选区间 · 沪深300为同期买入持有)</span></h2><table id="kpiTable"></table></div>

<div class="card"><h2>净值曲线 <span class="note" id="eqNote"></span></h2><div class="chart" id="eqChart"></div></div>

<div class="grid2">
<div class="card"><h2>卖出原因分布 <span class="note">(所选区间)</span></h2><div class="chart2" id="reasonChart"></div></div>
<div class="card"><h2>买入原因分布 <span class="note">(所选区间 · 按当日候选评分排名)</span></h2><table id="buyReasonTable"></table></div>
</div>

<div class="card"><h2>下一交易日计划 <span class="note" id="planNote"></span></h2>
<div class="sub-h">① 确定卖出（收盘已挂卖单, 下一开盘执行）</div>
<table id="planSell"></table>
<div class="sub-h">② 有可能卖出（距卖出条件很近, 仅供预警）</div>
<table id="planRisk"></table>
<div class="sub-h">③ 有可能买入（T日收盘信号, 按评分取前3, 每日最多买3只）</div>
<table id="planBuy"></table>
</div>

<div class="card"><h2>当前持仓 <span class="note" id="holdNote"></span></h2><table id="holdTable"></table></div>

<div class="card"><h2>交易明细 <span class="note" id="tradeNote"></span><button class="fold-btn" id="tradeToggle" onclick="toggleTrade()">展开明细 ▾</button></h2><div id="tradeWrap" style="display:none"><table id="tradeTable"></table></div></div>

<div class="card" id="stratDoc"><h2>策略说明</h2>
__STRAT_DOC__
</div>

<footer id="foot"></footer>
</div>
<script>
const MODES = __DATA__;
let curWin = 'y3';          // 唯一窗口: 回测全期 payload (区间切换由 rangeN 前端切片)
let D = MODES[curWin].on;  // 默认: 近一年切片 · 开启仓位控制
let curMode = 'on';
const fmtPct = x => (x>=0?'+':'') + (x*100).toFixed(2) + '%';
const cls = x => x>=0 ? 'up' : 'down';
const nf = x => x.toLocaleString('zh-CN', {maximumFractionDigits:0});

// ---------- 仓位控制模式切换 ----------
const WARN_ON  = '本报告回测覆盖<b>动量轮动</b>策略, <b>仓位控制开启</b>: 总仓位锚定上证指数「买卖点」状态机 (Z0 空仓 0% 且触发全线清仓 / Z1 轻仓 30% / Z2 半仓 50% / Z3 重仓 100%, 阶梯预算), 关闭切换见右上角。所有结果含佣金万1、印花税与滑点, 涨跌停与 T+1 规则已内建。';
const WARN_OFF = '本报告回测覆盖<b>动量轮动</b>策略, <b>仓位控制关闭</b>: 始终满仓运行 (最多同时持有 10 只, 每日最多新开仓 3 只), 无大盘择时。注意: 该口径在 2024 年初微盘踩踏中无任何系统性保护。所有结果含佣金万1、印花税与滑点, 涨跌停与 T+1 规则已内建。';
const FOOT_ON  = `回测口径: T日收盘出信号, T+1开盘成交; 开盘涨停放弃买入, 开盘跌停顺延卖出; 止损-8% / 放量滞涨 / 持有满__MAX_HOLD__日退出; 上证指数触发「买卖点」空仓条件(Z0)时全线清仓。<br>
   仓位控制(开): 每日最多新开仓 3 只; 总仓位锚定上证指数状态机 — Z0 空仓0% / Z1 轻仓30% / Z2 半仓50% / Z3 重仓100%(均线多头排列)。<br>
   买入股数: 按整手(100股整数倍, 最低1手)向下取整, 单只预算≈总资金/10 且不超状态机目标仓位。<br>
   策略贡献盈亏为交易盈亏加总(不含空仓资金占用), 胜率为区间内平仓交易口径。区间切换(近一周~近三年)=回测全期数据的前端切片, 切片起点重归一 ¥100万(策略与基准同一起点)。数据源: 腾讯行情 · 东财涨停池 · baostock 股票池(含退市)。仅供研究, 不构成投资建议。`;
const FOOT_OFF = `回测口径: T日收盘出信号, T+1开盘成交; 开盘涨停放弃买入, 开盘跌停顺延卖出; 止损-8% / 放量滞涨 / 持有满__MAX_HOLD__日退出; 无大盘择时, 始终满仓。<br>
   仓位控制(关): 每日最多新开仓 3 只, 最多同时持有 10 只, 单只预算≈总资金/10。<br>
   买入股数: 按整手(100股整数倍, 最低1手)向下取整。<br>
   策略贡献盈亏为交易盈亏加总, 胜率为区间内平仓交易口径。区间切换(近一周~近三年)=回测全期数据的前端切片, 切片起点重归一 ¥100万(策略与基准同一起点)。数据源: 腾讯行情 · 东财涨停池 · baostock 股票池(含退市)。仅供研究, 不构成投资建议。`;
function setMode(m){
  if(!MODES[curWin][m] || m===curMode) return;
  curMode = m; D = MODES[curWin][m];
  document.querySelectorAll('#modeSw .mode-btn').forEach(b=>b.classList.toggle('on', b.dataset.m===m));
  document.getElementById('warnBar').innerHTML = m==='on' ? WARN_ON : WARN_OFF;
  document.getElementById('foot').innerHTML = m==='on' ? FOOT_ON : FOOT_OFF;
  renderAll();
}

// ---------- 交易明细折叠 ----------
let tradeOpen = false;
function toggleTrade(){
  tradeOpen = !tradeOpen;
  document.getElementById('tradeWrap').style.display = tradeOpen ? '' : 'none';
  document.getElementById('tradeToggle').textContent = tradeOpen ? '收起明细 ▴' : '展开明细 ▾';
}

// ---------- 区间切换: 统一从全期 payload 切"最后 N+1 个交易日", 切片起点重归一 ¥100万 ----------
// (策略与三条基准同一切片起点归一, 归一语义唯一; 近三年=全期, 切片起点即 payload 起点为 no-op)
let rangeN = 244;   // 默认近一年
document.getElementById('rangeTabs').addEventListener('click', e => {
  const t = e.target;
  if(t.dataset.n === undefined) return;
  document.querySelectorAll('#rangeTabs .tab').forEach(x=>x.classList.remove('on'));
  t.classList.add('on');
  rangeN = +t.dataset.n;
  D = MODES[curWin][curMode];
  renderAll();
});

function slice(){
  const n = rangeN === 0 ? D.dates.length : rangeN + 1;
  const s = Math.max(0, D.dates.length - n);
  const renorm = a => { const x = a.slice(s); const k = x[0] ? 1e6/x[0] : 1;
                        return x.map(v => v*k); };   // k≈1 时多 map 一次开销可忽略, 不做浮点短路
  return {dates: D.dates.slice(s), equity: renorm(D.equity), bench: renorm(D.bench),
          bench1000: renorm(D.bench1000 || D.bench), start: D.dates[s]};
}

function kpis(){
  const sl = slice();
  const eq = sl.equity, n = eq.length - 1;
  const total = eq[eq.length-1]/eq[0]-1;
  const ann = n>0 ? Math.pow(1+total, 244/n)-1 : 0;
  const daily = []; for(let i=1;i<eq.length;i++) daily.push(eq[i]/eq[i-1]-1);
  const mean = daily.reduce((a,b)=>a+b,0)/Math.max(daily.length,1);
  const sd = Math.sqrt(daily.reduce((a,b)=>a+(b-mean)**2,0)/Math.max(daily.length-1,1));
  const sharpe = sd>0 ? mean/sd*Math.sqrt(244) : 0;
  let peak=eq[0], mdd=0; for(const v of eq){peak=Math.max(peak,v); mdd=Math.min(mdd, v/peak-1);}
  const startTs = sl.start;
  const tr = D.trades.filter(t => t.exit_date >= startTs);
  const wins = tr.filter(t=>t.pnl_pct>0);
  const win = tr.length ? wins.length/tr.length : 0;
  const gp = wins.reduce((a,t)=>a+t.pnl_cny,0);
  const gl = -tr.filter(t=>t.pnl_pct<=0).reduce((a,t)=>a+t.pnl_cny,0);
  const pf = gl>0 ? gp/gl : (gp>0?999:0);
  // 沪深300长持 (同期买入持有)
  const b = sl.bench, bn = b.length - 1;
  const bTotal = b[b.length-1]/b[0]-1;
  const bAnn = bn>0 ? Math.pow(1+bTotal, 244/bn)-1 : 0;
  const bDaily = []; for(let i=1;i<b.length;i++) bDaily.push(b[i]/b[i-1]-1);
  const bMean = bDaily.reduce((x,y)=>x+y,0)/Math.max(bDaily.length,1);
  const bSd = Math.sqrt(bDaily.reduce((x,y)=>x+(y-bMean)**2,0)/Math.max(bDaily.length-1,1));
  const bSharpe = bSd>0 ? bMean/bSd*Math.sqrt(244) : 0;
  let bPeak=b[0], bMdd=0; for(const v of b){bPeak=Math.max(bPeak,v); bMdd=Math.min(bMdd, v/bPeak-1);}
  // 中证1000长持 (同期买入持有)
  const c = sl.bench1000, cn = c.length - 1;
  const cTotal = c[c.length-1]/c[0]-1;
  const cDaily = []; for(let i=1;i<c.length;i++) cDaily.push(c[i]/c[i-1]-1);
  const cMean = cDaily.reduce((x,y)=>x+y,0)/Math.max(cDaily.length,1);
  const cSd = Math.sqrt(cDaily.reduce((x,y)=>x+(y-cMean)**2,0)/Math.max(cDaily.length-1,1));
  const cSharpe = cSd>0 ? cMean/cSd*Math.sqrt(244) : 0;
  let cPeak=c[0], cMdd=0; for(const v of c){cPeak=Math.max(cPeak,v); cMdd=Math.min(cMdd, v/cPeak-1);}
  document.getElementById('kpiTable').innerHTML =
    `<thead><tr><th>指标</th><th>本策略</th><th>沪深300长持</th><th>中证1000长持</th></tr></thead><tbody>`+
    `<tr><td>区间收益</td><td class="${cls(total)}"><b>${fmtPct(total)}</b></td><td class="${cls(bTotal)}">${fmtPct(bTotal)}</td><td class="${cls(cTotal)}">${fmtPct(cTotal)}</td></tr>`+
    `<tr><td>最大回撤</td><td class="down"><b>${fmtPct(mdd)}</b></td><td class="down">${fmtPct(bMdd)}</td><td class="down">${fmtPct(cMdd)}</td></tr>`+
    `<tr><td>夏普比率</td><td class="${sharpe>=1?'up':''}"><b>${sharpe.toFixed(2)}</b></td><td>${bSharpe.toFixed(2)}</td><td>${cSharpe.toFixed(2)}</td></tr>`+
    `<tr><td>胜率 <span class="note">${tr.length}笔 · 盈利因子${pf.toFixed(2)}<br>(Σ盈利/Σ亏损, 非平均盈亏比)</span></td><td class="${win>=0.5?'up':'down'}"><b>${(win*100).toFixed(1)}%</b></td><td class="note">—</td><td class="note">—</td></tr>`+
    `</tbody>`;
  return tr;
}

// ---------- 净值曲线 ----------
let eqChart, reasonChart;
function drawEquity(){
  const sl = slice();
  const series = [
    {name:'策略净值',type:'line',data:sl.equity,showSymbol:false,lineStyle:{width:1.6,color:'#185FA5'},areaStyle:{color:'rgba(24,95,165,0.06)'}},
    {name:'沪深300(等基准化)',type:'line',data:sl.bench,showSymbol:false,lineStyle:{width:1.2,color:'#B4B2A9',type:'dashed'}},
    {name:'中证1000(等基准化)',type:'line',data:sl.bench1000,showSymbol:false,lineStyle:{width:1.2,color:'#B07A2A',type:'dashed'}}
  ];
  eqChart.setOption({
    animation:false,
    tooltip:{trigger:'axis',valueFormatter:v=>nf(v)},
    legend:{data:series.map(s=>s.name),top:0,textStyle:{fontSize:11,color:'#88867E'}},
    grid:{left:56,right:24,top:30,bottom:26},
    xAxis:{type:'category',data:sl.dates,axisLabel:{fontSize:10,color:'#88867E'}},
    yAxis:{type:'value',scale:true,axisLabel:{fontSize:10,color:'#88867E',formatter:nf}},
    series}, true);
  document.getElementById('eqNote').textContent = `(${sl.start} ~ ${D.last_day}, 起点归一 ¥100万)`;
}

function drawReasons(tr){
  const rc = {}; tr.forEach(t=>rc[t.reason]=(rc[t.reason]||0)+1);
  reasonChart.setOption({animation:false,
    tooltip:{},
    grid:{left:80,right:30,top:10,bottom:26},
    xAxis:{type:'value',axisLabel:{fontSize:10,color:'#88867E'}},
    yAxis:{type:'category',data:Object.keys(rc),axisLabel:{fontSize:11}},
    series:[{type:'bar',data:Object.values(rc),barWidth:16,itemStyle:{color:'#534AB7'},
      label:{show:true,position:'right',fontSize:10,color:'#88867E'}}]}, true);
}

function tables(tr){
  // 交易明细
  const rows = tr.slice(0,400).map(t=>`<tr>
    <td>${t.exit_date}</td><td>${t.code}</td><td>${t.name}</td>
    <td><span class="tag tg-c">${t.strategy_cn}</span></td>
    <td>${t.entry_date}</td><td>${t.entry_px.toFixed(2)}</td><td>${t.shares.toLocaleString()}</td><td>${t.exit_px.toFixed(2)}</td>
    <td class="${cls(t.pnl_pct)}">${fmtPct(t.pnl_pct)}</td><td class="${cls(t.pnl_cny)}">${nf(t.pnl_cny)}</td>
    <td>${t.hold_days}天</td><td>${t.reason}</td></tr>`).join('');
  document.getElementById('tradeTable').innerHTML =
    `<thead><tr><th>卖出日</th><th>代码</th><th>名称</th><th>策略</th><th>买入日</th><th>买价</th><th>股数</th><th>卖价</th><th>收益率</th><th>盈亏(¥)</th><th>持有</th><th>原因</th></tr></thead><tbody>${rows||'<tr><td colspan=12 class="note">该区间无交易</td></tr>'}</tbody>`;
  document.getElementById('tradeNote').textContent = `(共 ${tr.length} 笔, 显示前 ${Math.min(tr.length,400)} 笔, 按卖出日排序 · 股数按整手交易, 1手=100股)`;
  // 当前持仓
  document.getElementById('holdTable').innerHTML =
    `<thead><tr><th>代码</th><th>名称</th><th>策略</th><th>买入日</th><th>买价</th><th>股数</th><th>市值(¥)</th><th>持有</th><th>浮动盈亏</th></tr></thead><tbody>`+
    (D.holdings.map(h=>`<tr><td>${h.code}</td><td>${h.name}</td><td>${h.strategy_cn}</td><td>${h.entry_date}</td><td>${h.entry_px.toFixed(2)}</td><td>${h.shares.toLocaleString()}</td><td>${nf(h.value)}</td><td>${h.hold_days}天</td><td class="${cls(h.pnl_pct)}">${fmtPct(h.pnl_pct)}</td></tr>`).join('')||'<tr><td colspan=9 class="note">空仓</td></tr>')+'</tbody>';
  const hr = D.hold_ratio || 0;
  document.getElementById('holdNote').textContent =
    `(${D.last_day} 收盘 · ${D.holdings.length}只 · 合计 ¥${nf(D.hold_value||0)} · 实际仓位 ${(hr*100).toFixed(0)}%)`;
  // 下一交易日计划
  document.getElementById('planNote').textContent = `(${D.last_day} 收盘判定 · 下一开盘执行)`;
  const sp = D.sell_plan || [];
  document.getElementById('planSell').innerHTML =
    `<thead><tr><th>代码</th><th>名称</th><th>买入日</th><th>持有</th><th>浮盈亏</th><th>卖出条件</th></tr></thead><tbody>`+
    (sp.map(s=>`<tr><td>${s.code}</td><td>${s.name}</td><td>${s.entry_date}</td><td>${s.hold_days}天</td><td class="${cls(s.pnl_pct)}">${fmtPct(s.pnl_pct)}</td><td><b>${s.reason}</b> · ${s.condition}</td></tr>`).join('')
     ||'<tr><td colspan=6 class="note">收盘时无待卖持仓, 下个交易日不卖出</td></tr>')+'</tbody>';
  const rl = D.risk_list || [];
  document.getElementById('planRisk').innerHTML =
    `<thead><tr><th>代码</th><th>名称</th><th>买入日</th><th>持有</th><th>浮盈亏</th><th>风险提示</th></tr></thead><tbody>`+
    (rl.map(r=>`<tr><td>${r.code}</td><td>${r.name}</td><td>${r.entry_date}</td><td>${r.hold_days}天</td><td class="${cls(r.pnl_pct)}">${fmtPct(r.pnl_pct)}</td><td>${r.risks.map(x=>'<span style="color:#633806">⚠</span> '+x).join('<br>')}</td></tr>`).join('')
     ||'<tr><td colspan=6 class="note">当前无接近卖出条件的持仓</td></tr>')+'</tbody>';
  const perStock = curMode==='on' ? '单只 ≤ 10% 总资金, 且不超状态机目标仓位剩余预算' : '单只 ≤ 10% 总资金 (整手买入, 最低1手)';
  const buyCond = curMode==='on'
    ? `信号已触发; 次日开盘价不为一字涨停即可买入, 开盘涨停放弃; 总仓位不超过状态机目标 (Z0 0%/Z1 30%/Z2 50%/Z3 100%)`
    : `信号已触发; 次日开盘价不为一字涨停即可买入, 开盘涨停放弃; 最多同时持有 10 只`;
  document.getElementById('planBuy').innerHTML =
    `<thead><tr><th>排名</th><th>代码</th><th>名称</th><th>评分</th><th>建议仓位</th><th>买入条件</th></tr></thead><tbody>`+
    (D.pending.map(p=>`<tr><td>第${p.rank}名</td><td>${p.code}</td><td>${p.name}</td><td>${p.score.toFixed(3)}</td><td>${perStock}</td><td>${buyCond}</td></tr>`).join('')
     ||'<tr><td colspan=6 class="note">最后交易日无信号, 下个交易日无新开仓计划</td></tr>')+'</tbody>';
}

function drawBuyReasons(tr){
  const lab = r => r===1 ? '当日候选第1名 (评分最高)' : r===2 ? '当日候选第2名' : r===3 ? '当日候选第3名' : '第4名及以后 (前列被涨停/停牌跳过后买入)';
  const order = ['当日候选第1名 (评分最高)','当日候选第2名','当日候选第3名','第4名及以后 (前列被涨停/停牌跳过后买入)'];
  const groups = {};
  tr.forEach(t=>{ const k = lab(t.buy_rank||0); (groups[k]=groups[k]||[]).push(t); });
  const rows = order.filter(k=>groups[k]).map(k=>{
    const g = groups[k], n = g.length;
    const w = g.filter(x=>x.pnl_pct>0).length/n;
    const avg = g.reduce((a,x)=>a+x.pnl_pct,0)/n;
    const pnl = g.reduce((a,x)=>a+x.pnl_cny,0);
    return `<tr><td>${k}</td><td>${n}</td><td class="${w>=0.5?'up':'down'}">${(w*100).toFixed(1)}%</td><td class="${cls(avg)}">${fmtPct(avg)}</td><td class="${cls(pnl)}">${nf(pnl)}</td></tr>`;
  }).join('');
  document.getElementById('buyReasonTable').innerHTML =
    `<thead><tr><th>买入原因</th><th>笔数</th><th>胜率</th><th>平均收益</th><th>盈亏合计(¥)</th></tr></thead><tbody>${rows||'<tr><td colspan=5 class="note">该区间无交易</td></tr>'}</tbody>`;
}

function renderAll(){
  const tr = kpis();
  drawEquity(); drawReasons(tr); drawBuyReasons(tr); tables(tr);
}

document.getElementById('sub').textContent =
  `生成于 ${D.generated} · 区间切换=回测全期切片(切片起点重归一 ¥100万) · 每日最多买3只 · 佣金万1+印花税0.05%+滑点0.1%`;
document.getElementById('warnBar').innerHTML = WARN_ON;
document.getElementById('foot').innerHTML = FOOT_ON;

eqChart = echarts.init(document.getElementById('eqChart'));
reasonChart = echarts.init(document.getElementById('reasonChart'));
renderAll();
window.addEventListener('resize', ()=>{eqChart.resize();reasonChart.resize();});
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
