#!/usr/bin/env python3
"""生成静态回测报告网页 report/index.html
数据: 引擎输出的权益曲线/交易/持仓 + 信号 + 涨停池汇总, 全部内嵌 JSON, 纯前端交互。
"""
import json
import glob
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录(本地/CI通用)
META, POOL = f"{BASE}/data/meta", f"{BASE}/data/pool"
STRAT_CN = {"momentum": "动量轮动"}
REASON_CN = {"stop_loss": "止损", "trend_break": "趋势破位", "vol_shrink": "放量滞涨",
             "expired": "持有到期", "market_exit": "逃顶清仓", "quick_fail": "快速认错"}
REASON_DESC = {"stop_loss": "收盘价 ≤ 买入价×92% (硬止损)",
               "trend_break": "收盘跌破 MA20",
               "vol_shrink": "成交量≥2×5日均量且收阴",
               "expired": "持有满 10 个交易日",
               "market_exit": "大盘触发 Z0 空仓信号, 全线清仓",
               "quick_fail": "首日收盘浮亏超阈值"}
WINDOW_DAYS = 244        # 报告窗口: 近一年(交易日)


def main():
    eq = pd.read_csv(f"{META}/equity.csv", parse_dates=["date"]).set_index("date")["equity"]
    bench = pd.read_parquet(f"{META}/bench_daily.parquet").set_index("date")["close"]
    bench = bench.reindex(eq.index).ffill()
    # 中证1000 基准 (缺失时退化为沪深300)
    p1000 = f"{META}/csi1000_daily.parquet"
    if os.path.exists(p1000):
        _c = pd.read_parquet(p1000)
        _c["date"] = pd.to_datetime(_c["date"])
        bench1000 = _c.set_index("date")["close"].reindex(eq.index).ffill()
    else:
        bench1000 = bench
    trades = pd.read_parquet(f"{META}/trades.parquet")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    hold = pd.read_parquet(f"{META}/holdings.parquet")
    hold["date"] = pd.to_datetime(hold["date"])
    hold["entry_date"] = pd.to_datetime(hold["entry_date"])
    sigs = pd.read_parquet(f"{META}/signals.parquet")
    sigs["date"] = pd.to_datetime(sigs["date"])
    basic = pd.read_parquet(f"{META}/stock_basic.parquet").set_index("secid")["name"]

    last_day = eq.index[-1]
    # ---- 报告窗口: 近一年 ----
    if len(eq) > WINDOW_DAYS:
        eq = eq.iloc[-WINDOW_DAYS:]
        bench = bench.reindex(eq.index).ffill()
    start_day = eq.index[0]
    trades = trades[trades["exit_date"] >= start_day].reset_index(drop=True)

    # ---- 待买清单: 最后交易日的信号 ----
    last_sigs = sigs[sigs["date"] == last_day].sort_values("score", ascending=False).reset_index(drop=True)
    last_sigs["rank"] = last_sigs.index + 1          # 当日候选排名 (评分第1名起)
    pending = last_sigs.head(3).to_dict("records")   # 每日最多买 3 只
    for x in pending:
        x["code"] = x["code"][-6:]
        x["strategy_cn"] = STRAT_CN[x["strategy"]]
        x["date"] = str(pd.Timestamp(x["date"]).date())

    # ---- 大盘状态 (建议仓位用) ----
    reg_state, reg_ratio = "-", 0.0
    if os.path.exists(f"{META}/market_regime.parquet"):
        _reg = pd.read_parquet(f"{META}/market_regime.parquet")
        reg_state = str(_reg.iloc[-1]["state"])
        reg_ratio = float(_reg.iloc[-1]["target_ratio"])

    # ---- 下个交易日卖出计划: 收盘已挂单的持仓 ----
    sell_plan = []
    hold_last = hold[hold["date"] == last_day].set_index("code")
    ps_file = f"{META}/pending_sells.parquet"
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
    ps_codes_all = set(ps["code"]) if (os.path.exists(ps_file) and "code" in ps.columns) else set()
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
                    m20 = last_row.at[code, "ma20"]
                    # cl/entry_qfq 由浮盈亏反推 (cost 含佣金+滑点系数≈1.0011)
                    ratio = (1 + float(h["pnl_pct"])) * 1.0011
                    if ratio > 0:
                        drop = 1 - 0.92 / ratio          # 再跌多少触发 -8% 硬止损
                        if 0 <= drop <= 0.02:
                            risks.append(f"再跌 {drop:.1%} 触发硬止损(-8%)")
                    if pd.notna(m20) and m20 > 0:
                        gap = cl / m20 - 1               # 距 MA20 破位
                        if 0 <= gap <= 0.02:
                            tag = "跌破MA20即触发趋势破位" if h["hold_days"] >= 3 else "跌破MA20且持有满3日后触发趋势破位"
                            risks.append(f"收盘距MA20仅 {gap:.1%}, {tag}")
                if h["hold_days"] == 9:
                    risks.append("下个交易日满 10 日, 强制持有到期退出")
                elif h["hold_days"] == 8:
                    risks.append("2 个交易日后满 10 日强制退出")
                if risks:
                    risk_list.append({
                        "code": code[-6:], "name": h["name"],
                        "entry_date": str(pd.Timestamp(h["entry_date"]).date()),
                        "hold_days": int(h["hold_days"]), "pnl_pct": float(h["pnl_pct"]),
                        "risks": risks})
    except Exception as e:
        print(f"[WARN] 卖出风险预警计算失败: {e}", flush=True)

    # ---- 持仓合计 vs 目标仓位 ----
    hold_list = [
        {"code": r["code"][-6:], "name": r["name"], "strategy_cn": STRAT_CN[r["strategy"]],
         "entry_date": str(r["entry_date"].date()), "entry_px": float(r["entry_px"]),
         "shares": int(r["shares"]), "value": float(r["value"]),
         "pnl_pct": float(r["pnl_pct"]), "hold_days": int(r["hold_days"])}
        for _, r in hold[hold["date"] == last_day].sort_values("strategy").iterrows()
    ]
    hold_value = sum(x["value"] for x in hold_list)
    hold_ratio = hold_value / float(eq.iloc[-1]) if float(eq.iloc[-1]) > 0 else 0.0

    # ---- JSON 载荷 ----
    payload = {
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
        "reg_state": reg_state,
        "reg_ratio": reg_ratio,
    }

    html = render_html(payload)
    os.makedirs(f"{BASE}/report", exist_ok=True)
    out = f"{BASE}/report/index.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"报告已生成: {out}  ({len(html)/1024:.0f} KB)", flush=True)
    return payload


def render_html(P):
    data_json = json.dumps(P, ensure_ascii=False)
    return (HTML_TEMPLATE
            .replace("__DATA__", data_json)
            .replace("__STRAT_DOC__", STRAT_DOC))


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

<h3>一、策略总览</h3>
<p>本系统为<b>纯日线量化选股系统</b>，运行于 A 股市场，目标持仓 5–10 个交易日，每日最多新开仓 3 只，总仓位由大盘状态机控制。策略层与执行层完全分离：策略只在 T 日收盘后基于当日及历史数据产生信号，T+1 日开盘价执行，<b>任何信号均不使用未来数据</b>（已两次专项排查未来函数：信号对齐 T-1、状态机对齐 T-1）。</p>

<h3>二、选股策略 · 动量轮动（momentum）</h3>
<p>经典横截面动量策略，买强势股的惯性延续。同时满足以下全部条件才触发信号：</p>
<ul>
<li>动量强度：20 日收益率在全市场横截面排名 <b>Top 5%</b>；</li>
<li>趋势结构：收盘 &gt; MA20 &gt; MA60，且 60 日线相比 20 天前上行（均线多头排列）；</li>
<li>排除极端妖股：20 日收益 &lt; 60%（避免追在鱼尾行情顶部）；</li>
<li>长期背景：近 120 日收益为正；</li>
<li>流动性过滤：20 日日均成交额 ≥ 3000 万元；</li>
<li>评分：20 日收益率，策略内按日转排名分位（0–1），同日多信号按评分从高到低依次买入。</li>
</ul>
<p><b>股票池过滤</b>：剔除 ST、*ST、退市整理期个股；北交所因数据源不支持天然不含。全池基于 baostock 日线（含退市股，规避幸存者偏差）。价格口径：信号与收益使用<b>后复权序列</b>（历史值永久冻结，任意日期重跑结果可复现），成交明细展示不复权真实价；流动性过滤使用<b>真实成交额</b>（非前复权近似）。</p>

<h3>三、大盘仓位状态机（「买卖点」，锚定上证指数）</h3>
<p>总仓位不靠主观判断，由上证指数均线状态决定，参考用户腾讯文档《买卖点》实现为四态状态机。当日目标仓位由<b>前一交易日收盘状态</b>决定（防未来函数）：</p>
<table>
<thead><tr><th>状态</th><th>目标仓位</th><th>触发条件</th><th>含义</th></tr></thead>
<tbody>
<tr><td><b>Z0 空仓</b></td><td>0%（清仓）</td><td>5MA&lt;10MA 且 5MA&lt;20MA；或 5MA&lt;20MA 且当日跌幅&lt;-1%</td><td>逃顶：触发当日所有持仓挂卖出</td></tr>
<tr><td><b>Z1 轻仓</b></td><td>30%</td><td>5MA 与 10MA 同时拐头向上（抄底/回补）；或重仓/半仓状态收盘跌破 20 日线</td><td>大跌后试探性回补</td></tr>
<tr><td><b>Z2 半仓</b></td><td>50%</td><td>收盘站上 20 日线；或 Z3 状态收盘跌破 10 日线</td><td>趋势修复中，半仓参与</td></tr>
<tr><td><b>Z3 重仓</b></td><td>100%</td><td>收盘 &gt; 5MA &gt; 10MA &gt; 20MA（均线多头排列）</td><td>多头行情全仓参与</td></tr>
</tbody>
</table>
<p>除状态切换外，同状态内仓位阶梯回落：重仓破 10 日线降半仓、破 20 日线降轻仓。回测全期该状态机处于「逃顶清仓」（Z0）共 297 个交易日（占比约 41%），其中近一年 93 天（38%），是回撤控制的主要来源。仓位控制以「约束新开仓」实现：状态降档后存量持仓随止损/到期自然回落（不做强制减仓，仅 Z0 强制清仓）。</p>
<p><b>锚定指数 A/B（引擎修复后重测）</b>：仓位记账修复后，锚定指数对照在真实口径下重跑——<b>中证1000锚定全面占优</b>：全期 +44.4% vs 上证 +21.2%，最大回撤 -38.9% vs -49.1%，夏普 0.60 vs 0.38，分年 2024 +3.8% vs -30.6%、2025 +11.9% vs +21.7%、2026 +35.9% vs +58.7%。原因：本策略持仓为小盘动量股，中证1000 与其相关性远高于上证，2024 年初微盘踩踏中中证1000 状态机更早触发 Z0 空仓、真实躲过踩踏（修复前旧口径的 A/B 结论「维持上证」建立在失效的仓位记账上，已被推翻）。<b>当前版本仍锚定上证，换锚待决策</b>——若切换，预期收益/回撤/夏普全面改善，2026 年兑现略慢。</p>

<h3>四、卖出规则（优先级从高到低）</h3>
<ul>
<li><b>逃顶清仓</b>：大盘进入 Z0 空仓状态，全部持仓无条件挂卖（最高优先级）；</li>
<li><b>止损</b>：收盘价 ≤ 买入价 × 0.92（-8%硬止损）；</li>
<li><b>持有到期</b>：持有满 10 个交易日强制退出；</li>
<li><b>趋势破位</b>：持有 ≥ 3 日后，收盘跌破 MA20；</li>
<li><b>放量滞涨</b>：持有 ≥ 2 日后，成交量 ≥ 2 × 前一日5日均量且当日收阴（主力出货嫌疑）。</li>
</ul>

<h3>五、交易执行与成本假设</h3>
<ul>
<li><b>执行时点</b>：T 日收盘出信号（含退出判定），T+1 日开盘价成交，先卖后买；</li>
<li><b>A股规则内建</b>：T+1（买入当日不可卖）；开盘涨停（开盘价≥涨停价）放弃买入；开盘跌停顺延至下一开盘卖出；停牌顺延；</li>
<li><b>整手交易</b>：买入股数为 100 股整数倍、最低 1 手，单只预算 ≈ 总资金/10，且不超过状态机目标仓位对应预算；</li>
<li><b>成本</b>：佣金万1（双边，单笔最低 5 元）+ 印花税 0.05%（卖出）+ 滑点 0.1%（单边）；</li>
<li>涨跌幅限制按板块区分：主板 10%，创业板/科创板 20%；停牌复牌首日跳空以复牌开盘价对比停牌前收盘判定；</li>
</ul>

<h3>六、数据源与已知局限</h3>
<ul>
<li>日线数据：baostock（含退市股）+ 腾讯行情增量更新；指数：上证指数日线；</li>
<li>回测区间自 2023-09 起（信号所需的 120 日动量前置数据自 2023-01 回补，<b>窗口首日信号即有效</b>，不存在前期"假空仓"）；网页展示窗口默认近一年；</li>
<li><b>可复现性</b>：信号基于后复权序列，历史值不随新除权事件改变；</li>
<li><b>局限</b>：未建模盘中撮合排队（以开盘价全额成交近似）；滑点为固定比例，小市值极端行情可能更大；未含融资利息（纯现货）；信号日若开盘涨停则机会直接放弃，实盘可能以更高成本追入；历史回测表现不代表未来收益。</li>
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
.tg-a{background:#E6F1FB;color:#185FA5;} .tg-b{background:#E1F5EE;color:#0F6E56;} .tg-c{background:#EEEDFE;color:#534AB7;}
.warn{background:#FAEEDA;border:1px solid #EF9F27;border-radius:10px;padding:10px 14px;font-size:12.5px;color:#633806;margin-bottom:14px;}
footer{color:var(--muted);font-size:11.5px;margin-top:20px;line-height:1.8;}
</style>
</head>
<body>
<div class="wrap">
<h1>A股短线策略实验室</h1>
<div class="sub" id="sub"></div>
<div class="warn" style="margin-top:14px">本报告回测覆盖<b>动量轮动</b>策略。仓位控制: 每日最多买入 3 只, 总仓位按<b>上证指数「买卖点」状态机</b>调节 (Z0 空仓 / Z1 轻仓 30% / Z2 半仓 50% / Z3 重仓 100%)。所有结果含佣金万1、印花税与滑点, 涨跌停与 T+1 规则已内建。</div>

<div class="tabs" id="rangeTabs">
  <div class="tab" data-n="5">近一周</div>
  <div class="tab" data-n="21">近一月</div>
  <div class="tab" data-n="63">近三月</div>
  <div class="tab" data-n="122">近六月</div>
  <div class="tab on" data-n="244">近一年</div>
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
const D = __DATA__;
const fmtPct = x => (x>=0?'+':'') + (x*100).toFixed(2) + '%';
const cls = x => x>=0 ? 'up' : 'down';
const nf = x => x.toLocaleString('zh-CN', {maximumFractionDigits:0});
let rangeN = 0;

// ---------- 交易明细折叠 ----------
let tradeOpen = false;
function toggleTrade(){
  tradeOpen = !tradeOpen;
  document.getElementById('tradeWrap').style.display = tradeOpen ? '' : 'none';
  document.getElementById('tradeToggle').textContent = tradeOpen ? '收起明细 ▴' : '展开明细 ▾';
}

// ---------- 区间切换 ----------
document.getElementById('rangeTabs').addEventListener('click', e => {
  if(!e.target.dataset.n) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  e.target.classList.add('on');
  rangeN = +e.target.dataset.n;
  renderAll();
});

function slice(){
  const n = rangeN === 0 ? D.dates.length : rangeN + 1;
  const s = Math.max(0, D.dates.length - n);
  return {dates: D.dates.slice(s), equity: D.equity.slice(s), bench: D.bench.slice(s),
          bench1000: (D.bench1000 || D.bench).slice(s), start: D.dates[s]};
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
    {name:'沪深300(等基准化)',type:'line',data:sl.bench,showSymbol:false,lineStyle:{width:1.2,color:'#B4B2A9',type:'dashed'}}
  ];
  eqChart.setOption({
    animation:false,
    tooltip:{trigger:'axis',valueFormatter:v=>nf(v)},
    legend:{data:series.map(s=>s.name),top:0,textStyle:{fontSize:11,color:'#88867E'}},
    grid:{left:56,right:24,top:30,bottom:26},
    xAxis:{type:'category',data:sl.dates,axisLabel:{fontSize:10,color:'#88867E'}},
    yAxis:{type:'value',scale:true,axisLabel:{fontSize:10,color:'#88867E',formatter:nf}},
    series}, true);
  document.getElementById('eqNote').textContent = `(${sl.start} ~ ${D.last_day}, 初始资金 ¥100万)`;
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
  const hr = D.hold_ratio || 0, tgt = D.reg_ratio || 0;
  const diffPt = ((hr - tgt) * 100).toFixed(0);
  const overNote = (hr - tgt) > 0.02
    ? ` · 超出目标 ${diffPt}pt (状态机仅约束新开仓, 超额部分随止损/10日到期自然回落, 仅 Z0 强制清仓)`
    : (hr - tgt) < -0.02 ? ` · 低于目标 ${Math.abs(diffPt)}pt` : '';
  document.getElementById('holdNote').textContent =
    `(${D.last_day} 收盘 · ${D.holdings.length}只 · 合计 ¥${nf(D.hold_value||0)} · 实际仓位 ${(hr*100).toFixed(0)}% vs 目标 ${(tgt*100).toFixed(0)}%${overNote})`;
  // 下一交易日计划
  document.getElementById('planNote').textContent = `(${D.last_day} 收盘判定 · 下一开盘执行 · 大盘状态 ${D.reg_state}, 目标仓位 ${(D.reg_ratio*100).toFixed(0)}%)`;
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
  const ratioPct = (D.reg_ratio*100).toFixed(0);
  const perStock = D.reg_ratio>0 ? '单只 ≤ 10% 总资金 (整手买入, 最低1手)' : '—';
  const buyCond = D.reg_ratio>0
    ? `信号已触发; 次日开盘价不为一字涨停即可买入, 开盘涨停放弃; 总仓位不超过 ${ratioPct}% (${D.reg_state})`
    : `大盘处于空仓状态 (Z0), 暂不开新仓; 待状态机回升至 Z1 及以上恢复买入`;
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
  `生成于 ${D.generated} · 报告窗口 ${D.start_day} ~ ${D.last_day} (近一年) · 初始资金 ¥100万 · 每日最多买3只 · 佣金万1+印花税0.05%+滑点0.1%`;
document.getElementById('foot').innerHTML =
  `回测口径: T日收盘出信号, T+1开盘成交; 开盘涨停放弃买入, 开盘跌停顺延卖出; 止损-8% / 趋势破位(收盘破MA20) / 放量滞涨 / 持有满10日退出; 上证指数触发「买卖点」空仓条件(Z0)时全线清仓。<br>
   仓位控制: 每日最多新开仓 3 只; 总仓位锚定上证指数状态机 — Z0 空仓0% / Z1 轻仓30% / Z2 半仓50% / Z3 重仓100%(均线多头排列)。<br>
   买入股数: 按整手(100股整数倍, 最低1手)向下取整, 单只预算≈总资金/10 且不超状态机目标仓位。<br>
   策略贡献盈亏为交易盈亏加总(不含空仓资金占用), 胜率为区间内平仓交易口径。数据源: 腾讯行情 · 东财涨停池 · baostock 股票池(含退市)。仅供研究, 不构成投资建议。`;

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
