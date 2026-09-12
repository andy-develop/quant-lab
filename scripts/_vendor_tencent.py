#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""§0.4 沪市覆盖故障修复 · 自包含 vendored 依赖。

`daily_update.py`（修复版）优先从合并仓 `quant-hub/common` 导入
`parse_qt_batch_response` / `FetchStats` / `check_stock_coverage`；当 quant-lab
**单独运行**（合并过渡期、或本地/CI 未挂 quant-hub）时，回落到本文件。

本文件是 `quant-hub/common/datasource/{__init__,vendor_tencent}.py` 与
`common/gates/coverage.py` 中**相关部分的逐字快照**，只保留 daily_update 需要的
三样东西，去掉 TokenBucket / CircuitBreaker / Vendor 等编排件（daily_update 不用）。

★ 修复要点（§0.4 根因）：腾讯批量接口限流时会把响应降级成 `v_s_sh600004="..."`
  简化格式，前缀是 `v_s_` 而不是 `v_sh`。旧实现
      sym_full = ("sh" if line.startswith("v_sh") else "sz") + sym
  对降级行判成 sz → 沪市代码拼错 → 不在 sym_map → **整批静默 continue**。
  连续 4 个交易日沪市覆盖率 1.7%（39/2316），深市 99.6%，CI 全绿。
  本实现用正则显式识别 `v_(s_)?(sh|sz|bj)`，市场判定与格式解耦，且**四条丢弃
  路径全部计数**（FetchStats），配合覆盖率**分市场硬门禁**（沪市 1.7% → 红、不落盘）。

同步纪律：本文件是快照，若 quant-hub/common 对应实现变更，需同步更新此处
（或在合并完成后删除本文件、统一走 common 导入）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

__all__ = [
    "FetchStats",
    "parse_qt_batch_response",
    "check_stock_coverage",
    "check_coverage",
    "CoverageResult",
    "CoverageGateError",
]


# ===========================================================================
# FetchStats —— 抓取统计（所有丢弃路径都必须计数，禁止静默 continue）
# ===========================================================================
@dataclass
class FetchStats:
    """抓取统计。所有 count 都必须进 runlog。"""

    vendor: str = "?"
    requested: int = 0
    ok: int = 0
    retried: int = 0
    dropped_batch_exc: int = 0        # 批次整体异常（原 snapshot_day 第一条 continue）
    dropped_short_format: int = 0     # 字段数不足 / 降级格式（第二条）
    dropped_unknown_code: int = 0     # 代码不在 universe（第三条 ★ §0.4 根因）
    dropped_parse_error: int = 0      # 解析异常（第四条，方案漏列）
    dropped_intraday: int = 0         # 半截 bar
    dropped_other: int = 0
    circuits_opened: list[str] = field(default_factory=list)
    missing_codes: list[str] = field(default_factory=list)

    @property
    def dropped_total(self) -> int:
        return (self.dropped_batch_exc + self.dropped_short_format
                + self.dropped_unknown_code + self.dropped_parse_error
                + self.dropped_intraday + self.dropped_other)

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor, "requested": self.requested, "ok": self.ok,
            "retried": self.retried, "dropped_total": self.dropped_total,
            "dropped_breakdown": {
                "batch_exception": self.dropped_batch_exc,
                "short_format": self.dropped_short_format,
                "unknown_code": self.dropped_unknown_code,
                "parse_error": self.dropped_parse_error,
                "intraday": self.dropped_intraday,
                "other": self.dropped_other,
            },
            "circuits_opened": self.circuits_opened,
            "missing_sample": self.missing_codes[:20],
            "missing_count": len(self.missing_codes),
        }


# ===========================================================================
# 腾讯批量快照解析 —— §0.4 修复核心
# ===========================================================================
SNAPSHOT_MIN_FIELDS = 40       # 完整快照响应字段数下限（简化格式远少于此）
SIMPLIFIED_MIN_FIELDS = 5      # 简化格式：v_s_sh600004="名称~代码~现价~..."

# 腾讯响应行前缀：
#   v_sh600004="..."      完整格式（沪）
#   v_sz000001="..."      完整格式（深）
#   v_s_sh600004="..."    ★ 限流降级简化格式（前缀是 v_s_）
_LINE_RE = re.compile(r'^v_(s_)?(sh|sz|bj)(\d{6})="(.*)"$')


def _market_of_line(line: str):
    """从一行响应里解析 (市场, 代码, 是否简化格式)。

    显式识别 `v_s_` 前缀，**不再让任何前缀形态落到错误的 else 分支**。
    """
    s = line.strip()
    if not s or "=" not in s:
        return None, None, False

    m = _LINE_RE.match(s)
    if not m:
        head = s.split("=", 1)[0].strip()
        mm = re.match(r'^v_?(s_)?(sh|sz|bj)(\d{6})$', head)
        if not mm:
            return None, None, False
        return mm.group(2), mm.group(3), bool(mm.group(1))

    simplified = bool(m.group(1))
    return m.group(2), m.group(3), simplified


def parse_qt_batch_response(text: str, sym_map: dict, stats: "FetchStats | None" = None,
                            *, logger=None) -> list:
    """解析腾讯批量快照响应。★ 与旧实现的区别：**不静默丢弃**，每条丢弃都计入 stats。

    sym_map: {"sh600004": "1.600004", ...}（按市场+代码索引到 secid）
    返回的 row 的 date 为 None，由调用方（snapshot_day）统一赋当日 day_ts。
    """
    stats = stats or FetchStats(vendor="tencent-qt")
    rows: list = []

    for line in text.strip().split(";"):
        line = line.strip()
        if not line or "=" not in line:
            continue

        mkt, num, simplified = _market_of_line(line)
        if mkt is None or num is None:
            stats.dropped_other += 1
            continue

        sym_full = mkt + num                      # ★ 用解析出的市场，不做二值猜测
        parts = line.split("=", 1)[1].strip().strip('"').split("~")

        if simplified:
            # ★ 限流降级格式：字段数少，只取可用字段；不再因为"字段数不足"整条丢弃
            stats.dropped_short_format += 1       # 记账：发生了降级（指标，不等于丢数据）
            row = _parse_simplified(sym_full, num, parts, sym_map, stats)
            if row:
                rows.append(row)
            continue

        if len(parts) < SNAPSHOT_MIN_FIELDS:
            stats.dropped_short_format += 1
            if logger:
                logger(f"[qt] 字段数 {len(parts)} < {SNAPSHOT_MIN_FIELDS}，"
                       f"代码 {sym_full} 疑似格式变更")
            continue

        if sym_full not in sym_map:
            # ★ §0.4 根因在这里被拦下：正常情况不应命中。
            stats.dropped_unknown_code += 1
            continue

        try:
            o, c = float(parts[5]), float(parts[3])
            h, l = float(parts[33]), float(parts[34])
            v = float(parts[6]) if parts[6] else 0.0
            prev = float(parts[4])
            # ★ unit: 腾讯 parts[37] 是**万元** -> 换算成**元**
            amt = float(parts[37]) * 1e4 if parts[37] else 0.0
        except (ValueError, IndexError):
            stats.dropped_parse_error += 1
            continue

        if c <= 0 or o <= 0:
            stats.dropped_parse_error += 1
            continue

        rows.append({
            "code": sym_map[sym_full], "date": None,
            "open": o, "close": c, "high": h, "low": l,
            "volume": int(v), "amount": amt, "prev_close": prev,
        })
    return rows


def _parse_simplified(sym_full: str, num: str, parts: list, sym_map: dict,
                      stats: FetchStats):
    """解析限流降级格式。字段不足以支撑 OHLCV 时返回 None（计数，由覆盖率门禁统一判定）。

    ★ 不臆造数据：high/low 置为现价（"当日无波动"的保守值），volume/amount 置 0，
      并打 _degraded 标记；真实值由 L1 备源（baostock / web.ifzq 日K）在后续 run 补齐。
    """
    if sym_full not in sym_map:
        stats.dropped_unknown_code += 1
        return None
    if len(parts) < SIMPLIFIED_MIN_FIELDS:
        stats.dropped_other += 1
        return None
    try:
        c = float(parts[3])
        prev = float(parts[4]) if parts[4] else 0.0
        o = float(parts[5]) if len(parts) > 5 and parts[5] else c
    except (ValueError, IndexError):
        stats.dropped_parse_error += 1
        return None
    if c <= 0:
        stats.dropped_parse_error += 1
        return None
    return {
        "code": sym_map[sym_full], "date": None,
        "open": o, "close": c, "high": c, "low": c,
        "volume": 0, "amount": 0.0, "prev_close": prev,
        "_degraded": True,
    }


# ===========================================================================
# 覆盖率硬门禁（§0.4 建议处理 3 / §9.9）—— 分市场下限是关键
# ===========================================================================
GREEN, YELLOW, RED = "green", "yellow", "red"
WARN_THRESHOLD = 0.95
FAIL_THRESHOLD = 0.80


class CoverageGateError(AssertionError):
    """覆盖率红：中止且不落盘。"""


@dataclass
class CoverageResult:
    asset: str
    day: str
    level: str = GREEN
    expected: int = 0
    got: int = 0
    overall: float = 0.0
    by_market: dict = field(default_factory=dict)
    missing_sample: list = field(default_factory=list)
    reasons: list = field(default_factory=list)

    @property
    def ok_to_write(self) -> bool:
        return self.level != RED

    def to_dict(self) -> dict:
        return {
            "asset": self.asset, "day": self.day, "level": self.level,
            "expected": self.expected, "got": self.got,
            "overall": round(self.overall, 4),
            "by_market": {k: round(v, 4) for k, v in self.by_market.items()},
            "missing_sample": self.missing_sample[:20],
            "reasons": self.reasons,
        }

    def summary(self) -> str:
        mm = " ".join(f"{k}={v:.1%}" for k, v in sorted(self.by_market.items()))
        return (f"[coverage/{self.asset}] {self.day} {self.level.upper()} "
                f"{self.got}/{self.expected} = {self.overall:.1%} ({mm})")


def _market_of(code: str) -> str:
    """短代码 -> 市场标识（沪/深/北）。"""
    c = str(code).strip()
    if c.upper().startswith("H") or not c.isdigit():
        return "other"
    if c[0] in ("6", "9"):
        return "sh"
    if c[0] in ("0", "2", "3"):
        return "sz"
    if c[0] in ("4", "8"):
        return "bj"
    return "other"


def check_coverage(got_codes: Iterable, expected_codes: Iterable, *, asset: str,
                   day: str, warn: float = WARN_THRESHOLD, fail: float = FAIL_THRESHOLD,
                   raise_on_red: bool = True, per_market_floor: bool = True) -> CoverageResult:
    """计算并判定覆盖率。

    per_market_floor: 额外按市场分别设下限（★ §0.4 关键：整体 55% 但沪市 1.7%，
                      只看整体会漏掉整市场缺失）。
    """
    got = {str(c).strip() for c in got_codes if str(c).strip()}
    exp = {str(c).strip() for c in expected_codes if str(c).strip()}
    res = CoverageResult(asset=asset, day=day, expected=len(exp), got=len(got & exp))

    if not exp:
        res.level = RED
        res.reasons.append("expected universe is empty —— 无法判定覆盖率（怀疑 universe 未加载）")
        if raise_on_red:
            raise CoverageGateError(res.summary() + " | " + "; ".join(res.reasons))
        return res

    res.overall = res.got / len(exp)

    exp_by_mkt: dict = {}
    for c in exp:
        exp_by_mkt.setdefault(_market_of(c), set()).add(c)
    got_set = got & exp
    for mkt, cs in exp_by_mkt.items():
        hit = len(cs & got_set)
        res.by_market[mkt] = hit / len(cs) if cs else 0.0

    missing = sorted(exp - got_set)
    res.missing_sample = missing[:20]

    worst = min(res.by_market.values()) if (per_market_floor and res.by_market) else res.overall
    if res.overall < fail or (per_market_floor and worst < fail):
        res.level = RED
    elif res.overall < warn or (per_market_floor and worst < warn):
        res.level = YELLOW
    else:
        res.level = GREEN

    if res.level == RED:
        bad = [f"{m}={v:.1%}" for m, v in sorted(res.by_market.items()) if v < fail]
        res.reasons.append(
            f"覆盖率红：整体 {res.overall:.1%}，市场最低 {worst:.1%}"
            + (f"（{', '.join(bad)}）" if bad else "")
            + f"；阈值 fail={fail:.0%}。宁缺勿错，中止且不落盘（21:00 补跑会补齐）"
        )
        if len(missing) > 20:
            res.reasons.append(f"缺失 {len(missing)} 只，示例 {missing[:10]}")
        if raise_on_red:
            raise CoverageGateError(res.summary() + " | " + "; ".join(res.reasons))
    elif res.level == YELLOW:
        soft = [f"{m}={v:.1%}" for m, v in sorted(res.by_market.items()) if v < warn]
        res.reasons.append(
            f"覆盖率黄：整体 {res.overall:.1%}"
            + (f"（{', '.join(soft)}）" if soft else "")
            + f"；阈值 warn={warn:.0%}。发布但状态条标黄"
        )
    return res


def check_stock_coverage(got_codes: Iterable, universe, *, day: str,
                         raise_on_red: bool = True, tradable_only: bool = True) -> CoverageResult:
    """个股专用入口：从 universe（DataFrame 或代码列表）取在市标的。

    §0.4 实测基线：正常日沪市 99.4% / 深市 95.9%；故障日沪市 **1.7%** / 深市 99.6%。
    """
    codes = _universe_codes(universe, tradable_only=tradable_only)
    return check_coverage(got_codes, codes, asset="stock", day=day, raise_on_red=raise_on_red)


def _universe_codes(universe, *, tradable_only: bool = True) -> list:
    if hasattr(universe, "columns") and hasattr(universe, "to_dict"):
        cols = set(universe.columns)
        df = universe
        if "code" not in cols:
            raise ValueError("universe DataFrame must have a 'code' column")
        if tradable_only and "status" in cols:
            try:
                df = df[df["status"].astype(str).isin(("1", "True", "true", "listed"))]
            except Exception:
                pass
        return [str(c).strip() for c in df["code"].tolist() if str(c).strip()]
    return [str(c).strip() for c in universe if str(c).strip()]
