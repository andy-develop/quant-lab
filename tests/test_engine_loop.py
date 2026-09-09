"""引擎主循环端到端测试 (合成数据, 不依赖真实 K 线库)。

锁住 run_backtest 最核心的语义 —— 此前整个每日循环零测试覆盖:
  1. T+1 对齐: T 日收盘信号 -> T+1 开盘买入, 绝不 T 日成交 (防未来函数)
  2. 止损: 收盘 <= 买入价*0.92 -> 挂卖出, 次日开盘成交
  3. 跌停顺延: 开盘跌停卖不出 -> pending_sells 顺延至下一可卖开盘
  4. 开盘涨停买不进 -> 放弃该候选
  5. 整手取整 + slot 预算: shares 是 100 的整数倍, 单只不超 总权益/MAX_POS
"""
import pandas as pd
import pytest

import engine


def _flags(dates, codes, opens, closes, shrink=False):
    """构造 load_wide 的长表 (raw 与复权价同值, raw 只用于展示不影响逻辑)。"""
    rows = []
    for d in dates:
        for i, c in enumerate(codes):
            rows.append({"date": d, "code": c,
                         "open": opens[d][i], "close": closes[d][i],
                         "raw_open": opens[d][i], "raw_close": closes[d][i],
                         "shrink": shrink})
    return pd.DataFrame(rows)


def _sigs(dates, codes, score=1.0):
    rows = [{"date": d, "code": c, "score": score, "strategy": "strat", "name": f"股{c}"}
            for d in dates for c in codes]
    return pd.DataFrame(rows)


def _mk_engine(monkeypatch, tmp_path, flags, sigs):
    """monkeypatch 掉两个加载器, 引擎对合成数据运行; 产物落 tmp_path。"""
    piv = {col: flags.pivot(index="date", columns="code", values=col)
           for col in ["open", "close", "raw_close", "raw_open", "shrink"]}
    monkeypatch.setattr(engine, "load_wide", lambda: piv)
    sw = sigs.pivot(index="date", columns="code", values="score")
    names = sigs.drop_duplicates("code").set_index("code")["name"]
    monkeypatch.setattr(engine, "load_signals_wide", lambda: ({"strat": sw}, names))
    return tmp_path


class TestTPlusOne:
    def test_signal_buys_next_open_not_same_day(self, monkeypatch, tmp_path):
        """day2 收盘出信号 -> day3 开盘买入; day2 绝无持仓。"""
        dates = pd.bdate_range("2026-01-05", periods=14)
        flat = {d: [10.0] for d in dates}
        flags = _flags(dates, ["1.600000"], flat, flat)
        sigs = _sigs([dates[2]], ["1.600000"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        eq, tr, hd = engine.run_backtest(use_regime=False, out_dir=str(out))

        assert len(tr) == 1
        assert tr.iloc[0]["entry_date"] == dates[3], "信号必须在 T+1 开盘成交"
        # day2 (信号日) 及之前没有任何持仓记录
        assert hd.empty or pd.to_datetime(hd["date"]).min() >= dates[3]

    def test_no_buy_on_first_day(self, monkeypatch, tmp_path):
        """首日无 prev_day, 不得有任何买入 (首日信号无意义)。"""
        dates = pd.bdate_range("2026-01-05", periods=5)
        flat = {d: [10.0] for d in dates}
        flags = _flags(dates, ["1.600000"], flat, flat)
        sigs = _sigs([dates[0]], ["1.600000"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        _, tr, _ = engine.run_backtest(use_regime=False, out_dir=str(out))
        assert tr.empty, "首日信号不可见, 不应有交易"


class TestStopLoss:
    def test_stop_loss_sells_next_open(self, monkeypatch, tmp_path):
        """day1 开盘 10 元买入, day2 收盘 9.0 <= 10*0.92 -> 挂卖出, day3 开盘成交。"""
        dates = pd.bdate_range("2026-01-05", periods=8)
        opens = {d: [10.0] for d in dates}
        closes = {d: [10.0] for d in dates}
        closes[dates[2]] = [9.0]          # 收盘触发止损 (9.0 <= 9.2)
        flags = _flags(dates, ["0.000001"], opens, closes)
        sigs = _sigs([dates[0]], ["0.000001"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        _, tr, _ = engine.run_backtest(use_regime=False, out_dir=str(out))

        assert len(tr) == 1
        row = tr.iloc[0]
        assert row["reason"] == "stop_loss"
        assert row["entry_date"] == dates[1]
        assert row["exit_date"] == dates[3], "止损挂单必须次日开盘执行"

    def test_limit_down_defers_sell(self, monkeypatch, tmp_path):
        """止损挂单次日开盘跌停 (8.0/9.0-1=-11.1% <= -9.8%) -> 顺延, 再下一日卖出。"""
        dates = pd.bdate_range("2026-01-05", periods=8)
        opens = {d: [10.0] for d in dates}
        closes = {d: [10.0] for d in dates}
        closes[dates[2]] = [9.0]          # day2 收盘触发止损
        opens[dates[3]] = [8.0]           # day3 开盘跌停 -> 卖不出
        closes[dates[3]] = [8.0]
        opens[dates[4]] = [7.9]           # day4 开盘 -1.25% > -9.8% -> 可卖
        flags = _flags(dates, ["0.000001"], opens, closes)
        sigs = _sigs([dates[0]], ["0.000001"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        _, tr, _ = engine.run_backtest(use_regime=False, out_dir=str(out))

        assert len(tr) == 1
        row = tr.iloc[0]
        assert row["exit_date"] == dates[4], "跌停日必须顺延, 不得在跌停开盘成交"
        assert row["reason"] == "stop_loss"


class TestLimitUpNoBuy:
    def test_open_limit_up_skipped(self, monkeypatch, tmp_path):
        """A 开盘涨停 (+10% >= 9.8% 阈值) 买不进; B 正常开盘买入。"""
        dates = pd.bdate_range("2026-01-05", periods=13)
        opens, closes = {}, {}
        for d in dates:
            opens[d] = [10.0, 10.0]
            closes[d] = [10.0, 10.0]
        opens[dates[1]] = [11.0, 10.0]    # day1: A 开盘涨停, B 正常
        codes = ["1.600000", "0.000001"]
        flags = _flags(dates, codes, opens, closes)
        sigs = _sigs([dates[0]], codes)
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        _, tr, hd = engine.run_backtest(use_regime=False, out_dir=str(out))

        assert "1.600000" not in set(hd["code"]) if len(hd) else True
        bought = tr[tr["code"] == "0.000001"]
        assert len(bought) == 1 and bought.iloc[0]["entry_date"] == dates[1]


class TestLotAndBudget:
    def test_whole_lot_and_slot_cap(self, monkeypatch, tmp_path):
        """slot = min(100万/10, 现金) = 10万; 10 元股价 -> 恰好 10000 股, 100 的整数倍。"""
        dates = pd.bdate_range("2026-01-05", periods=14)
        flat = {d: [10.0] for d in dates}
        flags = _flags(dates, ["1.600000"], flat, flat)
        sigs = _sigs([dates[2]], ["1.600000"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        eq, tr, _ = engine.run_backtest(use_regime=False, out_dir=str(out))

        row = tr.iloc[0]
        assert row["shares"] == 10_000
        assert row["shares"] % engine.LOT_SIZE == 0
        # 单只市值不得超 slot 上限 (10万 + 费用容差)
        assert row["shares"] * 10.0 <= engine.START_CAP / engine.MAX_POS * 1.01

    def test_expired_at_max_hold(self, monkeypatch, tmp_path):
        """持有 max_hold 个交易日后到期卖出 (buy 日记 1, 满 max_hold 收盘挂卖, 次日开盘成交)。"""
        dates = pd.bdate_range("2026-01-05", periods=14)
        flat = {d: [10.0] for d in dates}
        flags = _flags(dates, ["1.600000"], flat, flat)
        sigs = _sigs([dates[2]], ["1.600000"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        _, tr, _ = engine.run_backtest(use_regime=False, out_dir=str(out))

        row = tr.iloc[0]
        assert row["reason"] == "expired"
        assert row["hold_days"] == engine.MAX_HOLD
        assert row["exit_date"] == dates[13]   # day3 买入, day12 满 10 日挂卖, day13 成交


class TestEquityAccounting:
    def test_equity_starts_at_cap(self, monkeypatch, tmp_path):
        """净值曲线从初始资金起步 (买入前现金未动)。"""
        dates = pd.bdate_range("2026-01-05", periods=5)
        flat = {d: [10.0] for d in dates}
        flags = _flags(dates, ["1.600000"], flat, flat)
        sigs = _sigs([dates[2]], ["1.600000"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        eq, _, _ = engine.run_backtest(use_regime=False, out_dir=str(out))
        assert eq["equity"].iloc[0] == pytest.approx(engine.START_CAP)

    def test_flat_price_no_fee_leak(self, monkeypatch, tmp_path):
        """平价完整买卖一轮, 期末净值 = 100万 - 往返费用 (无记账泄漏)。"""
        dates = pd.bdate_range("2026-01-05", periods=14)
        flat = {d: [10.0] for d in dates}
        flags = _flags(dates, ["1.600000"], flat, flat)
        sigs = _sigs([dates[2]], ["1.600000"])
        out = _mk_engine(monkeypatch, tmp_path, flags, sigs)

        eq, tr, _ = engine.run_backtest(use_regime=False, out_dir=str(out))

        shares, px = 10_000, 10.0
        buy_cost = shares * px * (1 + engine.SLIP)
        sell_proceeds = shares * px * (1 - engine.SLIP)
        expected = engine.START_CAP - (engine.buy_fee(buy_cost) + engine.sell_fee(sell_proceeds)) \
            - (buy_cost - sell_proceeds)  # 滑点损耗
        assert eq["equity"].iloc[-1] == pytest.approx(expected, rel=1e-6)
