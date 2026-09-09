"""费率与涨跌停幅度 (被咬过的坑: 最低佣金 5 元 / 印花税无下限 / 板块区分)。"""
import pytest

from engine import COMM, STAMP, buy_fee, limit_pct, sell_fee


class TestFees:
    def test_buy_fee_minimum(self):
        # 1000 元 * 0.0001 = 0.1 元 < 最低 5 元 -> 取 5
        assert buy_fee(1000) == 5.0

    def test_buy_fee_proportional(self):
        assert buy_fee(100_000) == pytest.approx(100_000 * COMM)

    def test_sell_fee_stamp_no_floor(self):
        # 佣金有 5 元下限, 印花税无下限
        assert sell_fee(1000) == pytest.approx(5.0 + 1000 * STAMP)

    def test_sell_fee_proportional(self):
        assert sell_fee(100_000) == pytest.approx(10.0 + 50.0)


class TestLimitPct:
    @pytest.mark.parametrize("code", ["1.600000", "sh.600000", "0.000001", "sz.000001"])
    def test_main_board_10(self, code):
        assert limit_pct(code) == 0.10

    @pytest.mark.parametrize("code", ["0.300750", "sz.300750"])   # 创业板
    def test_chinext_20(self, code):
        assert limit_pct(code) == 0.20

    @pytest.mark.parametrize("code", ["1.688001", "sh.688001"])   # 科创板
    def test_star_20(self, code):
        assert limit_pct(code) == 0.20
