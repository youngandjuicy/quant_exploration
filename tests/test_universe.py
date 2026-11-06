
import unittest

import pandas as pd

from quant.data.universe import (
    clean_stock_basic,
    filter_a_share_stocks,
    build_historical_universe,
    get_universe_codes,
)

from quant.data.calendar import clean_trade_calendar


class TestUniverse(unittest.TestCase):

    def setUp(self):

        # 构造股票基本资料
        raw_stocks = pd.DataFrame(
            [
                ["sh.600000", "股票A", "2010-01-01", "", "1", "1"],
                ["sz.000002", "股票B", "2010-01-01", "2024-01-03", "1", "0"],
                ["sz.300001", "股票C", "2024-01-03", "", "1", "1"],
                ["sh.900901", "B股", "2010-01-01", "", "1", "1"],
                ["sz.399001", "指数", "2010-01-01", "", "2", "1"],
                ["sh.510300", "ETF", "2010-01-01", "", "5", "1"],
            ],
            columns=[
                "code", "code_name", "ipoDate",
                "outDate", "type", "status",
            ],
        )

        self.stock_basic = clean_stock_basic(raw_stocks)

        # 构造交易日历
        raw_calendar = pd.DataFrame({
            "calendar_date": [
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "is_trading_day": ["1", "1", "1", "1"],
        })

        self.calendar = clean_trade_calendar(raw_calendar)

    def test_filter_a_share_stocks(self):

        stocks = filter_a_share_stocks(
            self.stock_basic
        )

        codes = set(stocks["code"])

        self.assertEqual(
            codes,
            {
                "sh.600000",
                "sz.000002",
                "sz.300001",
            },
        )

    def test_build_historical_universe(self):

        universe = build_historical_universe(
            self.stock_basic,
            self.calendar,
        )

        # 退市股票在退市日仍然保留
        self.assertTrue(
            (
                (universe["date"] == "2024-01-03")
                & (universe["code"] == "sz.000002")
            ).any()
        )

        # 退市之后不再属于股票池
        self.assertFalse(
            (
                (universe["date"] == "2024-01-04")
                & (universe["code"] == "sz.000002")
            ).any()
        )

        # 新上市股票从上市日开始进入股票池
        self.assertFalse(
            (
                (universe["date"] == "2024-01-02")
                & (universe["code"] == "sz.300001")
            ).any()
        )

        self.assertTrue(
            (
                (universe["date"] == "2024-01-03")
                & (universe["code"] == "sz.300001")
            ).any()
        )

        # B股、指数和ETF不应该出现
        self.assertEqual(
            set(get_universe_codes(universe)),
            {
                "sh.600000",
                "sz.000002",
                "sz.300001",
            },
        )


if __name__ == "__main__":
    unittest.main()
