
import unittest

import pandas as pd

from quant.data.universe import (
    clean_stock_basic,
    filter_a_share_stocks,
)


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

if __name__ == "__main__":
    unittest.main()
