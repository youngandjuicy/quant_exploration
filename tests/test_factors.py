
import unittest

import numpy as np
import pandas as pd

from quant.research.factors import compute_momentum


class TestMomentum(unittest.TestCase):

    def setUp(self):

        self.trading_dates = pd.DatetimeIndex([
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
        ])

        # 故意打乱输入顺序
        self.daily_panel = pd.DataFrame(
            [
                ["2024-01-04", "sh.600000", "12", "1", "0"],
                ["2024-01-02", "sh.600000", "10", "1", "0"],
                ["2024-01-05", "sh.600000", "13", "1", "1"],
                ["2024-01-03", "sh.600000", "11", "0", "0"],

                ["2024-01-04", "sh.600001", "12", "1", "0"],
                ["2024-01-02", "sh.600001", "10", "0", "0"],
                ["2024-01-05", "sh.600001", "13", "1", "0"],
                ["2024-01-03", "sh.600001", "11", "1", "0"],
            ],
            columns=[
                "date",
                "code",
                "close",
                "tradestatus",
                "isST",
            ],
        )

    def test_factor_calculation_and_order(self):

        result = compute_momentum(
            self.daily_panel,
            self.trading_dates,
            window=2,
        )

        # 输出行数与输入一致
        self.assertEqual(
            len(result),
            len(self.daily_panel),
        )

        # 股票代码顺序保持不变
        self.assertEqual(
            result["code"].tolist(),
            self.daily_panel["code"].tolist(),
        )

        # 日期顺序保持不变
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            self.daily_panel["date"].tolist(),
        )

        expected = np.array([
            0.2,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            13 / 11 - 1,
            np.nan,
        ])

        np.testing.assert_allclose(
            result["factor"].to_numpy(),
            expected,
            equal_nan=True,
        )

    def test_invalid_window(self):

        for window in [0, -1, True, 1.5]:

            with self.subTest(window=window):

                with self.assertRaises(ValueError):

                    compute_momentum(
                        self.daily_panel,
                        self.trading_dates,
                        window=window,
                    )

    def test_duplicate_stock_date(self):

        duplicated = pd.concat(
            [
                self.daily_panel,
                self.daily_panel.iloc[[0]],
            ],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):

            compute_momentum(
                duplicated,
                self.trading_dates,
                window=2,
            )


if __name__ == "__main__":
    unittest.main()
