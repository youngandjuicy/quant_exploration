
import unittest

import numpy as np
import pandas as pd

from quant.research.labels import compute_forward_return


class TestForwardReturn(unittest.TestCase):

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
                ["2024-01-05", "sh.600000", "13", "14", "1"],
                ["2024-01-02", "sh.600000", "10", "10", "1"],
                ["2024-01-04", "sh.600000", "12", "12", "0"],
                ["2024-01-03", "sh.600000", "11", "12", "1"],
            ],
            columns=[
                "date",
                "code",
                "open",
                "close",
                "tradestatus",
            ],
        )

    def test_calculation_and_order(self):

        result = compute_forward_return(
            self.daily_panel,
            self.trading_dates,
            horizon=3,
        )

        # 输出行数与输入一致
        self.assertEqual(
            len(result),
            len(self.daily_panel),
        )

        # 保持输入顺序
        self.assertEqual(
            result["date"].dt.strftime("%Y-%m-%d").tolist(),
            self.daily_panel["date"].tolist(),
        )

        self.assertEqual(
            result["code"].tolist(),
            self.daily_panel["code"].tolist(),
        )

        # 2024-01-02:
        # Jan 3 开盘买入，Jan 5 收盘卖出
        # Jan 4 虽然停牌，但不直接排除
        expected = np.array([
            np.nan,
            14 / 11 - 1,
            np.nan,
            np.nan,
        ])

        np.testing.assert_allclose(
            result["target"].to_numpy(),
            expected,
            equal_nan=True,
        )

    def test_invalid_horizon(self):

        for horizon in [0, -1, True, 1.5]:

            with self.subTest(horizon=horizon):

                with self.assertRaises(ValueError):

                    compute_forward_return(
                        self.daily_panel,
                        self.trading_dates,
                        horizon=horizon,
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

            compute_forward_return(
                duplicated,
                self.trading_dates,
                horizon=3,
            )


if __name__ == "__main__":
    unittest.main()
