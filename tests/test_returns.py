
import unittest

import numpy as np
import pandas as pd

from quant.portfolio.returns import (
    compute_stock_holding_returns,
    compute_group_returns,
    compute_long_short_returns,
)


class TestPortfolioReturns(unittest.TestCase):

    def setUp(self):

        self.portfolio = pd.DataFrame({

            "signal_date": pd.to_datetime([
                "2024-01-05",
                "2024-01-05",
                "2024-01-05",
                "2024-01-05",
            ]),

            "execution_date": pd.to_datetime([
                "2024-01-08",
                "2024-01-08",
                "2024-01-08",
                "2024-01-08",
            ]),

            "next_execution_date": pd.to_datetime([
                "2024-01-15",
                "2024-01-15",
                "2024-01-15",
                "2024-01-15",
            ]),

            "code": ["A", "B", "C", "D"],
            "group": [1, 1, 2, 2],
            "weight": [0.5, 0.5, 0.5, 0.5],
        })

        self.daily_panel = pd.DataFrame({

            "date": pd.to_datetime([
                "2024-01-08",
                "2024-01-08",
                "2024-01-08",
                "2024-01-08",
                "2024-01-15",
                "2024-01-15",
                "2024-01-15",
                "2024-01-15",
            ]),

            "code": [
                "A", "B", "C", "D",
                "A", "B", "C", "D",
            ],

            "open": [
                10, 20, 10, 10,
                11, 18, 12, 13,
            ],

            "tradestatus": [1] * 8,
        })

    def test_stock_and_group_returns(self):

        stock_returns = compute_stock_holding_returns(
            self.portfolio,
            self.daily_panel,
        )

        np.testing.assert_allclose(
            stock_returns["stock_return"],
            [0.1, -0.1, 0.2, 0.3],
        )

        group_returns = compute_group_returns(
            stock_returns
        )

        np.testing.assert_allclose(
            group_returns["group_return"],
            [0.0, 0.25],
            atol=1e-12,
        )

        self.assertEqual(
            group_returns["n_stocks"].tolist(),
            [2, 2],
        )

    def test_long_short_return(self):

        stock_returns = compute_stock_holding_returns(
            self.portfolio,
            self.daily_panel,
        )

        group_returns = compute_group_returns(
            stock_returns
        )

        result = compute_long_short_returns(
            group_returns
        )

        self.assertAlmostEqual(
            result.loc[0, "long_short_return"],
            0.25,
        )

    def test_untradable_stock(self):

        panel = self.daily_panel.copy()

        # D 股票在卖出日停牌
        panel.loc[
            (panel["date"] == pd.Timestamp("2024-01-15"))
            & (panel["code"] == "D"),
            "tradestatus",
        ] = 0

        stock_returns = compute_stock_holding_returns(
            self.portfolio,
            panel,
        )

        self.assertEqual(
            stock_returns.loc[
                stock_returns["code"] == "D",
                "return_status",
            ].iloc[0],
            "exit_not_trading",
        )

        group_returns = compute_group_returns(
            stock_returns
        )

        group_2 = group_returns.loc[
            group_returns["group"] == 2
        ].iloc[0]

        self.assertTrue(
            np.isnan(group_2["group_return"])
        )

        self.assertAlmostEqual(
            group_2["invalid_weight"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
