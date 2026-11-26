
import unittest

import numpy as np
import pandas as pd

from quant.portfolio.construction import (
    build_equal_weight_portfolios,
)


class TestPortfolioConstruction(unittest.TestCase):

    def setUp(self):

        records = []

        dates = [
            "2024-01-05",
            "2024-01-12",
            "2024-01-19",
        ]

        for date in dates:

            for i in range(10):

                records.append({
                    "date": date,
                    "code": f"stock_{i:02d}",
                    "factor": float(i + 1),
                })

        self.factor_df = pd.DataFrame(records)

        self.schedule = pd.DataFrame({
            "signal_date": pd.to_datetime(dates),

            "execution_date": pd.to_datetime([
                "2024-01-08",
                "2024-01-15",
                "2024-01-22",
            ]),

            "next_execution_date": pd.to_datetime([
                "2024-01-15",
                "2024-01-22",
                None,
            ]),

            "is_complete": [
                True,
                True,
                False,
            ],
        })

    def test_equal_weight_and_group_assignment(self):

        result = build_equal_weight_portfolios(
            self.factor_df,
            self.schedule,
            factor_col="factor",
            n_groups=5,
            min_stocks=10,
        )

        # 每天构建5组，每组2只股票
        self.assertEqual(
            len(result),
            3 * 10,
        )

        # 每组权重之和为1
        group_weights = (
            result.groupby(
                ["signal_date", "group"]
            )["weight"]
            .sum()
        )

        np.testing.assert_allclose(
            group_weights.to_numpy(),
            np.ones(15),
        )

        # 每组两只股票，各占50%
        self.assertTrue(
            (result["n_stocks"] == 2).all()
        )

        self.assertTrue(
            np.allclose(
                result["weight"],
                0.5,
            )
        )

        # 因子值最高的股票属于第五组
        highest = result.loc[
            result["signal_date"]
            == pd.Timestamp("2024-01-05")
        ].query("group == 5")

        self.assertEqual(
            set(highest["code"]),
            {"stock_08", "stock_09"},
        )

    def test_schedule_alignment_and_incomplete_period(self):

        result = build_equal_weight_portfolios(
            self.factor_df,
            self.schedule,
            n_groups=5,
            min_stocks=10,
        )

        last_period = result.loc[
            result["signal_date"]
            == pd.Timestamp("2024-01-19")
        ]

        self.assertTrue(
            last_period["execution_date"].notna().all()
        )

        self.assertTrue(
            last_period["next_execution_date"].isna().all()
        )

        self.assertFalse(
            last_period["is_complete"].any()
        )

        first_period = result.loc[
            result["signal_date"]
            == pd.Timestamp("2024-01-05")
        ]

        self.assertTrue(
            (
                first_period["execution_date"]
                == pd.Timestamp("2024-01-08")
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
