
import unittest

import numpy as np
import pandas as pd

from quant.research.grouping import assign_quantile_groups


class TestGrouping(unittest.TestCase):

    def setUp(self):

        self.factor_df = pd.DataFrame({
            "date": pd.to_datetime([
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
                "2024-01-02",
            ]),
            "code": ["A", "B", "C", "D", "E", "F"],
            "factor": [0.12, -0.08, 0.05, 0.20, -0.03, 0.08],
        })

    def test_group_assignment_and_order(self):

        result = assign_quantile_groups(
            self.factor_df,
            n_groups=3,
            min_stocks=3,
        )

        self.assertEqual(
            result["group"].tolist(),
            [3, 1, 2, 3, 1, 2],
        )

        pd.testing.assert_frame_equal(
            result[["date", "code"]],
            self.factor_df[["date", "code"]],
        )

    def test_missing_factor(self):

        factor_df = self.factor_df.copy()

        factor_df.loc[0, "factor"] = np.nan

        result = assign_quantile_groups(
            factor_df,
            n_groups=3,
            min_stocks=3,
        )

        self.assertTrue(
            pd.isna(result.loc[0, "group"])
        )

        self.assertEqual(
            result["group"].notna().sum(),
            5,
        )

    def test_insufficient_stocks(self):

        result = assign_quantile_groups(
            self.factor_df,
            n_groups=3,
            min_stocks=10,
        )

        self.assertTrue(
            result["group"].isna().all()
        )

    def test_unequal_group_sizes(self):

        result = assign_quantile_groups(
            self.factor_df,
            n_groups=4,
            min_stocks=4,
        )

        counts = result["group"].value_counts()

        self.assertEqual(len(counts), 4)
        self.assertEqual(counts.sum(), 6)

        self.assertLessEqual(
            counts.max() - counts.min(),
            1,
        )

    def test_tied_factor_values(self):

        factor_df = self.factor_df.copy()

        factor_df["factor"] = 1.0

        result = assign_quantile_groups(
            factor_df,
            n_groups=3,
            min_stocks=3,
        )

        self.assertEqual(
            result["group"].tolist(),
            [1, 1, 2, 2, 3, 3],
        )

    def test_independent_daily_grouping(self):

        second_day = self.factor_df.copy()

        second_day["date"] = pd.Timestamp("2024-01-03")
        second_day["factor"] *= -1

        factor_df = pd.concat(
            [self.factor_df, second_day],
            ignore_index=True,
        )

        result = assign_quantile_groups(
            factor_df,
            n_groups=3,
            min_stocks=3,
        )

        self.assertEqual(
            result["group"].tolist(),
            [
                3, 1, 2, 3, 1, 2,
                1, 3, 2, 1, 3, 2,
            ],
        )


if __name__ == "__main__":
    unittest.main()
