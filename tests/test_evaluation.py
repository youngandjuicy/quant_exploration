
import unittest

import numpy as np
import pandas as pd

from quant.research.evaluation import (
    compute_daily_rank_ic,
    summarize_rank_ic,
)


class TestEvaluation(unittest.TestCase):

    def setUp(self):

        dates = [
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ]

        codes = [
            "sh.600001",
            "sh.600002",
            "sh.600003",
            "sh.600004",
        ]

        factor_records = []
        label_records = []

        for date in dates:

            for i, code in enumerate(codes):

                factor_records.append({
                    "date": date,
                    "code": code,
                    "factor": i + 1,
                })

                if date == "2024-01-02":
                    target = i + 1

                elif date == "2024-01-03":
                    target = 4 - i

                else:
                    target = i + 1

                label_records.append({
                    "date": date,
                    "code": code,
                    "target": target,
                })

        # 第三天只保留两个有效样本
        factor_records[8]["factor"] = np.nan
        factor_records[9]["factor"] = np.nan

        self.factor_df = pd.DataFrame(factor_records)
        self.label_df = pd.DataFrame(label_records)

    def test_daily_rank_ic(self):

        result = compute_daily_rank_ic(
            self.factor_df,
            self.label_df,
            min_stocks=3,
        )

        self.assertEqual(len(result), 3)

        np.testing.assert_allclose(
            result["rank_ic"].to_numpy(),
            [1.0, -1.0, np.nan],
            equal_nan=True,
        )

        self.assertEqual(
            result["n_stocks"].tolist(),
            [4, 4, 2],
        )

    def test_alignment_check(self):

        misaligned = self.label_df.copy()

        # 调换前两只股票的标签代码
        misaligned.loc[0, "code"], misaligned.loc[1, "code"] = (
            misaligned.loc[1, "code"],
            misaligned.loc[0, "code"],
        )

        with self.assertRaises(ValueError):

            compute_daily_rank_ic(
                self.factor_df,
                misaligned,
                min_stocks=3,
            )

    def test_invalid_min_stocks(self):

        for value in [0, 1, True, 1.5]:

            with self.subTest(value=value):

                with self.assertRaises(ValueError):

                    compute_daily_rank_ic(
                        self.factor_df,
                        self.label_df,
                        min_stocks=value,
                    )

    def test_summary(self):

        daily_ic = pd.DataFrame({
            "date": pd.date_range(
                "2024-01-02",
                periods=3,
            ),
            "rank_ic": [0.5, -0.5, np.nan],
            "n_stocks": [4, 6, 2],
        })

        summary = summarize_rank_ic(daily_ic)

        self.assertEqual(summary["n_valid_days"], 2)

        self.assertAlmostEqual(
            summary["mean_ic"],
            0.0,
        )

        self.assertAlmostEqual(
            summary["ic_std"],
            np.sqrt(0.5),
        )

        self.assertAlmostEqual(
            summary["icir"],
            0.0,
        )

        self.assertAlmostEqual(
            summary["positive_ratio"],
            0.5,
        )

        self.assertAlmostEqual(
            summary["mean_n_stocks"],
            5.0,
        )

    def test_summary_without_valid_days(self):

        daily_ic = pd.DataFrame({
            "rank_ic": [np.nan, np.nan],
            "n_stocks": [10, 20],
        })

        summary = summarize_rank_ic(daily_ic)

        self.assertEqual(summary["n_valid_days"], 0)
        self.assertTrue(np.isnan(summary["mean_ic"]))
        self.assertTrue(np.isnan(summary["icir"]))


if __name__ == "__main__":
    unittest.main()
