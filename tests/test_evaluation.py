
import unittest

import numpy as np
import pandas as pd

from quant.research.evaluation import (
    compute_daily_rank_ic,
    summarize_rank_ic,
    compute_daily_group_returns,
    summarize_group_returns,
    compute_group_monotonicity,
    compute_long_short_returns,
    summarize_long_short_returns,
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

                if date == "2024-01-03":
                    target = 4 - i
                else:
                    target = i + 1

                label_records.append({
                    "date": date,
                    "code": code,
                    "target": target,
                })

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

        self.assertAlmostEqual(summary["mean_ic"], 0.0)

        self.assertAlmostEqual(
            summary["ic_std"],
            np.sqrt(0.5),
        )

        self.assertAlmostEqual(summary["icir"], 0.0)
        self.assertAlmostEqual(summary["positive_ratio"], 0.5)
        self.assertAlmostEqual(summary["mean_n_stocks"], 5.0)

    def test_summary_without_valid_days(self):

        daily_ic = pd.DataFrame({
            "rank_ic": [np.nan, np.nan],
            "n_stocks": [10, 20],
        })

        summary = summarize_rank_ic(daily_ic)

        self.assertEqual(summary["n_valid_days"], 0)
        self.assertTrue(np.isnan(summary["mean_ic"]))
        self.assertTrue(np.isnan(summary["icir"]))


# ==================================================
# 分组收益测试
# ==================================================

class TestGroupEvaluation(unittest.TestCase):

    def setUp(self):

        dates = pd.to_datetime([
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
        ])

        codes = [
            "sh.600001",
            "sh.600002",
            "sh.600003",
            "sh.600004",
        ]

        records = []

        for date in dates:

            for i, code in enumerate(codes):

                if date == pd.Timestamp("2024-01-04"):
                    group = pd.NA
                else:
                    group = 1 if i < 2 else 2

                records.append({
                    "date": date,
                    "code": code,
                    "group": group,
                })

        self.group_df = pd.DataFrame(records)

        targets = [
            0.01, 0.03, 0.05, 0.07,
            0.02, np.nan, 0.06, 0.08,
            0.10, 0.20, 0.30, 0.40,
        ]

        self.target_df = pd.DataFrame({
            "date": np.repeat(dates, 4),
            "code": codes * 3,
            "target": targets,
        })

    def test_daily_group_returns(self):

        result = compute_daily_group_returns(
            self.group_df,
            self.target_df,
            n_groups=2,
        )

        self.assertEqual(len(result), 6)

        day1 = result[
            result["date"] == pd.Timestamp("2024-01-02")
        ]

        self.assertAlmostEqual(
            day1.loc[day1["group"] == 1, "mean_target"].iloc[0],
            0.02,
        )

        self.assertAlmostEqual(
            day1.loc[day1["group"] == 2, "mean_target"].iloc[0],
            0.06,
        )

        # 第二天第一组有两只股票被分配，
        # 但只有一只有效未来收益。
        day2_group1 = result[
            (result["date"] == pd.Timestamp("2024-01-03"))
            & (result["group"] == 1)
        ].iloc[0]

        self.assertEqual(day2_group1["n_assigned"], 2)
        self.assertEqual(day2_group1["n_valid"], 1)
        self.assertAlmostEqual(day2_group1["mean_target"], 0.02)

        # 第三天没有有效分组。
        day3_group1 = result[
            (result["date"] == pd.Timestamp("2024-01-04"))
            & (result["group"] == 1)
        ].iloc[0]

        self.assertEqual(day3_group1["n_assigned"], 0)
        self.assertEqual(day3_group1["n_valid"], 0)
        self.assertTrue(np.isnan(day3_group1["mean_target"]))

    def test_group_alignment_check(self):

        misaligned = self.target_df.copy()

        misaligned.loc[0, "code"], misaligned.loc[1, "code"] = (
            misaligned.loc[1, "code"],
            misaligned.loc[0, "code"],
        )

        with self.assertRaises(ValueError):

            compute_daily_group_returns(
                self.group_df,
                misaligned,
                n_groups=2,
            )

    def test_duplicate_stock_date(self):

        duplicated_groups = pd.concat(
            [self.group_df, self.group_df.iloc[[0]]],
            ignore_index=True,
        )

        duplicated_targets = pd.concat(
            [self.target_df, self.target_df.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):

            compute_daily_group_returns(
                duplicated_groups,
                duplicated_targets,
                n_groups=2,
            )

    def test_invalid_group_value(self):

        invalid_groups = self.group_df.copy()

        invalid_groups.loc[0, "group"] = 3

        with self.assertRaises(ValueError):

            compute_daily_group_returns(
                invalid_groups,
                self.target_df,
                n_groups=2,
            )

    def test_group_summary_and_monotonicity(self):

        daily = compute_daily_group_returns(
            self.group_df,
            self.target_df,
            n_groups=2,
        )

        summary = summarize_group_returns(daily)

        group1 = summary[
            summary["group"] == 1
        ].iloc[0]

        group2 = summary[
            summary["group"] == 2
        ].iloc[0]

        self.assertEqual(group1["n_days"], 3)
        self.assertEqual(group1["n_valid_days"], 2)

        self.assertAlmostEqual(
            group1["mean_target"],
            0.02,
        )

        self.assertAlmostEqual(
            group2["mean_target"],
            0.065,
        )

        self.assertAlmostEqual(
            group2["target_std"],
            np.std([0.06, 0.07], ddof=1),
        )

        monotonicity = compute_group_monotonicity(
            summary
        )

        self.assertEqual(monotonicity["n_groups"], 2)

        self.assertAlmostEqual(
            monotonicity["group_return_spearman"],
            1.0,
        )

    def test_long_short_returns(self):

        daily = compute_daily_group_returns(
            self.group_df,
            self.target_df,
            n_groups=2,
        )

        long_short = compute_long_short_returns(
            daily,
            low_group=1,
            high_group=2,
        )

        self.assertEqual(len(long_short), 3)

        np.testing.assert_allclose(
            long_short["long_short_return"].to_numpy(),
            [0.04, 0.05, np.nan],
            equal_nan=True,
        )

        summary = summarize_long_short_returns(
            long_short
        )

        self.assertEqual(summary["n_valid_days"], 2)

        self.assertAlmostEqual(
            summary["mean_long_short"],
            0.045,
        )

        self.assertAlmostEqual(
            summary["positive_ratio"],
            1.0,
        )

    def test_invalid_number_of_groups(self):

        for value in [0, 1, True, 1.5]:

            with self.subTest(value=value):

                with self.assertRaises(ValueError):

                    compute_daily_group_returns(
                        self.group_df,
                        self.target_df,
                        n_groups=value,
                    )


if __name__ == "__main__":
    unittest.main()
