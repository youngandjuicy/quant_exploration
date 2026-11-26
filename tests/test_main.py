
import unittest
from unittest.mock import patch

import pandas as pd

import quant.main as experiment_main
from quant.main import get_split_bounds


class TestSplitBounds(unittest.TestCase):

    def setUp(self):
        self.dates = pd.bdate_range(
            "2024-01-02",
            "2024-01-15",
        )

    def test_non_final_split_excludes_last_horizon(self):

        start, end = get_split_bounds(
            self.dates,
            self.dates[0],
            self.dates[3],
            horizon=1,
            allow_future_buffer=False,
        )

        self.assertEqual(start, self.dates[0])
        self.assertEqual(end, self.dates[2])

    def test_final_test_keeps_all_signal_dates(self):

        start, end = get_split_bounds(
            self.dates,
            self.dates[7],
            self.dates[-1],
            horizon=1,
            allow_future_buffer=True,
        )

        self.assertEqual(start, self.dates[7])
        self.assertEqual(end, self.dates[-1])


class TestExperimentIntegration(unittest.TestCase):

    def test_evaluation_outputs(self):

        dates = pd.bdate_range(
            "2024-01-02",
            "2024-01-15",
        )

        factor_records = []
        target_records = []

        for date in dates:

            for i in range(30):

                code = f"sh.{600000 + i:06d}"

                factor_records.append({
                    "date": date,
                    "code": code,
                    "factor": i,
                })

                target_records.append({
                    "date": date,
                    "code": code,
                    "target": float(i),
                })

        factor_df = pd.DataFrame(factor_records)
        target_df = pd.DataFrame(target_records)

        splits = [
            (
                "Discovery",
                dates[0],
                dates[3],
                False,
            ),
            (
                "Validation",
                dates[4],
                dates[6],
                False,
            ),
            (
                "Final_Test",
                dates[7],
                dates[9],
                True,
            ),
        ]

        with patch.object(
            experiment_main,
            "SPLITS",
            splits,
        ):

            results = experiment_main.evaluate_factor(
                factor_df,
                target_df,
                dates,
                horizon=1,
            )

        # 检查三个区间的有效信号日数量
        summary = results["rank_ic_summary"].set_index("split")

        self.assertEqual(
            summary.loc["Discovery", "n_signal_days"],
            3,
        )

        self.assertEqual(
            summary.loc["Validation", "n_signal_days"],
            2,
        )

        self.assertEqual(
            summary.loc["Final_Test", "n_signal_days"],
            3,
        )

        # 每个区间应当有十组
        self.assertEqual(
            len(results["group_summary"]),
            30,
        )

        # 每个区间应当有一条单调性汇总
        self.assertEqual(
            len(results["group_monotonicity"]),
            3,
        )

        # 总计8个有效信号日
        self.assertEqual(
            len(results["daily_long_short_returns"]),
            8,
        )

        # 构造的数据是严格正向关系
        self.assertTrue(
            (
                results["group_monotonicity"][
                    "group_return_spearman"
                ] > 0.99
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
