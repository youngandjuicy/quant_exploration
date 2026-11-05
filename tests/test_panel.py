
import unittest

import pandas as pd

from quant.data.panel import (
    clean_daily_panel,
    validate_daily_panel,
)


class TestDailyPanel(unittest.TestCase):

    def setUp(self):

        # 构造模拟的 BaoStock 原始数据
        self.raw_df = pd.DataFrame(
            [
                [
                    "2024-01-03", "sh.600000",
                    "10.2", "10.6", "10.0", "10.5",
                    "10.2", "1200", "12600",
                    "1", "0.12", "1", "2.94", "0"
                ],
                [
                    "2024-01-02", "sh.600000",
                    "10.0", "10.5", "9.8", "10.2",
                    "10.0", "1000", "10200",
                    "1", "0.10", "1", "2.00", "0"
                ],
            ],
            columns=[
                "date", "code", "open", "high",
                "low", "close", "preclose", "volume",
                "amount", "adjustflag", "turn",
                "tradestatus", "pctChg", "isST"
            ],
        )

    def test_clean_and_validate(self):

        df = clean_daily_panel(self.raw_df)

        # 检查排序
        self.assertTrue(
            df["date"].is_monotonic_increasing
        )

        # 检查数据类型
        self.assertTrue(
            pd.api.types.is_numeric_dtype(df["close"])
        )

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(df["date"])
        )

        # 检查质量报告
        report = validate_daily_panel(df)

        self.assertEqual(report["n_rows"], 2)
        self.assertEqual(report["n_stocks"], 1)
        self.assertEqual(report["duplicate_count"], 0)

    def test_duplicate_detection(self):

        # 人为制造重复记录
        duplicated = pd.concat(
            [self.raw_df, self.raw_df.iloc[[0]]],
            ignore_index=True,
        )

        df = clean_daily_panel(duplicated)

        with self.assertRaises(ValueError):
            validate_daily_panel(df)

    def test_invalid_numeric_value(self):

        invalid_df = self.raw_df.copy()

        invalid_df.loc[0, "close"] = "invalid"

        with self.assertRaises(ValueError):
            clean_daily_panel(invalid_df)


if __name__ == "__main__":
    unittest.main()
