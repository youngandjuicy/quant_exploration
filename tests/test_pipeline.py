
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from quant.data.cache import (
    save_daily_cache,
    load_daily_cache,
)
from quant.pipeline import build_daily_panel_to_parquet


class TestPipeline(unittest.TestCase):

    def setUp(self):

        self.code = "sh.600000"
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-31"

        self.raw_df = pd.DataFrame(
            [
                [
                    "2024-01-02", self.code,
                    "10.0", "10.5", "9.8", "10.2",
                    "10.0", "1000", "10200",
                    "1", "0.10", "1", "2.00", "0"
                ],
                [
                    "2024-01-03", self.code,
                    "10.2", "10.6", "10.0", "10.5",
                    "10.2", "1200", "12600",
                    "1", "0.12", "1", "2.94", "0"
                ],
            ],
            columns=[
                "date", "code", "open", "high",
                "low", "close", "preclose", "volume",
                "amount", "adjustflag", "turn",
                "tradestatus", "pctChg", "isST"
            ],
        )

    def test_cache_hit_skips_download(self):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)
            cache_dir = root / "cache"
            output_path = root / "daily_panel.parquet"

            # 预先写入缓存
            save_daily_cache(
                self.code,
                self.start_date,
                self.end_date,
                self.raw_df,
                cache_dir=cache_dir,
            )

            # 缓存命中时不应该访问网络
            with patch(
                "quant.pipeline.baostock_client.login"
            ) as mock_login, patch(
                "quant.pipeline.baostock_client.query_daily"
            ) as mock_query:

                report = build_daily_panel_to_parquet(
                    codes=[self.code],
                    start_date=self.start_date,
                    end_date=self.end_date,
                    output_path=output_path,
                    batch_size=1,
                    cache_dir=cache_dir,
                )

                mock_login.assert_not_called()
                mock_query.assert_not_called()

            # 检查构建报告
            self.assertEqual(report["n_rows"], 2)
            self.assertEqual(report["n_stocks"], 1)
            self.assertEqual(report["cache_hits"], 1)
            self.assertEqual(report["downloaded"], 0)

            # 检查 Parquet 文件
            self.assertTrue(output_path.exists())

            df = pd.read_parquet(output_path)

            self.assertEqual(len(df), 2)
            self.assertEqual(df["code"].nunique(), 1)

    def test_cache_miss_triggers_download(self):

        with tempfile.TemporaryDirectory() as tmp:

            root = Path(tmp)
            cache_dir = root / "cache"
            output_path = root / "daily_panel.parquet"

            with patch(
                "quant.pipeline.baostock_client.login"
            ) as mock_login, patch(
                "quant.pipeline.baostock_client.query_daily",
                return_value=self.raw_df,
            ) as mock_query, patch(
                "quant.pipeline.baostock_client.logout"
            ) as mock_logout:

                report = build_daily_panel_to_parquet(
                    codes=[self.code],
                    start_date=self.start_date,
                    end_date=self.end_date,
                    output_path=output_path,
                    batch_size=1,
                    cache_dir=cache_dir,
                )

                mock_login.assert_called_once()
                mock_query.assert_called_once()
                mock_logout.assert_called_once()

            # 检查构建报告
            self.assertEqual(report["n_rows"], 2)
            self.assertEqual(report["cache_hits"], 0)
            self.assertEqual(report["downloaded"], 1)

            # 检查 Parquet 文件
            self.assertTrue(output_path.exists())

            df = pd.read_parquet(output_path)

            self.assertEqual(len(df), 2)

            # 检查下载结果已经写入缓存
            cached_df = load_daily_cache(
                self.code,
                self.start_date,
                self.end_date,
                cache_dir=cache_dir,
            )

            self.assertIsNotNone(cached_df)
            self.assertEqual(len(cached_df), 2)


if __name__ == "__main__":
    unittest.main()
