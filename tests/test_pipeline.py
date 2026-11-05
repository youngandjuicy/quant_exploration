
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from quant.data.cache import save_daily_cache
from quant.pipeline import build_daily_panel


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

            # 提前写入缓存
            save_daily_cache(
                self.code,
                self.start_date,
                self.end_date,
                self.raw_df,
                cache_dir=tmp,
            )

            # 如果 pipeline 调用登录或下载，就让测试失败
            with patch(
                "quant.pipeline.baostock_client.login"
            ) as mock_login, patch(
                "quant.pipeline.baostock_client.query_daily"
            ) as mock_query:

                df, report = build_daily_panel(
                    [self.code],
                    self.start_date,
                    self.end_date,
                    cache_dir=tmp,
                )

                mock_login.assert_not_called()
                mock_query.assert_not_called()

            self.assertEqual(len(df), 2)
            self.assertEqual(report["n_stocks"], 1)
            self.assertEqual(report["duplicate_count"], 0)

    def test_cache_miss_triggers_download(self):

        with tempfile.TemporaryDirectory() as tmp:

            with patch(
                "quant.pipeline.baostock_client.login"
            ) as mock_login, patch(
                "quant.pipeline.baostock_client.query_daily",
                return_value=self.raw_df,
            ) as mock_query, patch(
                "quant.pipeline.baostock_client.logout"
            ) as mock_logout:

                df, report = build_daily_panel(
                    [self.code],
                    self.start_date,
                    self.end_date,
                    cache_dir=tmp,
                )

                mock_login.assert_called_once()
                mock_query.assert_called_once()
                mock_logout.assert_called_once()

            self.assertEqual(len(df), 2)

            # 确认下载结果已经写入缓存
            self.assertEqual(report["n_rows"], 2)


if __name__ == "__main__":
    unittest.main()
