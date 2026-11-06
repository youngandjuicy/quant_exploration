
import unittest

import pandas as pd

from quant.data.calendar import (
    clean_trade_calendar,
    get_trading_dates,
    get_required_data_range,
)


class TestCalendar(unittest.TestCase):

    def setUp(self):

        dates = pd.date_range(
            "2023-12-01",
            "2024-02-15",
            freq="D",
        )

        raw = pd.DataFrame({
            "calendar_date": dates.strftime("%Y-%m-%d"),
            "is_trading_day": [
                "1" if d.weekday() < 5 else "0"
                for d in dates
            ],
        })

        self.calendar = clean_trade_calendar(raw)

    def test_get_trading_dates(self):

        dates = get_trading_dates(self.calendar)

        self.assertTrue(
            all(d.weekday() < 5 for d in dates)
        )

        self.assertGreater(len(dates), 0)

    def test_required_data_range(self):

        result = get_required_data_range(
            self.calendar,
            research_start="2024-01-01",
            research_end="2024-01-31",
            lookback=20,
            forward_horizon=5,
        )

        trading_dates = get_trading_dates(self.calendar)

        first = pd.Timestamp(
            result["research_first_trading_date"]
        )

        last = pd.Timestamp(
            result["research_last_trading_date"]
        )

        data_start = pd.Timestamp(
            result["data_start_date"]
        )

        data_end = pd.Timestamp(
            result["data_end_date"]
        )

        previous = trading_dates[
            (trading_dates >= data_start)
            & (trading_dates < first)
        ]

        future = trading_dates[
            (trading_dates > last)
            & (trading_dates <= data_end)
        ]

        self.assertEqual(len(previous), 20)
        self.assertEqual(len(future), 5)


if __name__ == "__main__":
    unittest.main()
