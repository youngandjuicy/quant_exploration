
import unittest

import pandas as pd

from quant.portfolio.schedule import (
    build_rebalance_schedule,
)


class TestRebalanceSchedule(unittest.TestCase):

    def setUp(self):

        self.trading_dates = pd.to_datetime([
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
            "2024-01-11",
            "2024-01-12",
            "2024-01-15",
            "2024-01-16",
            "2024-01-17",
            "2024-01-18",
            "2024-01-19",
            "2024-01-22",
            "2024-01-23",
            "2024-01-24",
            "2024-01-25",
            "2024-01-26",
            "2024-01-29",
        ])

    def test_weekly_schedule(self):

        result = build_rebalance_schedule(
            self.trading_dates,
            frequency="weekly",
            start_date="2024-01-01",
            end_date="2024-01-19",
        )

        self.assertEqual(
            result["signal_date"].dt.strftime(
                "%Y-%m-%d"
            ).tolist(),
            [
                "2024-01-05",
                "2024-01-12",
                "2024-01-19",
            ],
        )

        self.assertEqual(
            result["execution_date"].dt.strftime(
                "%Y-%m-%d"
            ).tolist(),
            [
                "2024-01-08",
                "2024-01-15",
                "2024-01-22",
            ],
        )

        self.assertEqual(
            result["next_execution_date"].dt.strftime(
                "%Y-%m-%d"
            ).tolist(),
            [
                "2024-01-15",
                "2024-01-22",
                "2024-01-29",
            ],
        )

        self.assertTrue(
            result["is_complete"].all()
        )

    def test_monthly_schedule(self):

        dates = pd.to_datetime([
            "2024-01-02",
            "2024-01-31",
            "2024-02-01",
            "2024-02-29",
            "2024-03-01",
            "2024-03-29",
            "2024-04-01",
            "2024-04-30",
            "2024-05-01",
        ])

        result = build_rebalance_schedule(
            dates,
            frequency="monthly",
            start_date="2024-01-01",
            end_date="2024-03-31",
        )

        self.assertEqual(
            result["signal_date"].dt.strftime(
                "%Y-%m-%d"
            ).tolist(),
            [
                "2024-01-31",
                "2024-02-29",
                "2024-03-29",
            ],
        )

        self.assertEqual(
            result["execution_date"].dt.strftime(
                "%Y-%m-%d"
            ).tolist(),
            [
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
            ],
        )

        self.assertEqual(
            result["next_execution_date"].dt.strftime(
                "%Y-%m-%d"
            ).tolist(),
            [
                "2024-03-01",
                "2024-04-01",
                "2024-05-01",
            ],
        )

    def test_invalid_calendar_and_frequency(self):

        with self.assertRaises(ValueError):

            build_rebalance_schedule(
                self.trading_dates[::-1],
                frequency="weekly",
            )

        with self.assertRaises(ValueError):

            build_rebalance_schedule(
                self.trading_dates,
                frequency="daily",
            )


if __name__ == "__main__":
    unittest.main()
