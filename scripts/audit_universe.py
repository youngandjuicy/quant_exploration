
from pathlib import Path

import pandas as pd

from quant import config
from quant.data import baostock_client
from quant.data.calendar import clean_trade_calendar
from quant.data.universe import (
    clean_stock_basic,
    filter_a_share_stocks,
)


AUDIT_START = "2017-12-01"
AUDIT_END = "2026-01-31"

SAMPLE_YEARS = [2018, 2021, 2025]


def main():

    output_dir = (
        config.DATA_RAW_DIR
        / "baostock"
        / "metadata"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------
    # 1. 获取原始数据
    # ----------------------------------

    baostock_client.login()

    try:

        print("下载股票基本资料...")
        raw_basic = baostock_client.query_stock_basic()

        print("下载交易日历...")
        raw_calendar = baostock_client.query_trade_dates(
            AUDIT_START,
            AUDIT_END,
        )

        calendar = clean_trade_calendar(raw_calendar)

        trading_dates = calendar.loc[
            calendar["is_trading_day"] == 1,
            "date",
        ]

        # 每年选取一个实际交易日作为审计日期
        sample_dates = []

        for year in SAMPLE_YEARS:

            year_dates = trading_dates[
                trading_dates.dt.year == year
            ]

            if len(year_dates) == 0:
                raise ValueError(f"{year} 年没有交易日数据")

            sample_dates.append(
                year_dates.iloc[len(year_dates) // 2]
            )

        # 查询历史证券列表
        snapshots = {}

        for date in sample_dates:

            day = date.strftime("%Y-%m-%d")

            print(f"查询历史证券列表：{day}")

            snapshots[day] = (
                baostock_client.query_all_stock(day)
            )

    finally:
        baostock_client.logout()

    # ----------------------------------
    # 2. 保存原始数据
    # ----------------------------------

    raw_basic.to_csv(
        output_dir / "stock_basic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raw_calendar.to_csv(
        output_dir / "trade_calendar.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for day, df in snapshots.items():

        filename = f"all_stock_{day}.csv"

        df.to_csv(
            output_dir / filename,
            index=False,
            encoding="utf-8-sig",
        )

    # ----------------------------------
    # 3. 整理股票基本资料
    # ----------------------------------

    stock_basic = clean_stock_basic(raw_basic)

    a_stocks = filter_a_share_stocks(stock_basic)

    print("\n========== 股票基本资料 ==========")

    print("全部证券数量：", len(stock_basic))
    print("沪深 A 股数量：", len(a_stocks))

    print("\n证券类型分布：")
    print(stock_basic["type"].value_counts(dropna=False))

    print("\n上市状态分布：")
    print(stock_basic["status"].value_counts(dropna=False))

    print("\n上市日期范围：")
    print(stock_basic["ipo_date"].min())
    print(stock_basic["ipo_date"].max())

    print("\n退市证券数量：")
    print(stock_basic["out_date"].notna().sum())

    # ----------------------------------
    # 4. 对比历史股票池
    # ----------------------------------

    print("\n========== 历史股票池审计 ==========")

    for day, snapshot in snapshots.items():

        date = pd.Timestamp(day)

        # BaoStock 历史证券列表
        api_codes = (
            snapshot["code"]
            .astype("string")
            .str.strip()
            .dropna()
        )

        api_codes = {
            code
            for code in api_codes
            if code.startswith(
                config.A_SHARE_CODE_PREFIXES
            )
        }

        # 根据 IPO / Out Date 推导历史股票池
        mask = (
            (a_stocks["ipo_date"] <= date)
            & (
                a_stocks["out_date"].isna()
                | (a_stocks["out_date"] >= date)
            )
        )

        derived_codes = set(
            a_stocks.loc[mask, "code"]
        )

        missing = api_codes - derived_codes
        extra = derived_codes - api_codes

        print(f"\n日期：{day}")
        print("API 股票数量：", len(api_codes))
        print("推导股票数量：", len(derived_codes))
        print("API 有但推导没有：", len(missing))
        print("推导有但 API 没有：", len(extra))

        print("API 有但推导没有的示例：",
              sorted(missing)[:10])

        print("推导有但 API 没有的示例：",
              sorted(extra)[:10])

    print("\n原始审计数据保存至：", output_dir)


if __name__ == "__main__":
    main()
