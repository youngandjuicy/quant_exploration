
from quant import config
from quant.data import baostock_client


# ==================================================
# 1. 元数据日期范围
# ==================================================

# 必须覆盖研究区间所需的历史回看和未来收益窗口
CALENDAR_START_DATE = "2017-12-01"
CALENDAR_END_DATE = "2026-01-31"


# ==================================================
# 2. 下载并保存元数据
# ==================================================

def main():

    output_dir = (
        config.DATA_RAW_DIR
        / "baostock"
        / "metadata"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------
    # 1. 下载原始元数据
    # ------------------------------------------

    baostock_client.login()

    try:

        print("下载股票基本资料...")
        stock_basic = baostock_client.query_stock_basic()

        print("下载交易日历...")
        trade_calendar = baostock_client.query_trade_dates(
            CALENDAR_START_DATE,
            CALENDAR_END_DATE,
        )

    finally:
        baostock_client.logout()

    # ------------------------------------------
    # 2. 保存原始数据
    # ------------------------------------------

    stock_basic.to_csv(
        output_dir / "stock_basic.csv",
        index=False,
        encoding="utf-8-sig",
    )

    trade_calendar.to_csv(
        output_dir / "trade_calendar.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== 元数据下载完成 ==========")
    print("股票基本资料：", len(stock_basic))
    print("交易日历记录：", len(trade_calendar))
    print("保存目录：", output_dir)


if __name__ == "__main__":
    main()
