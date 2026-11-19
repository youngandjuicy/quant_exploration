
import baostock as bs
import pandas as pd

from .. import config


# ==================================================
# 1. 通用 API 结果转换
# ==================================================

def _result_to_dataframe(rs, api_name):
    """
    将 BaoStock QueryResult 转换为 DataFrame。
    """

    if rs.error_code != "0":
        raise RuntimeError(
            f"{api_name} 查询失败："
            f"{rs.error_code}, {rs.error_msg}"
        )

    rows = []

    while rs.next():
        rows.append(rs.get_row_data())

    return pd.DataFrame(
        rows,
        columns=rs.fields,
    )


# ==================================================
# 2. 登录与退出
# ==================================================

def login():
    result = bs.login()

    if result.error_code != "0":
        raise RuntimeError(
            f"BaoStock 登录失败：{result.error_msg}"
        )


def logout():
    bs.logout()


# ==================================================
# 3. 历史日线数据
# ==================================================

def query_daily(code, start_date, end_date):

    rs = bs.query_history_k_data_plus(
        code,
        config.DAILY_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency=config.DAILY_FREQUENCY,
        adjustflag=config.DAILY_ADJUST_FLAG,
    )

    return _result_to_dataframe(
        rs,
        "query_history_k_data_plus",
    )


# ==================================================
# 4. 股票基本资料
# ==================================================

def query_stock_basic():

    rs = bs.query_stock_basic()

    return _result_to_dataframe(
        rs,
        "query_stock_basic",
    )


# ==================================================
# 5. 交易日历
# ==================================================

def query_trade_dates(start_date, end_date):

    rs = bs.query_trade_dates(
        start_date=start_date,
        end_date=end_date,
    )

    return _result_to_dataframe(
        rs,
        "query_trade_dates",
    )

