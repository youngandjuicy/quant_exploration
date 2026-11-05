
import baostock as bs
import pandas as pd

from .. import config


# ==================================================
# 1. 登录与退出
# ==================================================

def login():
    """
    登录 BaoStock。
    """

    result = bs.login()

    if result.error_code != "0":
        raise RuntimeError(
            f"BaoStock 登录失败：{result.error_msg}"
        )


def logout():
    """
    退出 BaoStock。
    """

    bs.logout()


# ==================================================
# 2. 下载单只股票的日线数据
# ==================================================

def query_daily(code, start_date, end_date):
    """
    查询单只股票的历史日线数据。

    Parameters
    ----------
    code : str
        股票代码，例如 sh.600000

    start_date : str
        开始日期，例如 2024-01-01

    end_date : str
        结束日期，例如 2024-06-30

    Returns
    -------
    pd.DataFrame
        BaoStock 返回的原始日线数据。
        此时数值字段仍然是字符串。
    """

    # 1. 向 BaoStock 发送请求
    rs = bs.query_history_k_data_plus(
        code,
        config.DAILY_FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency=config.DAILY_FREQUENCY,
        adjustflag=config.DAILY_ADJUST_FLAG,
    )

    # 2. 检查请求是否成功
    if rs.error_code != "0":
        raise RuntimeError(
            f"行情查询失败：{code}, "
            f"错误码：{rs.error_code}, "
            f"错误信息：{rs.error_msg}"
        )

    # 3. 读取 API 返回的数据
    rows = []

    while rs.next():
        rows.append(rs.get_row_data())

    # 4. 转换成 DataFrame
    df = pd.DataFrame(
        rows,
        columns=rs.fields,
    )

    return df
