
import pandas as pd


def clean_trade_calendar(raw_df):
    """
    清洗 BaoStock 返回的交易日历。

    输出字段：
    date
    is_trading_day
    """

    required = [
        "calendar_date",
        "is_trading_day",
    ]

    missing = set(required) - set(raw_df.columns)

    if missing:
        raise ValueError(f"交易日历缺少字段：{missing}")

    if raw_df.empty:
        raise ValueError("交易日历为空")

    df = raw_df[required].copy()

    # 日期转换
    raw_date = df["calendar_date"].astype("string").str.strip()

    df["date"] = pd.to_datetime(
        raw_date,
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError("交易日历存在无效日期")

    # 交易日标记转换
    raw_flag = df["is_trading_day"].astype("string").str.strip()

    flag = pd.to_numeric(
        raw_flag,
        errors="coerce",
    )

    if flag.isna().any():
        raise ValueError("交易日标记存在无效值")

    if not flag.isin([0, 1]).all():
        raise ValueError("交易日标记只能为0或1")

    df["is_trading_day"] = flag.astype("Int64")

    # 检查重复日期
    if df["date"].duplicated().any():
        raise ValueError("交易日历存在重复日期")

    df = (
        df[["date", "is_trading_day"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def get_trading_dates(calendar_df):
    """
    获取所有实际交易日。
    """

    required = ["date", "is_trading_day"]

    if not set(required).issubset(calendar_df.columns):
        raise ValueError("输入不是标准交易日历")

    dates = calendar_df.loc[
        calendar_df["is_trading_day"] == 1,
        "date",
    ]

    return pd.DatetimeIndex(dates)


def get_required_data_range(
    calendar_df,
    research_start,
    research_end,
    lookback=20,
    forward_horizon=5,
):
    """
    根据研究区间、因子窗口和预测窗口，
    自动计算需要下载的完整行情区间。
    """

    start = pd.Timestamp(research_start)
    end = pd.Timestamp(research_end)

    if start > end:
        raise ValueError("研究开始日期不能晚于结束日期")

    if lookback < 0 or forward_horizon < 0:
        raise ValueError("窗口长度不能为负数")

    trading_dates = get_trading_dates(calendar_df)

    research_dates = trading_dates[
        (trading_dates >= start)
        & (trading_dates <= end)
    ]

    if len(research_dates) == 0:
        raise ValueError("研究区间内没有交易日")

    first_date = research_dates[0]
    last_date = research_dates[-1]

    # 研究开始日之前的交易日
    previous_dates = trading_dates[
        trading_dates < first_date
    ]

    # 研究结束日之后的交易日
    future_dates = trading_dates[
        trading_dates > last_date
    ]

    if len(previous_dates) < lookback:
        raise ValueError("交易日历缺少足够的历史数据")

    if len(future_dates) < forward_horizon:
        raise ValueError("交易日历缺少足够的未来数据")

    data_start = (
        previous_dates[-lookback]
        if lookback > 0
        else first_date
    )

    data_end = (
        future_dates[forward_horizon - 1]
        if forward_horizon > 0
        else last_date
    )

    return {
        "research_first_trading_date": first_date.strftime("%Y-%m-%d"),
        "research_last_trading_date": last_date.strftime("%Y-%m-%d"),
        "data_start_date": data_start.strftime("%Y-%m-%d"),
        "data_end_date": data_end.strftime("%Y-%m-%d"),
    }
