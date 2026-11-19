
import numpy as np
import pandas as pd


def compute_forward_return(
    daily_panel,
    trading_dates,
    horizon,
):
    """
    计算未来固定市场交易日的收益标签。

    target = close_{t+horizon} / open_{t+1} - 1

    规则：
    1. 使用市场交易日作为时间尺度。
    2. 下一交易日必须正常交易，且开盘价有效。
    3. 第 horizon 个后续交易日必须正常交易，
       且收盘价有效。
    4. 中间发生停牌不直接排除。
    5. 不进行价格填充。
    6. 不根据未来 ST 状态筛选样本。

    Returns
    -------
    pd.DataFrame
        date, code, target

        保留所有输入记录，无效标签记为 NaN。
    """

    # ------------------------------------------
    # 1. 参数与字段检查
    # ------------------------------------------

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, (int, np.integer))
        or horizon < 1
    ):
        raise ValueError("horizon 必须是正整数")

    required = [
        "date",
        "code",
        "open",
        "close",
        "tradestatus",
    ]

    missing = set(required) - set(daily_panel.columns)

    if missing:
        raise ValueError(f"行情数据缺少字段：{missing}")

    if daily_panel.empty:
        raise ValueError("行情数据为空")

    # ------------------------------------------
    # 2. 处理交易日历
    # ------------------------------------------

    calendar = pd.DatetimeIndex(
        pd.to_datetime(trading_dates, errors="coerce")
    ).normalize()

    if calendar.hasnans:
        raise ValueError("交易日历存在无效日期")

    if len(calendar) == 0:
        raise ValueError("交易日历为空")

    if (
        not calendar.is_monotonic_increasing
        or calendar.has_duplicates
    ):
        raise ValueError("交易日历必须严格递增且不重复")

    calendar_ns = (
        calendar.to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )

    # ------------------------------------------
    # 3. 整理行情数据
    # ------------------------------------------

    dates = pd.to_datetime(
        daily_panel["date"],
        errors="coerce",
    ).dt.normalize()

    codes = (
        daily_panel["code"]
        .astype("string")
        .str.strip()
    )

    if dates.isna().any():
        raise ValueError("行情数据存在无效日期")

    if codes.isna().any() or codes.eq("").any():
        raise ValueError("行情数据存在无效股票代码")

    date_ns = (
        dates.to_numpy(dtype="datetime64[ns]")
        .astype("int64")
    )

    # 映射到市场交易日编号
    session_index = np.searchsorted(
        calendar_ns,
        date_ns,
    )

    inside = session_index < len(calendar)

    matched = np.zeros(len(dates), dtype=bool)

    matched[inside] = (
        calendar_ns[session_index[inside]]
        == date_ns[inside]
    )

    if not matched.all():
        raise ValueError(
            "行情数据中存在不属于交易日历的日期"
        )

    open_price = pd.to_numeric(
        daily_panel["open"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    close_price = pd.to_numeric(
        daily_panel["close"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    tradestatus = pd.to_numeric(
        daily_panel["tradestatus"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    # ------------------------------------------
    # 4. 寻找未来两个交易日
    # ------------------------------------------

    result = np.full(
        len(daily_panel),
        np.nan,
        dtype=np.float64,
    )

    groups = (
        pd.DataFrame({
            "code": codes.reset_index(drop=True)
        })
        .groupby("code", sort=False)
        .indices
    )

    for positions in groups.values():

        positions = np.asarray(
            positions,
            dtype=np.int64,
        )

        order = np.argsort(
            session_index[positions],
            kind="stable",
        )

        positions = positions[order]

        stock_sessions = session_index[positions]

        if np.any(np.diff(stock_sessions) == 0):
            raise ValueError(
                "发现重复的股票-交易日记录"
            )

        # 下一交易日：t+1
        buy_target = stock_sessions + 1

        # 第 horizon 个后续交易日：t+horizon
        sell_target = stock_sessions + horizon

        buy_locations = np.searchsorted(
            stock_sessions,
            buy_target,
        )

        sell_locations = np.searchsorted(
            stock_sessions,
            sell_target,
        )

        # 检查目标日期是否存在
        buy_exists = buy_locations < len(stock_sessions)

        sell_exists = sell_locations < len(stock_sessions)

        # 防止数组越界
        safe_buy = np.minimum(
            buy_locations,
            len(stock_sessions) - 1,
        )

        safe_sell = np.minimum(
            sell_locations,
            len(stock_sessions) - 1,
        )

        buy_exists &= (
            stock_sessions[safe_buy] == buy_target
        )

        sell_exists &= (
            stock_sessions[safe_sell] == sell_target
        )

        buy_positions = positions[safe_buy]
        sell_positions = positions[safe_sell]

        # --------------------------------------
        # 5. 检查未来买入和卖出价格
        # --------------------------------------

        valid = (
            buy_exists
            & sell_exists

            # 买入日正常交易
            & (tradestatus[buy_positions] == 1)

            # 卖出日正常交易
            & (tradestatus[sell_positions] == 1)

            # 开盘价有效
            & np.isfinite(open_price[buy_positions])
            & (open_price[buy_positions] > 0)

            # 收盘价有效
            & np.isfinite(close_price[sell_positions])
            & (close_price[sell_positions] > 0)
        )

        result[positions[valid]] = (
            close_price[sell_positions[valid]]
            / open_price[buy_positions[valid]]
            - 1
        )

    # ------------------------------------------
    # 6. 输出结果
    # ------------------------------------------

    return pd.DataFrame({
        "date": dates.reset_index(drop=True),
        "code": codes.reset_index(drop=True),
        "target": result,
    })
