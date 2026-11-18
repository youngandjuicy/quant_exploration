
import numpy as np
import pandas as pd

from .. import config


def compute_momentum(
    daily_panel,
    trading_dates,
    window=None,
):
    """
    计算横截面股票动量因子。

    mom20 = close_t / close_t-window - 1

    时间窗口以市场交易日为准，而不是股票自身的
    有效行情记录数量。

    规则：
    1. 当前交易日必须正常交易。
    2. 当前交易日不能是 ST。
    3. window 个市场交易日前必须存在正常交易记录。
    4. 两个端点收盘价必须有效且大于零。
    5. 中间发生停牌不直接排除。
    6. 不进行价格填充。

    Returns
    -------
    pd.DataFrame
        date, code, mom20

        保留所有输入记录，无效因子记为 NaN。
    """

    # ------------------------------------------
    # 1. 参数与字段检查
    # ------------------------------------------

    if window is None:
        window = config.MOMENTUM_WINDOW

    if (
        isinstance(window, bool)
        or not isinstance(window, (int, np.integer))
        or window <= 0
    ):
        raise ValueError("window 必须是正整数")

    required = [
        "date",
        "code",
        "close",
        "tradestatus",
        "isST",
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
    # 3. 整理必要数据
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

    # 将每条行情记录映射到市场交易日编号
    session_index = np.searchsorted(
        calendar_ns,
        date_ns,
    )

    inside = session_index < len(calendar_ns)

    matched = np.zeros(len(dates), dtype=bool)

    matched[inside] = (
        calendar_ns[session_index[inside]]
        == date_ns[inside]
    )

    if not matched.all():
        raise ValueError(
            "行情数据中存在不属于交易日历的日期"
        )

    # 转换数值字段
    close = pd.to_numeric(
        daily_panel["close"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    tradestatus = pd.to_numeric(
        daily_panel["tradestatus"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    is_st = pd.to_numeric(
        daily_panel["isST"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)

    # ------------------------------------------
    # 4. 按股票寻找精确的历史交易日
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

        # 按市场交易日编号排序
        order = np.argsort(
            session_index[positions],
            kind="stable",
        )

        positions = positions[order]

        stock_sessions = session_index[positions]

        # 同一股票同一交易日不能有多条记录
        if np.any(np.diff(stock_sessions) == 0):
            raise ValueError(
                "发现重复的股票-交易日记录"
            )

        # 当前交易日对应的历史目标编号
        target_sessions = (
            stock_sessions - window
        )

        # 在该股票自己的行情记录中寻找目标日期
        locations = np.searchsorted(
            stock_sessions,
            target_sessions,
        )

        exists = (
            (target_sessions >= 0)
            & (locations < len(stock_sessions))
        )

        # 避免数组越界
        safe_locations = np.minimum(
            locations,
            len(stock_sessions) - 1,
        )

        exists &= (
            stock_sessions[safe_locations]
            == target_sessions
        )

        # 当前价格和历史价格
        current_close = close[positions]

        lag_positions = positions[safe_locations]

        lag_close = close[lag_positions]

        lag_status = tradestatus[lag_positions]

        # --------------------------------------
        # 5. 筛选有效样本
        # --------------------------------------

        valid = (
            exists
            & (tradestatus[positions] == 1)
            & (is_st[positions] == 0)
            & np.isfinite(current_close)
            & (current_close > 0)
            & (lag_status == 1)
            & np.isfinite(lag_close)
            & (lag_close > 0)
        )

        result[positions[valid]] = (
            current_close[valid]
            / lag_close[valid]
            - 1
        )

    # ------------------------------------------
    # 6. 输出结果
    # ------------------------------------------

    return pd.DataFrame({
        "date": dates.reset_index(drop=True),
        "code": codes.reset_index(drop=True),
        "mom20": result,
    })
