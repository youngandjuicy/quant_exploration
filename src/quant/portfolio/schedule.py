
import pandas as pd


def build_rebalance_schedule(
    trading_dates,
    frequency="weekly",
    start_date=None,
    end_date=None,
):
    """
    根据市场交易日历生成调仓计划。

    Parameters
    ----------
    trading_dates : DatetimeIndex or array-like
        完整市场交易日历，必须严格递增且不重复。

    frequency : str
        调仓频率：
        - weekly：每周最后一个交易日产生信号
        - monthly：每月最后一个交易日产生信号

    start_date : str or None
        信号日期的起始筛选边界。

    end_date : str or None
        信号日期的结束筛选边界。

    Returns
    -------
    pd.DataFrame

        signal_date:
            信号计算日期。

        execution_date:
            信号之后的下一个市场交易日，
            即计划执行调仓的日期。

        next_signal_date:
            下一次信号计算日期。

        next_execution_date:
            下一次计划执行调仓的日期。

        is_complete:
            当前持有区间是否具有完整的
            起始和结束执行日期。
    """

    # ------------------------------------------
    # 1. 参数检查
    # ------------------------------------------

    if frequency not in {"weekly", "monthly"}:
        raise ValueError(
            "frequency 必须是 weekly 或 monthly"
        )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            trading_dates,
            errors="coerce",
        )
    ).normalize()

    if len(dates) == 0:
        raise ValueError("交易日历不能为空")

    if dates.hasnans:
        raise ValueError("交易日历存在无效日期")

    if (
        not dates.is_monotonic_increasing
        or dates.has_duplicates
    ):
        raise ValueError(
            "交易日历必须严格递增且不重复"
        )

    start = (
        pd.Timestamp(start_date).normalize()
        if start_date is not None
        else None
    )

    end = (
        pd.Timestamp(end_date).normalize()
        if end_date is not None
        else None
    )

    if (
        (start is not None and pd.isna(start))
        or (end is not None and pd.isna(end))
    ):
        raise ValueError("研究区间存在无效日期")

    if start is not None and end is not None:
        if start > end:
            raise ValueError(
                "开始日期不能晚于结束日期"
            )

    # ------------------------------------------
    # 2. 确定每个周期最后一个交易日
    # ------------------------------------------

    if frequency == "weekly":
        periods = dates.to_period("W-FRI")

    else:
        periods = dates.to_period("M")

    period_df = pd.DataFrame({
        "period": periods,
        "signal_date": dates,
    })

    signal_dates = (
        period_df.groupby(
            "period",
            sort=True,
        )["signal_date"]
        .max()
        .reset_index(drop=True)
    )

    signal_dates = pd.DatetimeIndex(signal_dates)

    # ------------------------------------------
    # 3. 确定下一交易日的执行日期
    # ------------------------------------------

    positions = dates.get_indexer(
        signal_dates
    )

    execution_dates = [
        dates[pos + 1]
        if pos + 1 < len(dates)
        else pd.NaT
        for pos in positions
    ]

    schedule = pd.DataFrame({
        "signal_date": signal_dates,
        "execution_date": pd.to_datetime(
            execution_dates
        ),
    })

    # ------------------------------------------
    # 4. 确定下一次调仓执行日期
    # ------------------------------------------

    schedule["next_signal_date"] = (
        schedule["signal_date"].shift(-1)
    )

    schedule["next_execution_date"] = (
        schedule["execution_date"].shift(-1)
    )

    schedule["is_complete"] = (
        schedule["execution_date"].notna()
        & schedule["next_execution_date"].notna()
    )

    # ------------------------------------------
    # 5. 按研究区间筛选信号日期
    # ------------------------------------------

    if start is not None:
        schedule = schedule.loc[
            schedule["signal_date"] >= start
        ]

    if end is not None:
        schedule = schedule.loc[
            schedule["signal_date"] <= end
        ]

    schedule = schedule.reset_index(drop=True)

    if schedule.empty:
        raise ValueError(
            "指定研究区间内没有调仓信号日期"
        )

    return schedule
