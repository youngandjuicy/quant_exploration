
import numpy as np
import pandas as pd

from ..research.grouping import assign_quantile_groups


def build_equal_weight_portfolios(
    factor_df,
    schedule,
    factor_col="factor",
    n_groups=10,
    min_stocks=30,
):
    """
    根据因子值和调仓计划构建等权分组投资组合。

    Parameters
    ----------
    factor_df : pd.DataFrame
        date, code, factor_col

        每个交易日的横截面因子数据。

    schedule : pd.DataFrame
        signal_date,
        execution_date,
        next_execution_date

        由 build_rebalance_schedule 生成。

    factor_col : str
        实际因子列名，例如 factor 或 mom20。

    n_groups : int
        横截面分组数量。

    min_stocks : int
        每个信号日要求的最小有效股票数量。

    Returns
    -------
    pd.DataFrame

        signal_date:
            因子计算日期。

        execution_date:
            计划执行调仓的日期。

        next_execution_date:
            下一次计划执行调仓的日期。

        code:
            股票代码。

        group:
            因子分组，1 到 n_groups。

        n_stocks:
            当前分组的股票数量。

        weight:
            组内等权目标权重。

        is_complete:
            当前持有区间是否完整。

    Notes
    -----
    1. 每个分组独立构建，组内等权。
    2. 每个完整分组的目标权重之和为1。
    3. 不使用未来收益信息。
    4. 保留执行日期已知但持有区间尚不完整的组合，
       供后续收益模块识别和处理。
    """

    # ------------------------------------------
    # 1. 参数与字段检查
    # ------------------------------------------

    required_factor = {
        "date",
        "code",
        factor_col,
    }

    required_schedule = {
        "signal_date",
        "execution_date",
        "next_execution_date",
    }

    if not required_factor.issubset(factor_df.columns):
        raise ValueError("因子数据缺少必要字段")

    if not required_schedule.issubset(schedule.columns):
        raise ValueError("调仓计划缺少必要字段")

    if factor_df.empty or schedule.empty:
        raise ValueError("因子数据和调仓计划不能为空")

    # ------------------------------------------
    # 2. 整理调仓计划
    # ------------------------------------------

    rebalance = schedule[
        [
            "signal_date",
            "execution_date",
            "next_execution_date",
        ]
    ].copy()

    for col in [
        "signal_date",
        "execution_date",
        "next_execution_date",
    ]:

        raw = rebalance[col]

        converted = pd.to_datetime(
            raw,
            errors="coerce",
        ).dt.normalize()

        invalid = raw.notna() & converted.isna()

        if invalid.any():
            raise ValueError(
                f"调仓计划字段 {col} 存在无效日期"
            )

        rebalance[col] = converted

    if rebalance["signal_date"].isna().any():
        raise ValueError("信号日期不能缺失")

    if rebalance["signal_date"].duplicated().any():
        raise ValueError("调仓计划存在重复信号日期")

    invalid_execution = (
        rebalance["execution_date"].notna()
        & (
            rebalance["execution_date"]
            <= rebalance["signal_date"]
        )
    )

    if invalid_execution.any():
        raise ValueError("执行日期必须晚于信号日期")

    invalid_next = (
        rebalance["next_execution_date"].notna()
        & rebalance["execution_date"].notna()
        & (
            rebalance["next_execution_date"]
            <= rebalance["execution_date"]
        )
    )

    if invalid_next.any():
        raise ValueError("下一次执行日期必须晚于当前执行日期")

    # 没有执行日期的信号无法构建可执行组合
    rebalance = rebalance.loc[
        rebalance["execution_date"].notna()
    ].copy()

    if rebalance.empty:
        raise ValueError("没有可执行的调仓日期")

    rebalance["is_complete"] = (
        rebalance["next_execution_date"].notna()
    )

    # ------------------------------------------
    # 3. 整理因子数据
    # ------------------------------------------

    factor = factor_df[
        ["date", "code", factor_col]
    ].copy()

    factor["date"] = pd.to_datetime(
        factor["date"],
        errors="coerce",
    ).dt.normalize()

    if factor["date"].isna().any():
        raise ValueError("因子数据存在无效日期")

    factor["code"] = (
        factor["code"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    if factor["code"].isna().any():
        raise ValueError("因子数据存在缺失股票代码")

    # 只保留实际需要调仓的信号日期
    signal_dates = rebalance["signal_date"]

    factor = factor.loc[
        factor["date"].isin(signal_dates)
    ].copy()

    missing_dates = signal_dates.loc[
        ~signal_dates.isin(factor["date"].unique())
    ]

    if not missing_dates.empty:
        raise ValueError(
            f"部分信号日期缺少因子数据："
            f"{missing_dates.dt.strftime('%Y-%m-%d').tolist()}"
        )

    # ------------------------------------------
    # 4. 按信号日进行横截面分组
    # ------------------------------------------

    grouping_input = factor.rename(
        columns={
            factor_col: "factor"
        }
    )

    grouped = assign_quantile_groups(
        grouping_input,
        n_groups=n_groups,
        min_stocks=min_stocks,
    )

    # 检查是否每个信号日都有有效分组
    valid_groups = grouped.loc[
        grouped["group"].notna()
    ]

    missing_groups = signal_dates.loc[
        ~signal_dates.isin(
            valid_groups["date"].unique()
        )
    ]

    if not missing_groups.empty:
        raise ValueError(
            "部分信号日有效因子样本不足，无法构建组合："
            f"{missing_groups.dt.strftime('%Y-%m-%d').tolist()}"
        )

    # 无效因子不参与组合
    result = valid_groups.copy()

    # ------------------------------------------
    # 5. 计算组内等权目标权重
    # ------------------------------------------

    result["n_stocks"] = (
        result.groupby(
            ["date", "group"],
            sort=False,
        )["code"]
        .transform("size")
    )

    result["weight"] = (
        1.0 / result["n_stocks"]
    )

    # ------------------------------------------
    # 6. 添加调仓日期信息
    # ------------------------------------------

    schedule_lookup = rebalance.set_index(
        "signal_date"
    )

    result["execution_date"] = (
        result["date"].map(
            schedule_lookup["execution_date"]
        )
    )

    result["next_execution_date"] = (
        result["date"].map(
            schedule_lookup["next_execution_date"]
        )
    )

    result["is_complete"] = (
        result["date"].map(
            schedule_lookup["is_complete"]
        )
    )

    result = result.rename(
        columns={
            "date": "signal_date"
        }
    )

    # ------------------------------------------
    # 7. 整理输出
    # ------------------------------------------

    result = result[
        [
            "signal_date",
            "execution_date",
            "next_execution_date",
            "code",
            "group",
            "n_stocks",
            "weight",
            "is_complete",
        ]
    ]

    result = (
        result.sort_values(
            ["signal_date", "group", "code"]
        )
        .reset_index(drop=True)
    )

    return result
