
import numpy as np
import pandas as pd


def assign_quantile_groups(
    factor_df,
    n_groups=10,
    min_stocks=30,
):
    """
    按交易日进行横截面因子排序与等数量分组。

    Parameters
    ----------
    factor_df : pd.DataFrame
        date, code, factor

    n_groups : int
        分组数量。

    min_stocks : int
        每日进行分组所要求的最小有效股票数量。

    Returns
    -------
    pd.DataFrame
        date, code, group

        group 从 1 到 n_groups，
        因子值越高，组编号越大。

        保留输入行数与行顺序，
        无效分组记为缺失值。

    Notes
    -----
    不使用未来收益信息。
    相同因子值按照股票代码确定排序顺序。
    """

    # ------------------------------------------
    # 1. 参数与字段检查
    # ------------------------------------------

    if (
        isinstance(n_groups, bool)
        or not isinstance(n_groups, (int, np.integer))
        or n_groups < 2
    ):
        raise ValueError("n_groups 必须是大于等于2的整数")

    if (
        isinstance(min_stocks, bool)
        or not isinstance(min_stocks, (int, np.integer))
        or min_stocks < n_groups
    ):
        raise ValueError("min_stocks 不能小于 n_groups")

    required = {"date", "code", "factor"}

    if not required.issubset(factor_df.columns):
        raise ValueError("因子数据缺少必要字段")

    if factor_df.empty:
        raise ValueError("因子数据为空")

    # ------------------------------------------
    # 2. 整理数据
    # ------------------------------------------

    result = (
        factor_df[["date", "code"]]
        .reset_index(drop=True)
        .copy()
    )

    if result.isna().any().any():
        raise ValueError("日期或股票代码存在缺失值")

    if result.duplicated(["date", "code"]).any():
        raise ValueError("存在重复的股票-日期记录")

    values = pd.to_numeric(
        factor_df["factor"],
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )

    codes = result["code"].to_numpy()

    # 使用 0 暂时表示无效分组
    group_ids = np.zeros(
        len(result),
        dtype=np.int64,
    )

    # ------------------------------------------
    # 3. 按交易日进行横截面分组
    # ------------------------------------------

    daily_groups = (
        result.groupby("date", sort=False)
        .indices
    )

    for positions in daily_groups.values():

        positions = np.asarray(
            positions,
            dtype=np.int64,
        )

        # 只使用有限的因子值
        valid_positions = positions[
            np.isfinite(values[positions])
        ]

        n_valid = len(valid_positions)

        if n_valid < min_stocks:
            continue

        # 因子值为主要排序依据，
        # 股票代码为次要排序依据
        order = np.lexsort((
            codes[valid_positions],
            values[valid_positions],
        ))

        sorted_positions = valid_positions[order]

        # 按照排序位置分组
        ranks = np.arange(
            n_valid,
            dtype=np.int64,
        )

        assigned_groups = (
            ranks * n_groups // n_valid + 1
        )

        group_ids[sorted_positions] = assigned_groups

    # ------------------------------------------
    # 4. 输出结果
    # ------------------------------------------

    groups = pd.array(
        group_ids,
        dtype="Int64",
    )

    groups[group_ids == 0] = pd.NA

    result["group"] = groups

    return result
