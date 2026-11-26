
import numpy as np
import pandas as pd


# ==================================================
# 1. 通用数据检查
# ==================================================

def _validate_aligned_keys(
    left_df,
    right_df,
    left_name="左侧数据",
    right_name="右侧数据",
):
    """
    检查两个 DataFrame 是否具有相同且逐行对齐的
    date-code 键。

    不进行大型 merge，适合大规模日线数据。
    """

    if left_df.empty or right_df.empty:
        raise ValueError("输入数据不能为空")

    if len(left_df) != len(right_df):
        raise ValueError("两个数据集行数不一致")

    for name, df in [
        (left_name, left_df),
        (right_name, right_df),
    ]:

        if df[["date", "code"]].isna().any().any():
            raise ValueError(f"{name}存在缺失日期或股票代码")

        if df.duplicated(["date", "code"]).any():
            raise ValueError(f"{name}存在重复的股票-日期记录")

    for col, description in [
        ("date", "日期"),
        ("code", "股票代码"),
    ]:

        if not left_df[col].reset_index(drop=True).equals(
            right_df[col].reset_index(drop=True)
        ):
            raise ValueError(
                f"两个数据集的{description}未对齐"
            )


def _to_float_array(series):
    """将 Series 转换为 float64，无法转换的值记为 NaN。"""

    return pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )


# ==================================================
# 2. 每日 Rank IC
# ==================================================

def compute_daily_rank_ic(
    factor_df,
    label_df,
    min_stocks=30,
):
    """
    计算每日横截面 Rank IC。

    Rank IC = Spearman(factor, target)

    Parameters
    ----------
    factor_df : pd.DataFrame
        date, code, factor

    label_df : pd.DataFrame
        date, code, target

    Returns
    -------
    pd.DataFrame
        date, rank_ic, n_stocks
    """

    if (
        isinstance(min_stocks, bool)
        or not isinstance(min_stocks, (int, np.integer))
        or min_stocks < 2
    ):
        raise ValueError("min_stocks 必须是大于等于2的整数")

    if not {"date", "code", "factor"}.issubset(factor_df.columns):
        raise ValueError("因子数据缺少必要字段")

    if not {"date", "code", "target"}.issubset(label_df.columns):
        raise ValueError("收益标签缺少必要字段")

    _validate_aligned_keys(
        factor_df,
        label_df,
        "因子数据",
        "收益标签",
    )

    research_df = pd.DataFrame({
        "date": factor_df["date"].to_numpy(),
        "factor": _to_float_array(factor_df["factor"]),
        "target": _to_float_array(label_df["target"]),
    })

    records = []

    for date, group in research_df.groupby("date", sort=True):

        valid = (
            np.isfinite(group["factor"])
            & np.isfinite(group["target"])
        )

        sample = group.loc[
            valid,
            ["factor", "target"],
        ]

        n_stocks = len(sample)

        if n_stocks < min_stocks:
            rank_ic = np.nan

        else:
            factor_rank = sample["factor"].rank(
                method="average"
            )

            target_rank = sample["target"].rank(
                method="average"
            )

            rank_ic = factor_rank.corr(
                target_rank,
                method="pearson",
            )

        records.append({
            "date": date,
            "rank_ic": rank_ic,
            "n_stocks": n_stocks,
        })

    return pd.DataFrame(
        records,
        columns=["date", "rank_ic", "n_stocks"],
    )


def summarize_rank_ic(daily_ic):
    """
    汇总每日 Rank IC。

    ICIR = Mean IC / Std IC

    使用样本标准差（ddof=1）。
    """

    if not {"rank_ic", "n_stocks"}.issubset(daily_ic.columns):
        raise ValueError("每日 IC 数据缺少必要字段")

    ic = pd.to_numeric(
        daily_ic["rank_ic"],
        errors="coerce",
    )

    valid_mask = np.isfinite(
        ic.to_numpy(dtype=np.float64)
    )

    valid_ic = ic.loc[valid_mask]

    if len(valid_ic) == 0:
        return {
            "n_valid_days": 0,
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "positive_ratio": np.nan,
            "mean_n_stocks": np.nan,
        }

    mean_ic = valid_ic.mean()
    ic_std = valid_ic.std(ddof=1)

    icir = (
        mean_ic / ic_std
        if pd.notna(ic_std) and ic_std > 0
        else np.nan
    )

    valid_n_stocks = pd.to_numeric(
        daily_ic.loc[valid_mask, "n_stocks"],
        errors="coerce",
    )

    return {
        "n_valid_days": len(valid_ic),
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "positive_ratio": (valid_ic > 0).mean(),
        "mean_n_stocks": valid_n_stocks.mean(),
    }


# ==================================================
# 3. 每日分组收益
# ==================================================

def compute_daily_group_returns(
    group_df,
    target_df,
    n_groups=10,
):
    """
    计算每日各组的等权平均目标收益。

    Parameters
    ----------
    group_df : pd.DataFrame
        date, code, group

    target_df : pd.DataFrame
        date, code, target

    n_groups : int
        分组数量。

    Returns
    -------
    pd.DataFrame
        date
        group
        mean_target
        n_assigned
        n_valid

    Notes
    -----
    分组结果必须事先根据因子计算。

    计算过程中不根据 target 是否有效
    改变股票的分组归属。
    """

    # ------------------------------------------
    # 1. 参数检查
    # ------------------------------------------

    if (
        isinstance(n_groups, bool)
        or not isinstance(n_groups, (int, np.integer))
        or n_groups < 2
    ):
        raise ValueError("n_groups 必须是大于等于2的整数")

    if not {"date", "code", "group"}.issubset(group_df.columns):
        raise ValueError("分组数据缺少必要字段")

    if not {"date", "code", "target"}.issubset(target_df.columns):
        raise ValueError("目标数据缺少必要字段")

    _validate_aligned_keys(
        group_df,
        target_df,
        "分组数据",
        "目标数据",
    )

    # ------------------------------------------
    # 2. 转换数据
    # ------------------------------------------

    raw_groups = group_df["group"]

    group_values = _to_float_array(raw_groups)
    target_values = _to_float_array(target_df["target"])

    # 非缺失的分组值必须是合法整数
    invalid_group = (
        raw_groups.notna().to_numpy()
        & (
            ~np.isfinite(group_values)
            | (group_values % 1 != 0)
            | (group_values < 1)
            | (group_values > n_groups)
        )
    )

    if invalid_group.any():
        raise ValueError("分组编号存在非法取值")

    # ------------------------------------------
    # 3. 按交易日计算各组收益
    # ------------------------------------------

    daily_positions = (
        group_df.groupby("date", sort=True).indices
    )

    records = []

    for date, positions in daily_positions.items():

        current_groups = group_values[positions]
        current_targets = target_values[positions]

        for group_id in range(1, n_groups + 1):

            # 当日实际分配到该组的股票
            assigned_mask = (
                current_groups == group_id
            )

            n_assigned = int(
                assigned_mask.sum()
            )

            # 其中具有有效目标收益的股票
            group_targets = current_targets[
                assigned_mask
            ]

            valid_targets = group_targets[
                np.isfinite(group_targets)
            ]

            n_valid = len(valid_targets)

            mean_target = (
                valid_targets.mean()
                if n_valid > 0
                else np.nan
            )

            records.append({
                "date": date,
                "group": group_id,
                "mean_target": mean_target,
                "n_assigned": n_assigned,
                "n_valid": n_valid,
            })

    return pd.DataFrame(
        records,
        columns=[
            "date",
            "group",
            "mean_target",
            "n_assigned",
            "n_valid",
        ],
    )


# ==================================================
# 4. 分组收益数据检查
# ==================================================

def _validate_daily_group_returns(daily_group_returns):

    required = {
        "date",
        "group",
        "mean_target",
        "n_assigned",
        "n_valid",
    }

    if not required.issubset(daily_group_returns.columns):
        raise ValueError("每日分组收益数据缺少必要字段")

    if daily_group_returns.empty:
        raise ValueError("每日分组收益数据为空")

    if daily_group_returns[
        ["date", "group"]
    ].isna().any().any():
        raise ValueError("每日分组收益存在缺失日期或分组")

    if daily_group_returns.duplicated(
        ["date", "group"]
    ).any():
        raise ValueError("存在重复的交易日-分组记录")

    groups = _to_float_array(
        daily_group_returns["group"]
    )

    if (
        not np.isfinite(groups).all()
        or (groups % 1 != 0).any()
        or (groups < 1).any()
    ):
        raise ValueError("分组编号不合法")

    n_assigned = _to_float_array(
        daily_group_returns["n_assigned"]
    )

    n_valid = _to_float_array(
        daily_group_returns["n_valid"]
    )

    for name, values in [
        ("n_assigned", n_assigned),
        ("n_valid", n_valid),
    ]:

        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values % 1 != 0).any()
        ):
            raise ValueError(f"{name} 存在非法取值")

    if (n_valid > n_assigned).any():
        raise ValueError("有效股票数量不能超过分组股票数量")

    mean_target = _to_float_array(
        daily_group_returns["mean_target"]
    )

    return groups.astype(np.int64), mean_target


# ==================================================
# 5. 分组收益汇总
# ==================================================

def summarize_group_returns(daily_group_returns):
    """
    汇总每个分组跨交易日的收益表现。

    Mean Target:
        每日组收益的时间序列平均值。

    Target Std:
        每日组收益的样本标准差。

    Positive Ratio:
        每日组收益大于0的比例。
    """

    groups, targets = _validate_daily_group_returns(
        daily_group_returns
    )

    records = []

    for group_id in sorted(np.unique(groups)):

        mask = groups == group_id

        group_data = daily_group_returns.loc[
            mask
        ]

        valid_targets = targets[
            mask & np.isfinite(targets)
        ]

        n_valid_days = len(valid_targets)

        mean_target = (
            valid_targets.mean()
            if n_valid_days > 0
            else np.nan
        )

        target_std = (
            valid_targets.std(ddof=1)
            if n_valid_days >= 2
            else np.nan
        )

        positive_ratio = (
            (valid_targets > 0).mean()
            if n_valid_days > 0
            else np.nan
        )

        records.append({
            "group": group_id,
            "n_days": len(group_data),
            "n_valid_days": n_valid_days,
            "mean_target": mean_target,
            "target_std": target_std,
            "positive_ratio": positive_ratio,
            "mean_n_assigned": group_data["n_assigned"].mean(),
            "mean_n_valid": group_data["n_valid"].mean(),
        })

    return pd.DataFrame(records)


# ==================================================
# 6. 分组收益单调性
# ==================================================

def compute_group_monotonicity(group_summary):
    """
    计算分组编号与分组平均收益之间的 Spearman 相关系数。

    group_return_spearman 接近：
        +1：收益随分组编号上升而上升。
        -1：收益随分组编号上升而下降。
         0：没有明显的单调排序关系。

    这是描述性指标，不代表统计显著性。
    """

    required = {"group", "mean_target"}

    if not required.issubset(group_summary.columns):
        raise ValueError("分组汇总数据缺少必要字段")

    if group_summary["group"].duplicated().any():
        raise ValueError("分组汇总存在重复分组")

    groups = _to_float_array(group_summary["group"])
    targets = _to_float_array(group_summary["mean_target"])

    valid = (
        np.isfinite(groups)
        & np.isfinite(targets)
    )

    groups = groups[valid]
    targets = targets[valid]

    if len(groups) < 2:
        correlation = np.nan

    else:
        correlation = pd.Series(groups).corr(
            pd.Series(targets),
            method="spearman",
        )

    return {
        "n_groups": len(groups),
        "group_return_spearman": correlation,
    }


# ==================================================
# 7. 每日理论多空收益
# ==================================================

def compute_long_short_returns(
    daily_group_returns,
    low_group=1,
    high_group=None,
):
    """
    计算最高组减最低组的每日理论多空收益。

    默认：
        Long = 最高编号组
        Short = 最低编号组

    不考虑融券、交易成本和资金约束。
    """

    groups, _ = _validate_daily_group_returns(
        daily_group_returns
    )

    available_groups = np.unique(groups)

    if (
        isinstance(low_group, bool)
        or not isinstance(low_group, (int, np.integer))
    ):
        raise ValueError("low_group 必须是整数")

    if high_group is None:
        high_group = int(available_groups.max())

    if (
        isinstance(high_group, bool)
        or not isinstance(high_group, (int, np.integer))
    ):
        raise ValueError("high_group 必须是整数")

    if low_group == high_group:
        raise ValueError("多空两组不能相同")

    if low_group not in available_groups:
        raise ValueError("找不到指定的低编号组")

    if high_group not in available_groups:
        raise ValueError("找不到指定的高编号组")

    # ------------------------------------------
    # 1. 提取低编号组
    # ------------------------------------------

    low = daily_group_returns.loc[
        daily_group_returns["group"] == low_group,
        [
            "date",
            "mean_target",
            "n_assigned",
            "n_valid",
        ],
    ].rename(columns={
        "mean_target": "low_target",
        "n_assigned": "low_n_assigned",
        "n_valid": "low_n_valid",
    })

    # ------------------------------------------
    # 2. 提取高编号组
    # ------------------------------------------

    high = daily_group_returns.loc[
        daily_group_returns["group"] == high_group,
        [
            "date",
            "mean_target",
            "n_assigned",
            "n_valid",
        ],
    ].rename(columns={
        "mean_target": "high_target",
        "n_assigned": "high_n_assigned",
        "n_valid": "high_n_valid",
    })

    # ------------------------------------------
    # 3. 按交易日对齐
    # ------------------------------------------

    result = low.merge(
        high,
        on="date",
        how="outer",
        validate="one_to_one",
    )

    result = result.sort_values(
        "date"
    ).reset_index(drop=True)

    result["long_short_return"] = (
        result["high_target"]
        - result["low_target"]
    )

    result["low_group"] = low_group
    result["high_group"] = high_group

    return result[
        [
            "date",
            "low_group",
            "high_group",
            "low_target",
            "high_target",
            "long_short_return",
            "low_n_assigned",
            "low_n_valid",
            "high_n_assigned",
            "high_n_valid",
        ]
    ]


# ==================================================
# 8. 多空收益汇总
# ==================================================

def summarize_long_short_returns(daily_long_short):
    """
    汇总每日理论多空收益。

    使用描述性统计，不计算基于独立样本假设的 t 检验。
    """

    required = {
        "date",
        "long_short_return",
        "low_n_valid",
        "high_n_valid",
    }

    if not required.issubset(daily_long_short.columns):
        raise ValueError("每日多空收益数据缺少必要字段")

    if daily_long_short["date"].duplicated().any():
        raise ValueError("每日多空收益存在重复日期")

    returns = _to_float_array(
        daily_long_short["long_short_return"]
    )

    valid = np.isfinite(returns)
    valid_returns = returns[valid]

    if len(valid_returns) == 0:
        return {
            "n_valid_days": 0,
            "mean_long_short": np.nan,
            "long_short_std": np.nan,
            "positive_ratio": np.nan,
            "mean_low_n_valid": np.nan,
            "mean_high_n_valid": np.nan,
        }

    return {
        "n_valid_days": len(valid_returns),
        "mean_long_short": valid_returns.mean(),
        "long_short_std": (
            valid_returns.std(ddof=1)
            if len(valid_returns) >= 2
            else np.nan
        ),
        "positive_ratio": (
            valid_returns > 0
        ).mean(),
        "mean_low_n_valid": pd.to_numeric(
            daily_long_short.loc[valid, "low_n_valid"],
            errors="coerce",
        ).mean(),
        "mean_high_n_valid": pd.to_numeric(
            daily_long_short.loc[valid, "high_n_valid"],
            errors="coerce",
        ).mean(),
    }
