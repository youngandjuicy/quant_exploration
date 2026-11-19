
import numpy as np
import pandas as pd


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

    min_stocks : int
        计算 Rank IC 所要求的最小有效股票数量。

    Returns
    -------
    pd.DataFrame
        date, rank_ic, n_stocks
    """

    # ------------------------------------------
    # 1. 参数检查
    # ------------------------------------------

    if (
        isinstance(min_stocks, bool)
        or not isinstance(min_stocks, (int, np.integer))
        or min_stocks < 2
    ):
        raise ValueError("min_stocks 必须是大于等于2的整数")

    required_factor = {"date", "code", "factor"}
    required_label = {"date", "code", "target"}

    if not required_factor.issubset(factor_df.columns):
        raise ValueError("因子数据缺少必要字段")

    if not required_label.issubset(label_df.columns):
        raise ValueError("收益标签缺少必要字段")

    if factor_df.empty or label_df.empty:
        raise ValueError("因子数据和收益标签不能为空")

    if len(factor_df) != len(label_df):
        raise ValueError("因子数据与收益标签行数不一致")

    # 两个计算函数都保留原始行顺序，
    # 因此可以直接对齐，避免大型 merge。
    if not factor_df["date"].reset_index(drop=True).equals(
        label_df["date"].reset_index(drop=True)
    ):
        raise ValueError("因子数据与收益标签日期未对齐")

    if not factor_df["code"].reset_index(drop=True).equals(
        label_df["code"].reset_index(drop=True)
    ):
        raise ValueError("因子数据与收益标签股票代码未对齐")

    # ------------------------------------------
    # 2. 构建最小研究数据
    # ------------------------------------------

    research_df = pd.DataFrame({
        "date": factor_df["date"].to_numpy(),

        "factor": pd.to_numeric(
            factor_df["factor"],
            errors="coerce",
        ).to_numpy(dtype=np.float64, na_value=np.nan),

        "target": pd.to_numeric(
            label_df["target"],
            errors="coerce",
        ).to_numpy(dtype=np.float64, na_value=np.nan),
    })

    # ------------------------------------------
    # 3. 按交易日计算横截面 Rank IC
    # ------------------------------------------

    records = []

    for date, group in research_df.groupby(
        "date",
        sort=True,
    ):

        # 只保留同时具有有效因子和标签的股票
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
            # Spearman = Pearson correlation of ranks
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

    valid_mask = np.isfinite(ic.to_numpy(dtype=np.float64))

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
