
import numpy as np
import pandas as pd


def compute_daily_rank_ic(
    factor_df,
    label_df,
    min_stocks=30,
):
    """
    计算每日横截面 Rank IC。

    Rank IC = Spearman(mom20, future_return_5d)

    Parameters
    ----------
    factor_df : pd.DataFrame
        date, code, mom20

    label_df : pd.DataFrame
        date, code, future_return_5d

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

    required_factor = {"date", "code", "mom20"}
    required_label = {
        "date",
        "code",
        "future_return_5d",
    }

    if not required_factor.issubset(factor_df.columns):
        raise ValueError("因子数据缺少必要字段")

    if not required_label.issubset(label_df.columns):
        raise ValueError("收益标签缺少必要字段")

    if len(factor_df) != len(label_df):
        raise ValueError("因子数据与收益标签行数不一致")

    # 两个计算函数都保留原始行顺序，
    # 因此可以直接对齐，避免对870万行数据进行大型merge。
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
        "mom20": pd.to_numeric(
            factor_df["mom20"],
            errors="coerce",
        ).to_numpy(dtype=np.float64),

        "future_return_5d": pd.to_numeric(
            label_df["future_return_5d"],
            errors="coerce",
        ).to_numpy(dtype=np.float64),
    })

    # ------------------------------------------
    # 3. 按交易日计算横截面 Rank IC
    # ------------------------------------------

    records = []

    for date, group in research_df.groupby(
        "date",
        sort=True,
    ):

        # 只保留同时具有因子和未来收益的股票
        valid = (
            np.isfinite(group["mom20"])
            & np.isfinite(group["future_return_5d"])
        )

        sample = group.loc[
            valid,
            ["mom20", "future_return_5d"],
        ]

        n_stocks = len(sample)

        if n_stocks < min_stocks:
            rank_ic = np.nan

        else:
            # Spearman = Pearson correlation of ranks
            factor_rank = sample["mom20"].rank(
                method="average"
            )

            return_rank = sample["future_return_5d"].rank(
                method="average"
            )

            rank_ic = factor_rank.corr(
                return_rank,
                method="pearson",
            )

        records.append({
            "date": date,
            "rank_ic": rank_ic,
            "n_stocks": n_stocks,
        })

    return pd.DataFrame(records)


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

    ic = ic[np.isfinite(ic)]

    if len(ic) == 0:
        return {
            "n_valid_days": 0,
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "positive_ratio": np.nan,
            "mean_n_stocks": np.nan,
        }

    mean_ic = ic.mean()
    ic_std = ic.std(ddof=1)

    icir = (
        mean_ic / ic_std
        if pd.notna(ic_std) and ic_std > 0
        else np.nan
    )

    valid_days = daily_ic.loc[
        ic.index
    ]

    return {
        "n_valid_days": len(ic),
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "positive_ratio": (ic > 0).mean(),
        "mean_n_stocks": valid_days["n_stocks"].mean(),
    }
