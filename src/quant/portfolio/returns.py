
import numpy as np
import pandas as pd


# ==================================================
# 1. 股票持有期收益
# ==================================================

def compute_stock_holding_returns(
    portfolio_df,
    daily_panel,
):
    """
    计算每只股票在一个调仓区间内的持有期收益。

    买入价格：execution_date 的 open
    卖出价格：next_execution_date 的 open

    不对停牌、缺失价格进行填充。

    Returns
    -------
    pd.DataFrame

        保留原组合信息，并增加：

        entry_open
        exit_open
        stock_return
        return_status
    """

    required_portfolio = {
        "signal_date",
        "execution_date",
        "next_execution_date",
        "code",
        "group",
        "weight",
    }

    required_market = {
        "date",
        "code",
        "open",
        "tradestatus",
    }

    if not required_portfolio.issubset(
        portfolio_df.columns
    ):
        raise ValueError("组合数据缺少必要字段")

    if not required_market.issubset(
        daily_panel.columns
    ):
        raise ValueError("行情数据缺少必要字段")

    if portfolio_df.empty or daily_panel.empty:
        raise ValueError("输入数据不能为空")

    # ------------------------------------------
    # 1. 整理组合数据
    # ------------------------------------------

    result = portfolio_df.copy()

    date_columns = [
        "signal_date",
        "execution_date",
        "next_execution_date",
    ]

    for col in date_columns:

        result[col] = pd.to_datetime(
            result[col],
            errors="coerce",
        ).dt.normalize()

    result["code"] = (
        result["code"]
        .astype("string")
        .str.strip()
    )

    result["weight"] = pd.to_numeric(
        result["weight"],
        errors="coerce",
    )

    if result["code"].isna().any():
        raise ValueError("组合存在缺失股票代码")

    if (
        ~np.isfinite(result["weight"])
    ).any() or (result["weight"] <= 0).any():
        raise ValueError("组合权重必须是有限正数")

    # ------------------------------------------
    # 2. 整理行情价格
    # ------------------------------------------

    prices = daily_panel[
        ["date", "code", "open", "tradestatus"]
    ].copy()

    prices["date"] = pd.to_datetime(
        prices["date"],
        errors="coerce",
    ).dt.normalize()

    prices["code"] = (
        prices["code"]
        .astype("string")
        .str.strip()
    )

    prices["open"] = pd.to_numeric(
        prices["open"],
        errors="coerce",
    )

    prices["tradestatus"] = pd.to_numeric(
        prices["tradestatus"],
        errors="coerce",
    )

    # 只保留当前组合需要的股票和日期
    required_dates = pd.Index(
        pd.concat([
            result["execution_date"],
            result["next_execution_date"],
        ]).dropna().unique()
    )

    required_codes = result["code"].unique()

    prices = prices.loc[
        prices["date"].isin(required_dates)
        & prices["code"].isin(required_codes)
    ].copy()

    if prices.duplicated(
        ["date", "code"]
    ).any():
        raise ValueError("行情数据存在重复股票日期记录")

    # ------------------------------------------
    # 3. 获取买入价格
    # ------------------------------------------

    entry_prices = prices.rename(
        columns={
            "date": "execution_date",
            "open": "entry_open",
            "tradestatus": "entry_tradestatus",
        }
    )

    result = result.merge(
        entry_prices,
        on=["execution_date", "code"],
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------
    # 4. 获取卖出价格
    # ------------------------------------------

    exit_prices = prices.rename(
        columns={
            "date": "next_execution_date",
            "open": "exit_open",
            "tradestatus": "exit_tradestatus",
        }
    )

    result = result.merge(
        exit_prices,
        on=["next_execution_date", "code"],
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------
    # 5. 判断收益是否可以计算
    # ------------------------------------------

    complete_period = (
        result["next_execution_date"].notna()
    )

    entry_valid = (
        result["entry_tradestatus"].eq(1)
        & result["entry_open"].gt(0)
        & np.isfinite(result["entry_open"])
    )

    exit_valid = (
        result["exit_tradestatus"].eq(1)
        & result["exit_open"].gt(0)
        & np.isfinite(result["exit_open"])
    )

    valid = (
        complete_period
        & entry_valid
        & exit_valid
    )

    # ------------------------------------------
    # 6. 记录不可计算原因
    # ------------------------------------------

    conditions = [
        ~complete_period,

        result["entry_tradestatus"].ne(1)
        | result["entry_tradestatus"].isna(),

        result["entry_open"].isna()
        | result["entry_open"].le(0),

        result["exit_tradestatus"].ne(1)
        | result["exit_tradestatus"].isna(),

        result["exit_open"].isna()
        | result["exit_open"].le(0),
    ]

    choices = [
        "incomplete_period",
        "entry_not_trading",
        "invalid_entry_price",
        "exit_not_trading",
        "invalid_exit_price",
    ]

    result["return_status"] = np.select(
        conditions,
        choices,
        default="ok",
    )

    # ------------------------------------------
    # 7. 计算股票持有期收益
    # ------------------------------------------

    result["stock_return"] = np.nan

    result.loc[valid, "stock_return"] = (
        result.loc[valid, "exit_open"]
        / result.loc[valid, "entry_open"]
        - 1
    )

    return result


# ==================================================
# 2. 分组投资组合收益
# ==================================================

def compute_group_returns(stock_returns):
    """
    根据股票收益和初始目标权重计算分组收益。

    如果组内存在无法计算收益的股票，
    则该组本期收益记为 NaN。

    Returns
    -------
    pd.DataFrame

        signal_date
        execution_date
        next_execution_date
        group
        group_return
        n_stocks
        n_valid_stocks
        invalid_weight
    """

    required = {
        "signal_date",
        "execution_date",
        "next_execution_date",
        "group",
        "weight",
        "stock_return",
    }

    if not required.issubset(stock_returns.columns):
        raise ValueError("股票收益数据缺少必要字段")

    if stock_returns.empty:
        raise ValueError("股票收益数据不能为空")

    keys = [
        "signal_date",
        "execution_date",
        "next_execution_date",
        "group",
    ]

    records = []

    for key, group in stock_returns.groupby(
        keys,
        sort=True,
        dropna=False,
    ):

        weight_sum = group["weight"].sum()

        if not np.isclose(
            weight_sum,
            1.0,
            atol=1e-8,
        ):
            raise ValueError(
                f"分组 {key} 的初始权重之和不为1"
            )

        valid = group["stock_return"].notna()

        invalid_weight = group.loc[
            ~valid,
            "weight",
        ].sum()

        # 严格口径：所有成分股收益均有效
        if valid.all():

            group_return = np.sum(
                group["weight"]
                * group["stock_return"]
            )

        else:
            group_return = np.nan

        records.append({
            "signal_date": key[0],
            "execution_date": key[1],
            "next_execution_date": key[2],
            "group": key[3],
            "group_return": group_return,
            "n_stocks": len(group),
            "n_valid_stocks": int(valid.sum()),
            "invalid_weight": invalid_weight,
        })

    return pd.DataFrame(records)


# ==================================================
# 3. 多空组合收益
# ==================================================

def compute_long_short_returns(
    group_returns,
    long_group=None,
    short_group=None,
):
    """
    计算多空组合收益。

    默认：
        做多最高因子分组
        做空最低因子分组

    Long-Short = Long Return - Short Return
    """

    required = {
        "signal_date",
        "execution_date",
        "next_execution_date",
        "group",
        "group_return",
    }

    if not required.issubset(group_returns.columns):
        raise ValueError("分组收益数据缺少必要字段")

    groups = sorted(
        group_returns["group"].dropna().unique()
    )

    if len(groups) < 2:
        raise ValueError("至少需要两个分组")

    if long_group is None:
        long_group = groups[-1]

    if short_group is None:
        short_group = groups[0]

    if long_group == short_group:
        raise ValueError("多头和空头分组不能相同")

    if (
        long_group not in groups
        or short_group not in groups
    ):
        raise ValueError("指定的多空分组不存在")

    keys = [
        "signal_date",
        "execution_date",
        "next_execution_date",
    ]

    long_returns = (
        group_returns.loc[
            group_returns["group"] == long_group,
            keys + ["group_return"],
        ]
        .rename(
            columns={
                "group_return": "long_return"
            }
        )
    )

    short_returns = (
        group_returns.loc[
            group_returns["group"] == short_group,
            keys + ["group_return"],
        ]
        .rename(
            columns={
                "group_return": "short_return"
            }
        )
    )

    result = long_returns.merge(
        short_returns,
        on=keys,
        how="outer",
        validate="one_to_one",
    )

    result["long_short_return"] = (
        result["long_return"]
        - result["short_return"]
    )

    return result.sort_values(
        "signal_date"
    ).reset_index(drop=True)
