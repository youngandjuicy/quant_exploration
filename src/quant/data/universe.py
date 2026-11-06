
import pandas as pd

from .. import config
from .calendar import get_trading_dates


# ==================================================
# 1. 清洗股票基本资料
# ==================================================

def clean_stock_basic(raw_df):

    required = [
        "code",
        "code_name",
        "ipoDate",
        "outDate",
        "type",
        "status",
    ]

    missing = set(required) - set(raw_df.columns)

    if missing:
        raise ValueError(f"股票基本资料缺少字段：{missing}")

    df = raw_df[required].copy()

    # 股票代码和名称
    df["code"] = (
        df["code"]
        .astype("string")
        .str.strip()
    )

    df["code_name"] = (
        df["code_name"]
        .astype("string")
        .str.strip()
    )

    if df["code"].isna().any() or df["code"].eq("").any():
        raise ValueError("股票基本资料存在无效代码")

    # 上市日期和退市日期
    for raw_col, new_col in [
        ("ipoDate", "ipo_date"),
        ("outDate", "out_date"),
    ]:

        raw_values = (
            df[raw_col]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

        parsed = pd.to_datetime(
            raw_values,
            errors="coerce",
        )

        invalid = raw_values.notna() & parsed.isna()

        if invalid.any():
            raise ValueError(
                f"{raw_col} 存在无法解析的日期"
            )

        df[new_col] = parsed

    # 类型和状态转换
    for col in ["type", "status"]:

        values = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(f"{col} 存在无效值")

        if not values.isin([0, 1, 2, 3, 4, 5]).all():
            raise ValueError(f"{col} 存在异常取值")

        df[col] = values.astype("Int64")

    # 上市日期必须存在
    if df["ipo_date"].isna().any():
        raise ValueError("存在缺少上市日期的证券")

    # 退市日期不能早于上市日期
    invalid_date = (
        df["out_date"].notna()
        & (df["out_date"] < df["ipo_date"])
    )

    if invalid_date.any():
        raise ValueError("存在退市日期早于上市日期的证券")

    # 股票代码唯一性
    if df["code"].duplicated().any():
        raise ValueError("股票基本资料存在重复代码")

    df = df[
        [
            "code",
            "code_name",
            "ipo_date",
            "out_date",
            "type",
            "status",
        ]
    ]

    return df.sort_values("code").reset_index(drop=True)


# ==================================================
# 2. 筛选沪深 A 股普通股票
# ==================================================

def filter_a_share_stocks(stock_basic):

    prefixes = config.A_SHARE_CODE_PREFIXES

    mask = (
        stock_basic["type"].eq(1)
        & stock_basic["code"].str.startswith(prefixes)
    )

    stocks = stock_basic.loc[mask].copy()

    return stocks.sort_values(
        "code"
    ).reset_index(drop=True)


# ==================================================
# 3. 构建历史股票池
# ==================================================

def build_historical_universe(
    stock_basic,
    calendar_df,
):
    """
    根据上市日期、退市日期和交易日历，
    构建 date-code 形式的历史股票池。

    注意：
    不根据当前 status 筛选股票。
    不在这里过滤 ST 和停牌股票。
    """

    stocks = filter_a_share_stocks(stock_basic)

    trading_dates = get_trading_dates(calendar_df)

    if len(trading_dates) == 0:
        raise ValueError("交易日历中没有交易日")

    start_date = trading_dates.min()
    end_date = trading_dates.max()

    # 先排除完全不在研究区间内的股票
    candidates = stocks[
        (stocks["ipo_date"] <= end_date)
        & (
            stocks["out_date"].isna()
            | (stocks["out_date"] >= start_date)
        )
    ].copy()

    # 股票与交易日期做笛卡尔积
    date_df = pd.DataFrame({
        "date": trading_dates
    })

    candidates["_key"] = 1
    date_df["_key"] = 1

    universe = candidates.merge(
        date_df,
        on="_key",
        how="inner",
    )

    universe = universe.drop(columns="_key")

    # 保留股票实际上市期间的记录
    mask = (
        (universe["date"] >= universe["ipo_date"])
        & (
            universe["out_date"].isna()
            | (universe["date"] <= universe["out_date"])
        )
    )

    universe = universe.loc[mask].copy()

    universe = universe[
        [
            "date",
            "code",
            "code_name",
            "ipo_date",
            "out_date",
        ]
    ]

    return (
        universe.sort_values(["date", "code"])
        .reset_index(drop=True)
    )


# ==================================================
# 4. 获取研究区间涉及的全部股票代码
# ==================================================

def get_universe_codes(universe_df):

    return sorted(
        universe_df["code"].unique().tolist()
    )
