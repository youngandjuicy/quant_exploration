
import pandas as pd

from .. import config


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
