
import pandas as pd

from .. import config


# ==================================================
# 1. 字段分类
# ==================================================

FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "amount",
    "turn",
    "pctChg",
]

INTEGER_COLUMNS = [
    "volume",
    "adjustflag",
    "tradestatus",
    "isST",
]

NUMERIC_COLUMNS = FLOAT_COLUMNS + INTEGER_COLUMNS


# ==================================================
# 2. 检查原始数据结构
# ==================================================

def check_schema(df):
    """
    检查 DataFrame 是否包含所需字段。
    """

    required_columns = config.DAILY_FIELDS.split(",")

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"缺少必要字段：{missing_columns}"
        )


# ==================================================
# 3. 清洗日线数据
# ==================================================

def clean_daily_panel(raw_df):
    """
    将 BaoStock 原始日线数据转换为标准 DataFrame。

    主要操作：
    1. 检查字段
    2. 转换日期类型
    3. 转换数值类型
    4. 按日期、股票代码排序

    不进行股票筛选，不填充缺失值。
    """

    check_schema(raw_df)

    df = raw_df.copy()

    # 日期转换
    raw_date = df["date"].astype("string").str.strip()

    df["date"] = pd.to_datetime(
        raw_date,
        errors="coerce",
    )

    invalid_date = (
        raw_date.notna()
        & raw_date.ne("")
        & df["date"].isna()
    )

    if invalid_date.any():
        raise ValueError("发现无法解析的日期")

    # 股票代码转换
    df["code"] = (
        df["code"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # 数值字段转换
    for col in FLOAT_COLUMNS + INTEGER_COLUMNS:

        raw_values = (
            df[col]
            .astype("string")
            .str.strip()
        )

        # 空字符串转换成缺失值
        raw_values = raw_values.replace("", pd.NA)

        numeric_values = pd.to_numeric(
            raw_values,
            errors="coerce",
        )

        # 检查非空字符串是否转换失败
        invalid = (
            raw_values.notna()
            & numeric_values.isna()
        )

        if invalid.any():
            raise ValueError(
                f"字段 {col} 存在无法转换的数值"
            )

        # 整数列检查
        if col in INTEGER_COLUMNS:

            fractional = (
                numeric_values.notna()
                & (numeric_values % 1 != 0)
            )

            if fractional.any():
                raise ValueError(
                    f"字段 {col} 存在非整数值"
                )

            df[col] = numeric_values.astype("Int64")

        else:
            df[col] = numeric_values.astype("float64")

    # 排序并重置索引
    df = (
        df.sort_values(["date", "code"])
        .reset_index(drop=True)
    )

    return df


# ==================================================
# 4. 数据质量检查
# ==================================================

def validate_daily_panel(df):
    """
    检查标准日线数据的基本完整性。

    返回数据质量报告。
    """

    check_schema(df)

    if df.empty:
        raise ValueError("日线数据为空")

    # 检查日期
    if df["date"].isna().any():
        raise ValueError("存在缺失或无效日期")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("date 必须是日期时间类型")

    # 检查股票代码
    if df["code"].isna().any():
        raise ValueError("存在缺失股票代码")

    if df["code"].eq("").any():
        raise ValueError("存在空股票代码")

    # 检查重复的股票-日期组合
    duplicate_count = df.duplicated(
        subset=["date", "code"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"发现 {duplicate_count} 条重复的股票-日期记录"
        )

    # 检查数值字段类型
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(
                f"字段 {col} 不是数值类型"
            )

    # 检查状态字段取值
    for col in ["tradestatus", "isST"]:

        invalid_values = set(
            df[col].dropna().unique()
        ) - {0, 1}

        if invalid_values:
            raise ValueError(
                f"字段 {col} 存在异常取值：{invalid_values}"
            )

    # 生成数据质量报告
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_stocks": df["code"].nunique(),
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "duplicate_count": int(duplicate_count),
        "missing_values": df.isna().sum().to_dict(),
        "tradestatus_counts": (
            df["tradestatus"]
            .value_counts(dropna=False)
            .to_dict()
        ),
        "isST_counts": (
            df["isST"]
            .value_counts(dropna=False)
            .to_dict()
        ),
    }

    return report
