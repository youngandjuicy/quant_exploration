
from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant import config
from quant.data.panel import FLOAT_COLUMNS, INTEGER_COLUMNS
from quant.data.calendar import (
    clean_trade_calendar,
    get_required_data_range,
)


# ==================================================
# 1. 路径与参数
# ==================================================

DATASET_PATH = (
    config.DATA_PROCESSED_DIR
    / "daily_panel.parquet"
)

UNIVERSE_PATH = (
    config.DATA_INTERIM_DIR
    / "universe_manifest.csv"
)

CALENDAR_PATH = (
    config.DATA_RAW_DIR
    / "baostock"
    / "metadata"
    / "trade_calendar.csv"
)

BATCH_SIZE = 100_000


# ==================================================
# 2. 读取预期股票池和日期范围
# ==================================================

def load_expected_universe():

    if not UNIVERSE_PATH.exists():
        raise FileNotFoundError(
            f"缺少历史股票清单：{UNIVERSE_PATH}"
        )

    df = pd.read_csv(
        UNIVERSE_PATH,
        dtype=str,
        keep_default_na=False,
    )

    if "code" not in df.columns or df.empty:
        raise ValueError("历史股票清单无效")

    codes = df["code"].str.strip()

    if codes.eq("").any():
        raise ValueError("历史股票清单存在空代码")

    if codes.duplicated().any():
        raise ValueError("历史股票清单存在重复代码")

    return set(codes)


def load_expected_date_range():

    if not CALENDAR_PATH.exists():
        raise FileNotFoundError(
            f"缺少交易日历：{CALENDAR_PATH}"
        )

    raw_calendar = pd.read_csv(
        CALENDAR_PATH,
        dtype=str,
        keep_default_na=False,
    )

    calendar = clean_trade_calendar(
        raw_calendar
    )

    return get_required_data_range(
        calendar_df=calendar,
        research_start=config.RESEARCH_START_DATE,
        research_end=config.RESEARCH_END_DATE,
        lookback=config.MOMENTUM_WINDOW,
        forward_horizon=config.FORWARD_HORIZON,
    )


# ==================================================
# 3. 检查 Parquet Schema
# ==================================================

def validate_schema(schema):

    expected_columns = (
        config.DAILY_FIELDS.split(",")
    )

    actual_columns = schema.names

    errors = []

    if actual_columns != expected_columns:
        errors.append(
            f"字段不一致。预期：{expected_columns}；"
            f"实际：{actual_columns}"
        )

    if not set(expected_columns).issubset(
        actual_columns
    ):
        return errors + ["缺少必要字段，无法继续完整审计"]

    field_types = {
        field.name: field.type
        for field in schema
    }

    if not pa.types.is_timestamp(
        field_types["date"]
    ):
        errors.append("date 不是 Timestamp 类型")

    if not (
        pa.types.is_string(field_types["code"])
        or pa.types.is_large_string(field_types["code"])
    ):
        errors.append("code 不是字符串类型")

    for col in FLOAT_COLUMNS:
        if not pa.types.is_floating(
            field_types[col]
        ):
            errors.append(
                f"{col} 应为浮点类型"
            )

    for col in INTEGER_COLUMNS:
        if not pa.types.is_integer(
            field_types[col]
        ):
            errors.append(
                f"{col} 应为整数类型"
            )

    return errors


# ==================================================
# 4. 非排序情况下的精确重复检查
# ==================================================

def count_duplicates_sqlite():

    """
    使用 SQLite 磁盘索引进行精确重复检查。

    仅当 Parquet 全局排序不符合预期时使用，
    避免将全部股票-日期组合放入内存。
    """

    duplicate_count = 0

    with tempfile.TemporaryDirectory(
        prefix="quant_dataset_audit_"
    ) as temp_dir:

        db_path = Path(temp_dir) / "keys.sqlite"

        conn = sqlite3.connect(db_path)

        try:

            conn.execute(
                "PRAGMA journal_mode=OFF"
            )

            conn.execute(
                "PRAGMA synchronous=OFF"
            )

            conn.execute("""
                CREATE TABLE records (
                    date INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    PRIMARY KEY (date, code)
                ) WITHOUT ROWID
            """)

            parquet_file = pq.ParquetFile(
                DATASET_PATH
            )

            for batch in parquet_file.iter_batches(
                batch_size=BATCH_SIZE,
                columns=["date", "code"],
            ):

                df = batch.to_pandas()

                dates = pd.to_datetime(
                    df["date"],
                    errors="coerce",
                )

                codes = df["code"].astype("string")

                valid = (
                    dates.notna()
                    & codes.notna()
                    & codes.ne("")
                )

                date_values = (
                    dates.loc[valid]
                    .astype("int64")
                    .tolist()
                )

                code_values = (
                    codes.loc[valid]
                    .tolist()
                )

                rows = list(
                    zip(date_values, code_values)
                )

                before = conn.total_changes

                conn.executemany(
                    """
                    INSERT OR IGNORE INTO records
                    (date, code)
                    VALUES (?, ?)
                    """,
                    rows,
                )

                inserted = (
                    conn.total_changes - before
                )

                duplicate_count += (
                    len(rows) - inserted
                )

                conn.commit()

        finally:
            conn.close()

    return duplicate_count


# ==================================================
# 5. 主审计程序
# ==================================================

def audit_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"数据集不存在：{DATASET_PATH}"
        )

    expected_codes = load_expected_universe()
    expected_range = load_expected_date_range()

    parquet_file = pq.ParquetFile(
        DATASET_PATH
    )

    schema = parquet_file.schema_arrow

    schema_errors = validate_schema(schema)

    required_columns = (
        config.DAILY_FIELDS.split(",")
    )

    missing_columns = (
        set(required_columns) - set(schema.names)
    )

    if missing_columns:
        raise ValueError(
            f"Parquet 缺少必要字段：{missing_columns}"
        )

    print("\n========== Parquet 基本信息 ==========")

    print("文件：", DATASET_PATH)
    print("文件大小：", round(
        DATASET_PATH.stat().st_size / 1024**2, 2
    ), "MB")

    print("元数据行数：", parquet_file.metadata.num_rows)
    print("Row Groups：", parquet_file.metadata.num_row_groups)

    # ------------------------------------------
    # 统计变量
    # ------------------------------------------

    total_rows = 0

    null_counts = {
        col: 0 for col in schema.names
    }

    stock_codes = set()

    min_date = None
    max_date = None

    invalid_key_rows = 0
    invalid_status = {
        "tradestatus": 0,
        "isST": 0,
    }

    invalid_ohlc = 0
    negative_values = {
        "volume": 0,
        "amount": 0,
    }

    order_violations = 0
    adjacent_duplicates = 0

    previous_key = None

    # ------------------------------------------
    # 分批读取
    # ------------------------------------------

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
    ):

        df = batch.to_pandas()

        total_rows += len(df)

        # 缺失值
        for col in schema.names:
            null_counts[col] += int(
                df[col].isna().sum()
            )

        # 日期和代码
        dates = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        codes = df["code"].astype("string")

        invalid_keys = (
            dates.isna()
            | codes.isna()
            | codes.str.strip().eq("")
        )

        invalid_key_rows += int(
            invalid_keys.sum()
        )

        stock_codes.update(
            codes.dropna().unique().tolist()
        )

        valid_dates = dates.dropna()

        if not valid_dates.empty:

            batch_min = valid_dates.min()
            batch_max = valid_dates.max()

            if min_date is None or batch_min < min_date:
                min_date = batch_min

            if max_date is None or batch_max > max_date:
                max_date = batch_max

        # 状态字段
        for col in ["tradestatus", "isST"]:

            invalid = (
                df[col].notna()
                & ~df[col].isin([0, 1])
            )

            invalid_status[col] += int(
                invalid.sum()
            )

        # OHLC 关系
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]

        bad_ohlc = (
            (high.notna() & open_.notna() & (high < open_))
            | (high.notna() & close.notna() & (high < close))
            | (low.notna() & open_.notna() & (low > open_))
            | (low.notna() & close.notna() & (low > close))
            | (high.notna() & low.notna() & (high < low))
        )

        invalid_ohlc += int(
            bad_ohlc.sum()
        )

        # 成交量、成交额不能为负
        for col in ["volume", "amount"]:

            negative_values[col] += int(
                (df[col] < 0).fillna(False).sum()
            )

        # --------------------------------------
        # 检查全局排序及相邻重复
        # --------------------------------------

        same_code = (
            codes.eq(codes.shift())
            .fillna(False)
        )

        same_date = (
            dates.eq(dates.shift())
            .fillna(False)
        )

        code_decrease = (
            codes.lt(codes.shift())
            .fillna(False)
        )

        date_decrease = (
            dates.lt(dates.shift())
            .fillna(False)
        )

        order_violations += int(
            (
                code_decrease
                | (same_code & date_decrease)
            ).sum()
        )

        adjacent_duplicates += int(
            (same_code & same_date).sum()
        )

        # 检查批次边界
        if len(df) > 0:

            first_code = codes.iloc[0]
            first_date = dates.iloc[0]

            if (
                previous_key is not None
                and pd.notna(first_code)
                and pd.notna(first_date)
            ):

                current_key = (
                    str(first_code),
                    pd.Timestamp(first_date),
                )

                if current_key < previous_key:
                    order_violations += 1

                elif current_key == previous_key:
                    adjacent_duplicates += 1

            last_code = codes.iloc[-1]
            last_date = dates.iloc[-1]

            if (
                pd.notna(last_code)
                and pd.notna(last_date)
            ):

                previous_key = (
                    str(last_code),
                    pd.Timestamp(last_date),
                )

    # ==================================================
    # 6. 整体结果检查
    # ==================================================

    hard_failures = []
    warnings = []

    # Schema
    hard_failures.extend(schema_errors)

    # 行数
    if total_rows != parquet_file.metadata.num_rows:
        hard_failures.append(
            "扫描行数与 Parquet 元数据行数不一致"
        )

    # 主键
    if invalid_key_rows > 0:
        hard_failures.append(
            f"存在 {invalid_key_rows} 条无效日期或股票代码"
        )

    # 股票池
    missing_codes = expected_codes - stock_codes
    extra_codes = stock_codes - expected_codes

    if missing_codes:
        hard_failures.append(
            f"数据缺少 {len(missing_codes)} 只预期股票"
        )

    if extra_codes:
        hard_failures.append(
            f"数据出现 {len(extra_codes)} 只非预期股票"
        )

    # 日期范围
    expected_start = pd.Timestamp(
        expected_range["data_start_date"]
    )

    expected_end = pd.Timestamp(
        expected_range["data_end_date"]
    )

    if min_date != expected_start:
        hard_failures.append(
            f"实际最早日期 {min_date} "
            f"与预期 {expected_start} 不一致"
        )

    if max_date != expected_end:
        hard_failures.append(
            f"实际最晚日期 {max_date} "
            f"与预期 {expected_end} 不一致"
        )

    # 状态字段
    for col, count in invalid_status.items():

        if count > 0:
            hard_failures.append(
                f"{col} 存在 {count} 条异常取值"
            )

    # OHLC
    if invalid_ohlc > 0:
        hard_failures.append(
            f"存在 {invalid_ohlc} 条 OHLC 关系异常"
        )

    # 非负检查
    for col, count in negative_values.items():

        if count > 0:
            hard_failures.append(
                f"{col} 存在 {count} 条负值"
            )

    # 重复检查
    if order_violations == 0:

        duplicate_count = adjacent_duplicates

    else:

        warnings.append(
            f"发现 {order_violations} 处全局排序异常，"
            "改用 SQLite 进行精确重复检查"
        )

        duplicate_count = count_duplicates_sqlite()

    if duplicate_count > 0:
        hard_failures.append(
            f"发现 {duplicate_count} 条重复的 date-code 记录"
        )

    # 缺失值作为警告报告
    for col, count in null_counts.items():

        if count > 0:
            warnings.append(
                f"{col} 存在 {count} 个缺失值"
            )

    # 排序异常不影响数据唯一性，但需要报告
    if order_violations > 0:
        warnings.append(
            f"全局排序异常数量：{order_violations}"
        )

    # ==================================================
    # 7. 输出验收报告
    # ==================================================

    print("\n========== 数据集验收报告 ==========")

    print("扫描总行数：", total_rows)
    print("唯一股票数量：", len(stock_codes))

    print("实际日期范围：")
    print(min_date, "~", max_date)

    print("\n重复记录数：", duplicate_count)
    print("无效主键记录数：", invalid_key_rows)
    print("全局排序异常数：", order_violations)

    print("\n各字段缺失值：")

    for col, count in null_counts.items():
        print(f"{col}: {count}")

    print("\n状态字段异常：", invalid_status)
    print("OHLC 异常记录：", invalid_ohlc)
    print("负成交量 / 成交额：", negative_values)

    print("\n========== 验收结论 ==========")

    if warnings:

        print("\n警告：")

        for warning in warnings:
            print("-", warning)

    if hard_failures:

        print("\n<数据验收未通过>")

        for failure in hard_failures:
            print("-", failure)

        return False

    print("所有核心检查均通过。")

    return True


if __name__ == "__main__":

    success = audit_dataset()

    if not success:
        raise SystemExit(1)
