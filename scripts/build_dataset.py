
import argparse
from pathlib import Path

import pandas as pd

from quant import config
from quant.data.calendar import (
    clean_trade_calendar,
    get_required_data_range,
)
from quant.data.universe import (
    clean_stock_basic,
    filter_a_share_stocks,
)
from quant.pipeline import build_daily_panel_to_parquet


# ==================================================
# 1. 固定构建参数
# ==================================================

BATCH_SIZE = 100

METADATA_DIR = (
    config.DATA_RAW_DIR
    / "baostock"
    / "metadata"
)

STOCK_BASIC_PATH = (
    METADATA_DIR / "stock_basic.csv"
)

TRADE_CALENDAR_PATH = (
    METADATA_DIR / "trade_calendar.csv"
)

UNIVERSE_OUTPUT_PATH = (
    config.DATA_INTERIM_DIR
    / "universe_manifest.csv"
)

DATASET_OUTPUT_PATH = (
    config.DATA_PROCESSED_DIR
    / "daily_panel.parquet"
)


# ==================================================
# 2. 读取历史元数据
# ==================================================

def load_metadata():

    required_files = [
        STOCK_BASIC_PATH,
        TRADE_CALENDAR_PATH,
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"缺少必要的历史元数据：{path}\n"
                "请先运行 scripts/download_metadata.py"
            )

    raw_stock_basic = pd.read_csv(
        STOCK_BASIC_PATH,
        dtype=str,
        keep_default_na=False,
    )

    raw_trade_calendar = pd.read_csv(
        TRADE_CALENDAR_PATH,
        dtype=str,
        keep_default_na=False,
    )

    if raw_stock_basic.empty:
        raise ValueError("股票基本资料为空")

    if raw_trade_calendar.empty:
        raise ValueError("交易日历为空")

    stock_basic = clean_stock_basic(
        raw_stock_basic
    )

    calendar = clean_trade_calendar(
        raw_trade_calendar
    )

    return stock_basic, calendar


# ==================================================
# 3. 计算行情数据覆盖范围
# ==================================================

def get_data_range(calendar):

    result = get_required_data_range(
        calendar_df=calendar,
        research_start=config.RESEARCH_START_DATE,
        research_end=config.RESEARCH_END_DATE,
        lookback=config.MOMENTUM_WINDOW,
        forward_horizon=config.FORWARD_HORIZON,
    )

    return result


# ==================================================
# 4. 构建历史股票下载清单
# ==================================================

def build_download_universe(
    stock_basic,
    research_start_date,
    research_end_date,
):

    # 首先筛选沪深 A 股普通股票
    a_stocks = filter_a_share_stocks(
        stock_basic
    )

    research_start = pd.Timestamp(
        research_start_date
    )

    research_end = pd.Timestamp(
        research_end_date
    )

    # 股票上市期间必须与正式研究区间有交集
    mask = (
        (a_stocks["ipo_date"] <= research_end)
        & (
            a_stocks["out_date"].isna()
            | (a_stocks["out_date"] >= research_start)
        )
    )

    candidates = a_stocks.loc[
        mask
    ].copy()

    if candidates.empty:
        raise ValueError(
            "研究区间内没有符合条件的 A 股"
        )

    candidates = (
        candidates
        .sort_values("code")
        .reset_index(drop=True)
    )

    codes = candidates["code"].tolist()

    if len(codes) != len(set(codes)):
        raise ValueError(
            "历史股票下载清单存在重复代码"
        )

    return candidates, codes


# ==================================================
# 5. 保存历史股票下载清单
# ==================================================

def save_universe(candidates):

    UNIVERSE_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = candidates[
        [
            "code",
            "code_name",
            "ipo_date",
            "out_date",
        ]
    ].copy()

    output.to_csv(
        UNIVERSE_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"\n历史股票清单已保存："
        f"{UNIVERSE_OUTPUT_PATH}"
    )


# ==================================================
# 6. 正式数据构建
# ==================================================

def build_dataset(codes, data_range):

    print("\n========== 开始正式数据构建 ==========")

    report = build_daily_panel_to_parquet(
        codes=codes,
        start_date=data_range["data_start_date"],
        end_date=data_range["data_end_date"],
        output_path=DATASET_OUTPUT_PATH,
        batch_size=BATCH_SIZE,
        force_refresh=False,
    )

    return report


# ==================================================
# 7. 主程序
# ==================================================

def main():

    parser = argparse.ArgumentParser(
        description="构建沪深 A 股历史日线数据集"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="正式下载并构建完整数据集",
    )

    args = parser.parse_args()

    # ------------------------------------------
    # 1. 加载元数据
    # ------------------------------------------

    print("读取历史元数据...")

    stock_basic, calendar = load_metadata()

    # ------------------------------------------
    # 2. 计算行情覆盖范围
    # ------------------------------------------

    data_range = get_data_range(
        calendar
    )

    # ------------------------------------------
    # 3. 构建历史股票清单
    # ------------------------------------------

    candidates, codes = build_download_universe(
        stock_basic=stock_basic,
        research_start_date=config.RESEARCH_START_DATE,
        research_end_date=config.RESEARCH_END_DATE,
    )

    # ------------------------------------------
    # 4. 输出预检查报告
    # ------------------------------------------

    delisted_during_period = (
        candidates["out_date"].notna()
        & (candidates["out_date"] >= pd.Timestamp(
            data_range["data_start_date"]
        ))
        & (candidates["out_date"] <= pd.Timestamp(
            data_range["data_end_date"]
        ))
    ).sum()

    print("\n========== 数据构建预检查 ==========")

    print(
        "研究区间：",
        config.RESEARCH_START_DATE,
        "~",
        config.RESEARCH_END_DATE,
    )

    print(
        "研究首个交易日：",
        data_range["research_first_trading_date"],
    )

    print(
        "研究最后交易日：",
        data_range["research_last_trading_date"],
    )

    print(
        "行情下载开始日期：",
        data_range["data_start_date"],
    )

    print(
        "行情下载结束日期：",
        data_range["data_end_date"],
    )

    print(
        "沪深 A 股普通股票总数：",
        len(filter_a_share_stocks(stock_basic)),
    )

    print(
        "符合历史覆盖条件的股票数：",
        len(codes),
    )

    print(
        "覆盖期内有记录退市日期的股票数：",
        int(delisted_during_period),
    )

    print(
        "预计批次数：",
        (len(codes) + BATCH_SIZE - 1) // BATCH_SIZE,
    )

    print(
        "批次大小：",
        BATCH_SIZE,
    )

    print(
        "复权方式：",
        config.DAILY_ADJUST_FLAG,
    )

    print(
        "最终数据文件：",
        DATASET_OUTPUT_PATH,
    )

    # ------------------------------------------
    # 5. 决定是否真正下载
    # ------------------------------------------

    if not args.build:

        print("\n当前为预检查模式，尚未下载行情。")

        print(
            "确认上述股票数量和日期范围后，"
            "运行 python scripts/build_dataset.py --build"
        )

        return

    # ------------------------------------------
    # 6. 正式构建
    # ------------------------------------------

    report = build_dataset(
        codes=codes,
        data_range=data_range,
    )

    # 正式数据集构建成功后，保存对应股票池清单
    save_universe(candidates)

    print("\n========== 最终构建报告 ==========")

    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
