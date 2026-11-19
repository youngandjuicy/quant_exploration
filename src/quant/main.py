
from pathlib import Path

import pandas as pd

from quant import config
from quant.data.calendar import (
    clean_trade_calendar,
    get_trading_dates,
    get_required_data_range,
)
from quant.research.factors import compute_momentum
from quant.research.labels import compute_forward_return
from quant.research.evaluation import (
    compute_daily_rank_ic,
    summarize_rank_ic,
)


# ==================================================
# 1. 路径与研究参数
# ==================================================

PANEL_PATH = (
    config.DATA_PROCESSED_DIR
    / "daily_panel.parquet"
)

CALENDAR_PATH = (
    config.DATA_RAW_DIR
    / "baostock"
    / "metadata"
    / "trade_calendar.csv"
)

RESULTS_DIR = (
    config.PROJECT_ROOT
    / "results"
    / "momentum_v1"
)

MIN_STOCKS = 30


# ==================================================
# 2. 时间区间划分
# ==================================================

SPLITS = [
    (
        "Discovery",
        config.DISCOVERY_START_DATE,
        config.DISCOVERY_END_DATE,
        False,
    ),
    (
        "Validation",
        config.VALIDATION_START_DATE,
        config.VALIDATION_END_DATE,
        False,
    ),
    (
        "Final_Test",
        config.TEST_START_DATE,
        config.TEST_END_DATE,
        True,
    ),
]


def get_split_bounds(
    trading_dates,
    start_date,
    end_date,
    horizon,
    allow_future_buffer,
):
    """
    获取一个研究区间内允许参与评价的信号日期。

    Discovery / Validation:
        未来收益必须在当前区间内完成。

    Final Test:
        允许使用研究区间之后的缓冲行情。
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    split_dates = trading_dates[
        (trading_dates >= start)
        & (trading_dates <= end)
    ]

    if len(split_dates) == 0:
        raise ValueError(
            f"研究区间没有交易日：{start_date} ~ {end_date}"
        )

    if allow_future_buffer:
        signal_end = split_dates[-1]

    else:
        if len(split_dates) <= horizon:
            raise ValueError(
                f"研究区间交易日不足：{start_date} ~ {end_date}"
            )

        # 最后 horizon 个信号日不参与评价
        signal_end = split_dates[-horizon - 1]

    return split_dates[0], signal_end


# ==================================================
# 3. 读取研究数据
# ==================================================

def load_research_data():

    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"行情数据不存在：{PANEL_PATH}"
        )

    if not CALENDAR_PATH.exists():
        raise FileNotFoundError(
            f"交易日历不存在：{CALENDAR_PATH}"
        )

    print("\n========== 读取研究数据 ==========")

    print("读取交易日历...")

    raw_calendar = pd.read_csv(
        CALENDAR_PATH,
        dtype=str,
    )

    calendar_df = clean_trade_calendar(
        raw_calendar
    )

    trading_dates = get_trading_dates(
        calendar_df
    )

    # 计算理论上需要的完整数据范围
    required_range = get_required_data_range(
        calendar_df,
        config.RESEARCH_START_DATE,
        config.RESEARCH_END_DATE,
        lookback=config.MOMENTUM_WINDOW,
        forward_horizon=config.FORWARD_HORIZON,
    )

    print("读取 Parquet 行情数据...")

    # 只读取研究所需的六个字段
    required_columns = [
        "date",
        "code",
        "open",
        "close",
        "tradestatus",
        "isST",
    ]

    daily_panel = pd.read_parquet(
        PANEL_PATH,
        columns=required_columns,
    )

    if daily_panel.empty:
        raise ValueError("行情数据为空")

    actual_start = pd.Timestamp(
        daily_panel["date"].min()
    )

    actual_end = pd.Timestamp(
        daily_panel["date"].max()
    )

    required_start = pd.Timestamp(
        required_range["data_start_date"]
    )

    required_end = pd.Timestamp(
        required_range["data_end_date"]
    )

    # 检查行情数据是否覆盖完整研究范围
    if actual_start > required_start:
        raise ValueError(
            f"行情数据缺少历史缓冲："
            f"需要 {required_start.date()}，"
            f"实际从 {actual_start.date()} 开始"
        )

    if actual_end < required_end:
        raise ValueError(
            f"行情数据缺少未来缓冲："
            f"需要 {required_end.date()}，"
            f"实际到 {actual_end.date()} 结束"
        )

    print("\n数据读取完成")
    print("行情记录数：", len(daily_panel))
    print("股票数量：", daily_panel["code"].nunique())
    print("实际日期范围：", actual_start.date(), "~", actual_end.date())

    print(
        "研究所需日期范围：",
        required_start.date(),
        "~",
        required_end.date(),
    )

    return daily_panel, trading_dates


# ==================================================
# 4. 因子与收益标签构造
# ==================================================

def build_research_variables(
    daily_panel,
    trading_dates,
):

    print("\n========== 因子构造 ==========")

    factor_df = compute_momentum(
        daily_panel,
        trading_dates,
        window=config.MOMENTUM_WINDOW,
    )

    print("20日动量计算完成")
    print(
        "有效因子数量：",
        factor_df["mom20"].notna().sum(),
    )

    print("\n========== 收益标签构造 ==========")

    label_df = compute_forward_return(
        daily_panel,
        trading_dates,
        horizon=config.FORWARD_HORIZON,
    )

    print("未来收益计算完成")
    print(
        "有效收益标签数量：",
        label_df["future_return_5d"].notna().sum(),
    )

    return factor_df, label_df


# ==================================================
# 5. 每日 Rank IC 计算
# ==================================================

def evaluate_factor(
    factor_df,
    label_df,
    trading_dates,
):

    print("\n========== 每日 Rank IC ==========")

    daily_ic = compute_daily_rank_ic(
        factor_df,
        label_df,
        min_stocks=MIN_STOCKS,
    )

    print("每日 IC 计算完成")
    print("总交易日数：", len(daily_ic))
    print(
        "有效 IC 交易日数：",
        daily_ic["rank_ic"].notna().sum(),
    )

    # 不再需要完整因子和标签数据
    del factor_df
    del label_df

    # ------------------------------------------
    # 5.1 按研究区间分别评价
    # ------------------------------------------

    daily_results = []
    summary_results = []

    for (
        split_name,
        start_date,
        end_date,
        allow_future_buffer,
    ) in SPLITS:

        signal_start, signal_end = get_split_bounds(
            trading_dates,
            start_date,
            end_date,
            config.FORWARD_HORIZON,
            allow_future_buffer,
        )

        mask = (
            (daily_ic["date"] >= signal_start)
            & (daily_ic["date"] <= signal_end)
        )

        split_ic = daily_ic.loc[mask].copy()

        split_ic.insert(
            0,
            "split",
            split_name,
        )

        # 计算区间汇总指标
        summary = summarize_rank_ic(
            split_ic
        )

        summary.update({
            "split": split_name,
            "signal_start": signal_start,
            "signal_end": signal_end,
            "n_signal_days": len(split_ic),
        })

        daily_results.append(split_ic)
        summary_results.append(summary)

        print(f"\n========== {split_name} ==========")
        print("信号开始日期：", signal_start.date())
        print("信号结束日期：", signal_end.date())

        for key, value in summary.items():
            print(f"{key}: {value}")

    daily_results = pd.concat(
        daily_results,
        ignore_index=True,
    )

    summary_results = pd.DataFrame(
        summary_results
    )

    return daily_results, summary_results


# ==================================================
# 6. 保存研究结果
# ==================================================

def save_results(
    daily_ic,
    summary,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_path = (
        RESULTS_DIR
        / "daily_rank_ic.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "summary.csv"
    )

    daily_ic.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print("\n========== 研究结果已保存 ==========")

    print("每日 Rank IC：", daily_path)
    print("汇总报告：", summary_path)


# ==================================================
# 7. 主程序
# ==================================================

def main():

    print("======================================")
    print("       Quant Research - Momentum V1")
    print("======================================")

    print("研究区间：",
          config.RESEARCH_START_DATE,
          "~",
          config.RESEARCH_END_DATE)

    print("动量窗口：", config.MOMENTUM_WINDOW)
    print("未来收益窗口：", config.FORWARD_HORIZON)

    # 1. 读取数据
    daily_panel, trading_dates = load_research_data()

    # 2. 构造因子和收益标签
    factor_df, label_df = build_research_variables(
        daily_panel,
        trading_dates,
    )

    # 原始行情不再需要
    del daily_panel

    # 3. 计算并评价每日 Rank IC
    daily_ic, summary = evaluate_factor(
        factor_df,
        label_df,
        trading_dates,
    )

    # 4. 保存结果
    save_results(
        daily_ic,
        summary,
    )

    print("\n========== Research Completed ==========")


if __name__ == "__main__":
    main()
