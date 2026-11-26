
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
    compute_daily_group_returns,
    summarize_group_returns,
    compute_group_monotonicity,
    compute_long_short_returns,
    summarize_long_short_returns,
)

from quant.research.grouping import assign_quantile_groups


# ==================================================
# 1. 实验配置
# ==================================================

EXPERIMENT_NAME = "momentum_v1"

FACTOR_SPEC = {
    "name": "momentum",
    "function": compute_momentum,
    "params": {
        "window": config.MOMENTUM_WINDOW,
    },
    "lookback": config.MOMENTUM_WINDOW,
    "required_columns": [
        "close",
        "tradestatus",
        "isST",
    ],
}

TARGET_SPEC = {
    "name": "forward_return",
    "function": compute_forward_return,
    "params": {
        "horizon": config.FORWARD_HORIZON,
    },
    "horizon": config.FORWARD_HORIZON,
    "required_columns": [
        "open",
        "close",
        "tradestatus",
    ],
}

MIN_STOCKS = 30
N_GROUPS = 10

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
    / EXPERIMENT_NAME
)


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

        signal_end = split_dates[-horizon - 1]

    return split_dates[0], signal_end


# ==================================================
# 3. 读取研究数据
# ==================================================

def load_research_data(
    lookback,
    forward_horizon,
    required_columns,
):
    """
    根据实验需求读取行情数据。

    自动计算历史回看和未来收益所需的数据范围，
    只读取因子和目标函数需要的行情字段。
    """

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
        lookback=lookback,
        forward_horizon=forward_horizon,
    )

    # 自动合并因子与目标所需字段
    columns = list(dict.fromkeys([
        "date",
        "code",
        *required_columns,
    ]))

    print("读取 Parquet 行情数据...")
    print("读取字段：", columns)

    daily_panel = pd.read_parquet(
        PANEL_PATH,
        columns=columns,
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
    print(
        "实际日期范围：",
        actual_start.date(),
        "~",
        actual_end.date(),
    )
    print(
        "研究所需日期范围：",
        required_start.date(),
        "~",
        required_end.date(),
    )

    return daily_panel, trading_dates


# ==================================================
# 4. 因子与目标构造
# ==================================================

def build_research_variables(
    daily_panel,
    trading_dates,
    factor_spec,
    target_spec,
):
    """
    根据实验配置构造因子和预测目标。

    两个函数均遵循统一接口：

    Factor:
        date, code, factor

    Target:
        date, code, target
    """

    # ------------------------------------------
    # 1. 构造因子
    # ------------------------------------------

    print("\n========== 因子构造 ==========")
    print("因子名称：", factor_spec["name"])

    factor_df = factor_spec["function"](
        daily_panel,
        trading_dates,
        **factor_spec["params"],
    )

    if len(factor_df) != len(daily_panel):
        raise ValueError(
            "因子函数输出行数与行情数据不一致"
        )

    if not {"date", "code", "factor"}.issubset(
        factor_df.columns
    ):
        raise ValueError(
            "因子函数输出不符合统一接口"
        )

    print("因子构造完成")
    print(
        "有效因子数量：",
        factor_df["factor"].notna().sum(),
    )

    # ------------------------------------------
    # 2. 构造预测目标
    # ------------------------------------------

    print("\n========== 预测目标构造 ==========")
    print("目标名称：", target_spec["name"])

    target_df = target_spec["function"](
        daily_panel,
        trading_dates,
        **target_spec["params"],
    )

    if len(target_df) != len(daily_panel):
        raise ValueError(
            "目标函数输出行数与行情数据不一致"
        )

    if not {"date", "code", "target"}.issubset(
        target_df.columns
    ):
        raise ValueError(
            "目标函数输出不符合统一接口"
        )

    print("预测目标构造完成")
    print(
        "有效目标数量：",
        target_df["target"].notna().sum(),
    )

    return factor_df, target_df


# ==================================================
# 5. 因子评价
# ==================================================


def evaluate_factor(
    factor_df,
    target_df,
    trading_dates,
    horizon,
):
    """
    综合评价因子：
    1. Rank IC
    2. 分组收益
    3. 分组收益单调性
    4. 理论多空收益

    所有评价统一使用 Discovery / Validation /
    Final Test 的信号日期边界。
    """

    # ------------------------------------------
    # 1. Rank IC
    # ------------------------------------------

    print("\n========== 每日 Rank IC ==========")

    daily_ic = compute_daily_rank_ic(
        factor_df,
        target_df,
        min_stocks=MIN_STOCKS,
    )

    print("有效 IC 交易日数：",
          daily_ic["rank_ic"].notna().sum())

    # ------------------------------------------
    # 2. 因子分组
    # ------------------------------------------

    print("\n========== 因子分组 ==========")

    group_df = assign_quantile_groups(
        factor_df,
        n_groups=N_GROUPS,
        min_stocks=MIN_STOCKS,
    )

    # 注意：分组只依据因子，不依据未来收益
    daily_group_returns = compute_daily_group_returns(
        group_df,
        target_df,
        n_groups=N_GROUPS,
    )

    del group_df

    # ------------------------------------------
    # 3. 理论多空收益
    # ------------------------------------------

    daily_long_short = compute_long_short_returns(
        daily_group_returns,
        low_group=1,
        high_group=N_GROUPS,
    )

    # ------------------------------------------
    # 4. 按研究区间分别评价
    # ------------------------------------------

    ic_daily_results = []
    ic_summary_results = []

    group_daily_results = []
    group_summary_results = []
    monotonicity_results = []

    long_short_daily_results = []
    long_short_summary_results = []

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
            horizon,
            allow_future_buffer,
        )

        # 三类每日结果统一使用相同信号日期边界
        split_ic = daily_ic.loc[
            daily_ic["date"].between(
                signal_start,
                signal_end,
            )
        ].copy()

        split_groups = daily_group_returns.loc[
            daily_group_returns["date"].between(
                signal_start,
                signal_end,
            )
        ].copy()

        split_long_short = daily_long_short.loc[
            daily_long_short["date"].between(
                signal_start,
                signal_end,
            )
        ].copy()

        # --------------------------------------
        # 4.1 Rank IC 汇总
        # --------------------------------------

        ic_summary = summarize_rank_ic(
            split_ic
        )

        ic_summary.update({
            "split": split_name,
            "signal_start": signal_start,
            "signal_end": signal_end,
            "n_signal_days": len(split_ic),
        })

        split_ic.insert(0, "split", split_name)

        ic_daily_results.append(split_ic)
        ic_summary_results.append(ic_summary)

        # --------------------------------------
        # 4.2 分组收益汇总
        # --------------------------------------

        group_summary = summarize_group_returns(
            split_groups
        )

        group_summary.insert(
            0,
            "split",
            split_name,
        )

        group_summary["signal_start"] = signal_start
        group_summary["signal_end"] = signal_end

        # 分组收益单调性
        monotonicity = compute_group_monotonicity(
            group_summary
        )

        monotonicity.update({
            "split": split_name,
            "signal_start": signal_start,
            "signal_end": signal_end,
            "n_signal_days": len(split_ic),
        })

        split_groups.insert(
            0,
            "split",
            split_name,
        )

        group_daily_results.append(split_groups)
        group_summary_results.append(group_summary)
        monotonicity_results.append(monotonicity)

        # --------------------------------------
        # 4.3 理论多空收益汇总
        # --------------------------------------

        long_short_summary = summarize_long_short_returns(
            split_long_short
        )

        long_short_summary.update({
            "split": split_name,
            "signal_start": signal_start,
            "signal_end": signal_end,
            "n_signal_days": len(split_ic),
        })

        split_long_short.insert(
            0,
            "split",
            split_name,
        )

        long_short_daily_results.append(split_long_short)
        long_short_summary_results.append(
            long_short_summary
        )

        # --------------------------------------
        # 4.4 输出区间摘要
        # --------------------------------------

        print(f"\n========== {split_name} ==========")

        print(
            "信号区间：",
            signal_start.date(),
            "~",
            signal_end.date(),
        )

        print("Mean IC：", ic_summary["mean_ic"])
        print("ICIR：", ic_summary["icir"])

        print(
            "Group Return Spearman：",
            monotonicity["group_return_spearman"],
        )

        print(
            "Mean Long-Short：",
            long_short_summary["mean_long_short"],
        )

    # ------------------------------------------
    # 5. 整理最终结果
    # ------------------------------------------

    return {
        "daily_rank_ic": pd.concat(
            ic_daily_results,
            ignore_index=True,
        ),

        "rank_ic_summary": pd.DataFrame(
            ic_summary_results
        ),

        "daily_group_returns": pd.concat(
            group_daily_results,
            ignore_index=True,
        ),

        "group_summary": pd.concat(
            group_summary_results,
            ignore_index=True,
        ),

        "group_monotonicity": pd.DataFrame(
            monotonicity_results
        ),

        "daily_long_short_returns": pd.concat(
            long_short_daily_results,
            ignore_index=True,
        ),

        "long_short_summary": pd.DataFrame(
            long_short_summary_results
        ),
    }


# ==================================================
# 6. 保存研究结果
# ==================================================


def save_results(results):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_files = {
        "daily_rank_ic.csv":
            results["daily_rank_ic"],

        "summary.csv":
            results["rank_ic_summary"],

        "daily_group_returns.csv":
            results["daily_group_returns"],

        "group_summary.csv":
            results["group_summary"],

        "group_monotonicity.csv":
            results["group_monotonicity"],

        "daily_long_short_returns.csv":
            results["daily_long_short_returns"],

        "long_short_summary.csv":
            results["long_short_summary"],
    }

    print("\n========== 研究结果已保存 ==========")

    for filename, df in output_files.items():

        path = RESULTS_DIR / filename

        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
            date_format="%Y-%m-%d",
        )

        print(filename, ":", path)


# ==================================================
# 7. 主程序
# ==================================================

def main():

    print("======================================")
    print("       Quant Research - Experiment")
    print("======================================")

    print("实验名称：", EXPERIMENT_NAME)
    print("研究区间：",
          config.RESEARCH_START_DATE,
          "~",
          config.RESEARCH_END_DATE)

    print("因子：", FACTOR_SPEC["name"])
    print("因子参数：", FACTOR_SPEC["params"])
    print("历史回看：", FACTOR_SPEC["lookback"])

    print("预测目标：", TARGET_SPEC["name"])
    print("目标参数：", TARGET_SPEC["params"])

    # 1. 读取研究数据
    required_columns = (
        FACTOR_SPEC["required_columns"]
        + TARGET_SPEC["required_columns"]
    )

    daily_panel, trading_dates = load_research_data(
        lookback=FACTOR_SPEC["lookback"],
        forward_horizon=TARGET_SPEC["horizon"],
        required_columns=required_columns,
    )

    # 2. 构造因子和预测目标
    factor_df, target_df = build_research_variables(
        daily_panel,
        trading_dates,
        FACTOR_SPEC,
        TARGET_SPEC,
    )

    # 原始行情不再需要
    del daily_panel

    # 3. 计算并评价每日 Rank IC
    results = evaluate_factor(
        factor_df,
        target_df,
        trading_dates,
        horizon=TARGET_SPEC["horizon"],
    )

    # 释放大型中间数据
    del factor_df
    del target_df

    # 4. 保存结果
    save_results(results)

    print("\n========== Research Completed ==========")


if __name__ == "__main__":
    main()
