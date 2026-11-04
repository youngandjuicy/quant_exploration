
from pathlib import Path

import baostock as bs
import pandas as pd


# 1. 研究样本：先选 5 只股票
STOCK_CODES = [
    "sh.600000",
    "sh.600519",
    "sh.601318",
    "sz.000001",
    "sz.000858",
]

START_DATE = "2024-01-01"
END_DATE = "2024-06-30"

FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
)

# 是否强制重新下载
# True：忽略本地缓存，重新下载
# False：优先读取本地缓存
FORCE_DOWNLOAD = False

# 2. 下载一只股票的日线数据
def download_daily(code, start_date, end_date):
    rs = bs.query_history_k_data_plus(
        code,
        FIELDS,
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="1",
    )

    # 检查接口是否成功返回
    if rs.error_code != "0":
        raise RuntimeError(
            f"下载失败：{code}, "
            f"错误码：{rs.error_code}, "
            f"错误信息：{rs.error_msg}"
        )

    # 逐行读取接口返回的数据
    rows = []

    while rs.next():
        rows.append(rs.get_row_data())

    # 将数据整理成表格
    df = pd.DataFrame(rows, columns=rs.fields)

    return df


# 3. 主程序  
def main():
    # 原始数据保存路径
    raw_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "sample_daily.csv"
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # 记录是否成功登录 BaoStock
    logged_in = False

    try:
        # 1. 优先检查本地缓存
        if raw_path.exists() and not FORCE_DOWNLOAD:
            print(f"\n发现本地缓存，直接读取：{raw_path}")
            df = pd.read_csv(raw_path)

        else:
            print("\n开始从 BaoStock 下载数据...")

            # 2. 登录 BaoStock
            login_result = bs.login()

            if login_result.error_code != "0":
                raise RuntimeError(
                    f"登录失败：{login_result.error_msg}"
                )

            logged_in = True

            # 3. 下载五只股票
            all_frames = []

            for code in STOCK_CODES:
                print(f"正在下载：{code}")

                stock_df = download_daily(
                    code, START_DATE, END_DATE
                )

                print(f"下载记录数：{len(stock_df)}")
                all_frames.append(stock_df)

            # 4. 合并数据
            df = pd.concat(
                all_frames,
                ignore_index=True,
            )

            # 5. 排序
            df = df.sort_values(
                ["date", "code"]
            ).reset_index(drop=True)

            # 6. 保存本地缓存
            df.to_csv(
                raw_path,
                index=False,
                encoding="utf-8-sig",
            )

            print(f"\n原始数据已保存：{raw_path}")

        # 统一排序，保证读取缓存和下载得到的数据顺序一致
        df = df.sort_values(
            ["date", "code"]
        ).reset_index(drop=True)

        # 从这里开始，继续保留你原来的数据检查代码


        # 检查结果
        print("\n前 10 行数据：")
        print(df.head(10))

        print("\n各列的数据类型：")
        print(df.dtypes)

        print("\n数据规模（行数, 列数）：")
        print(df.shape)

        duplicate_count = df.duplicated(
            subset=["date", "code"]
        ).sum()

        print(f"\n重复的日期-股票组合数：{duplicate_count}")

                
        # 1. 保存原始接口数据，不修改原始 df
        raw_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "raw"
            / "sample_daily.csv"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            raw_path,
            index=False,
            encoding="utf-8-sig",
        )
        print(f"原始数据已保存：{raw_path}")


        # 2. 创建清洗副本
        df_clean = df.copy()

        # 需要转换为数值的列
        numeric_cols = [
            "open", "high", "low", "close", "preclose",
            "volume", "amount", "adjustflag", "turn",
            "tradestatus", "pctChg", "isST",
        ]

        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(
                df_clean[col],
                errors="coerce",
            )

        # 日期字符串转换为日期类型
        df_clean["date"] = pd.to_datetime(
            df_clean["date"],
            errors="coerce",
        )


        # 3. 检查清洗结果
        print("\n清洗后的前 5 行：")
        print(df_clean.head())

        print("\n清洗后的数据类型：")
        print(df_clean.dtypes)

        print("\n数据规模：")
        print(df_clean.shape)

        print("\n各列缺失值数量：")
        print(df_clean.isna().sum())

        print("\n日期范围：")
        print(df_clean["date"].min())
        print(df_clean["date"].max())

        print("\n每只股票的记录数：")
        print(df_clean.groupby("code").size())


        # 4. 保存清洗后的数据
        processed_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "processed"
            / "sample_daily.csv"
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        df_clean.to_csv(
            processed_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"\n清洗后的数据已保存：{processed_path}")

        # 按股票和日期排序
        df_check = df_clean.sort_values(
            ["code", "date"]
        ).copy()

        # 计算每只股票自己的收盘价日收益率
        df_check["return"] = (
            df_check.groupby("code")["close"]
            .pct_change(fill_method=None)
        )

        # 转换成百分比，方便与 pctChg 比较
        df_check["return_pct"] = df_check["return"] * 100

        # 计算两者的差异
        df_check["diff"] = (
            df_check["return_pct"] - df_check["pctChg"]
        )

        # 查看每只股票的前几条记录
        print("\n收益率对比：")
        print(
            df_check.groupby("code")[
                ["date", "close", "return_pct", "pctChg", "diff"]
            ].head(5)
        )

        # 统计差异
        valid_diff = df_check["diff"].dropna()

        print("\n收益率差异统计：")
        print("最大绝对误差：", valid_diff.abs().max())
        print("平均绝对误差：", valid_diff.abs().mean())

        # 检查交易状态
        print("\n交易状态分布：")
        print(df_check["tradestatus"].value_counts(dropna=False))

        print("\nST 状态分布：")
        print(df_check["isST"].value_counts(dropna=False))

        # 按股票和日期排序
        df_factor = df_clean.sort_values(
            ["code", "date"]
        ).copy()

        # 计算20个交易间隔之前的收盘价
        df_factor["close_lag20"] = (
            df_factor.groupby("code")["close"]
            .shift(20)
        )

        # 计算20日动量
        df_factor["mom20"] = (
            df_factor["close"] / df_factor["close_lag20"] - 1
        )

        # 查看结果
        print("\n20日动量：")
        print(
            df_factor[
                ["date", "code", "close", "close_lag20", "mom20"]
            ].groupby("code").head(25)
        )

        # 检查有效因子数量
        print("\n各股票有效动量数量：")
        print(
            df_factor.groupby("code")["mom20"].count()
        )

        # 计算未来5日收益率

        # 下一交易日开盘价
        df_factor["open_next"] = (
            df_factor.groupby("code")["open"]
            .shift(-1)
        )

        # 未来第5个交易日收盘价
        df_factor["close_future5"] = (
            df_factor.groupby("code")["close"]
            .shift(-5)
        )

        # 未来5日收益率
        df_factor["future_return_5d"] = (
            df_factor["close_future5"]
            / df_factor["open_next"] - 1
        )

        # 查看因子与未来收益
        print("\n因子与未来收益：")
        print(
            df_factor[
                [
                    "date",
                    "code",
                    "mom20",
                    "open_next",
                    "close_future5",
                    "future_return_5d",
                ]
            ].groupby("code").head(25)
        )

        # 检查有效数据数量
        print("\n各股票有效未来收益数量：")
        print(
            df_factor.groupby("code")["future_return_5d"].count()
        )

        # 同时拥有因子和未来收益的样本数量
        valid_samples = df_factor.dropna(
            subset=["mom20", "future_return_5d"]
        )

        print("\n有效研究样本数量：")
        print(valid_samples.groupby("code").size())

        # 1. 筛选同时具有因子和未来收益的样本
        valid = df_factor.dropna(
            subset=["mom20", "future_return_5d"]
        ).copy()


        # 2. 按交易日计算 Rank IC
        records = []

        for date, group in valid.groupby("date"):
            ic = group["mom20"].corr(
                group["future_return_5d"],
                method="spearman",
            )

            records.append({
                "date": date,
                "rank_ic": ic,
                "n_stocks": len(group),
            })

        daily_ic = pd.DataFrame(records)


        # 3. 查看每日 Rank IC
        print("\n每日 Rank IC：")
        print(daily_ic.head(10))


        # 4. 汇总统计指标
        ic_mean = daily_ic["rank_ic"].mean()
        ic_std = daily_ic["rank_ic"].std()
        icir = ic_mean / ic_std if ic_std > 0 else float("nan")
        positive_ratio = (daily_ic["rank_ic"] > 0).mean()

        print("\nRank IC 统计：")
        print("有效交易日数量：", daily_ic["rank_ic"].count())
        print("Mean IC：", ic_mean)
        print("IC 标准差：", ic_std)
        print("ICIR：", icir)
        print("IC 正值比例：", positive_ratio)


        # 5. 查看某一天的横截面数据
        first_date = valid["date"].min()

        print("\n第一天的横截面数据：")
        print(
            valid[valid["date"] == first_date][
                ["code", "mom20", "future_return_5d"]
            ].sort_values("mom20", ascending=False)
        )


    


    finally:
        # 即使下载过程中发生异常，也尝试退出
        bs.logout()


if __name__ == "__main__":
    main()
