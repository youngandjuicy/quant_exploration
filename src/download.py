
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
    login_result = bs.login()

    if login_result.error_code != "0":
        raise RuntimeError(
            f"登录失败：{login_result.error_msg}"
        )

    try:
        all_frames = []

        for code in STOCK_CODES:
            print(f"正在下载：{code}")

            stock_df = download_daily(
                code, START_DATE, END_DATE
            )

            print(f"下载记录数：{len(stock_df)}")

            all_frames.append(stock_df)

        # 将 5 只股票的数据纵向合并
        df = pd.concat(
            all_frames,
            ignore_index=True,
        )

        # 按日期、股票代码排序
        df = df.sort_values(
            ["date", "code"]
        ).reset_index(drop=True)

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



    finally:
        # 即使下载过程中发生异常，也尝试退出
        bs.logout()


if __name__ == "__main__":
    main()
