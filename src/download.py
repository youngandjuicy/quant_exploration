
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

        # 保存到项目目录下的 data/raw
        output_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "raw"
            / "sample_daily.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"\n文件已保存至：{output_path}")

    finally:
        # 即使下载过程中发生异常，也尝试退出
        bs.logout()


if __name__ == "__main__":
    main()
