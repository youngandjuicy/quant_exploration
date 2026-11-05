
from quant import config
from quant.pipeline import build_daily_panel


def main():

    # 构建五只股票的 smoke test 数据
    df, report = build_daily_panel(
        codes=config.SMOKE_TEST_CODES,
        start_date=config.SMOKE_TEST_START_DATE,
        end_date=config.SMOKE_TEST_END_DATE,
    )

    # 打印数据质量报告
    print("\n========== Data Quality Report ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    # 保存清洗后的数据
    output_path = (
        config.DATA_INTERIM_DIR
        / "smoke_test_daily_panel.csv"
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

    print(f"\n数据已保存：{output_path}")


if __name__ == "__main__":
    main()
