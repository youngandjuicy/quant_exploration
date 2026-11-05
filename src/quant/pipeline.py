
import pandas as pd

from .data import baostock_client, cache, panel


def build_daily_panel(
    codes,
    start_date,
    end_date,
    force_refresh=False,
    cache_dir=None,
):
    """
    获取指定股票的日线数据，并构建标准行情面板。

    Parameters
    ----------
    codes : list
        股票代码列表。

    start_date : str
        开始日期。

    end_date : str
        结束日期。

    force_refresh : bool
        是否强制重新下载，默认 False。

    cache_dir : str or Path or None
        可选的缓存目录，主要用于测试。

    Returns
    -------
    df : pd.DataFrame
        清洗后的行情数据。

    report : dict
        数据质量报告。
    """

    codes = list(codes)

    if not codes:
        raise ValueError("股票代码列表不能为空")

    if len(codes) != len(set(codes)):
        raise ValueError("股票代码列表存在重复")

    # 保存每只股票的原始数据
    raw_data = {}

    # 记录需要下载的股票
    missing_codes = []

    # ------------------------------------------
    # 1. 检查本地缓存
    # ------------------------------------------

    for code in codes:

        cached_df = None

        if not force_refresh:
            cached_df = cache.load_daily_cache(
                code,
                start_date,
                end_date,
                cache_dir=cache_dir,
            )

        if cached_df is not None:
            print(f"缓存命中：{code}")
            raw_data[code] = cached_df

        else:
            print(f"缓存未命中：{code}")
            missing_codes.append(code)

    # ------------------------------------------
    # 2. 下载缺失的数据
    # ------------------------------------------

    if missing_codes:

        baostock_client.login()

        try:
            for code in missing_codes:

                print(f"正在下载：{code}")

                raw_df = baostock_client.query_daily(
                    code,
                    start_date,
                    end_date,
                )

                # 保存原始数据
                cache.save_daily_cache(
                    code,
                    start_date,
                    end_date,
                    raw_df,
                    cache_dir=cache_dir,
                )

                raw_data[code] = raw_df

        finally:
            baostock_client.logout()

    # ------------------------------------------
    # 3. 合并所有股票数据
    # ------------------------------------------

    raw_panel = pd.concat(
        [raw_data[code] for code in codes],
        ignore_index=True,
    )

    # ------------------------------------------
    # 4. 清洗数据
    # ------------------------------------------

    daily_panel = panel.clean_daily_panel(
        raw_panel
    )

    # ------------------------------------------
    # 5. 数据质量检查
    # ------------------------------------------

    report = panel.validate_daily_panel(
        daily_panel
    )

    return daily_panel, report
