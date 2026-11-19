
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import baostock_client, cache, panel
from . import config


# ==================================================
# 1. 通用辅助函数
# ==================================================

def _validate_codes(codes):
    """检查股票代码列表。"""

    codes = list(codes)

    if not codes:
        raise ValueError("股票代码列表不能为空")

    if any(
        not isinstance(code, str) or not code.strip()
        for code in codes
    ):
        raise ValueError("股票代码存在空值或非字符串")

    if any(code != code.strip() for code in codes):
        raise ValueError("股票代码存在首尾空格")

    if len(codes) != len(set(codes)):
        raise ValueError("股票代码列表存在重复")

    return codes


def _create_login_manager():
    """
    创建登录管理器。

    只有真正需要下载数据时才登录。
    同一次构建最多登录一次。
    """

    logged_in = False

    def ensure_login():
        nonlocal logged_in

        if not logged_in:
            baostock_client.login()
            logged_in = True

    def logout_if_needed():
        nonlocal logged_in

        if logged_in:
            try:
                baostock_client.logout()
            finally:
                logged_in = False

    return ensure_login, logout_if_needed


def _validate_single_stock_data(df, code):
    """检查单只股票的原始行情数据。"""

    if df is None or df.empty:
        raise RuntimeError(
            f"{code} 没有获取到有效行情数据"
        )

    panel.check_schema(df)

    actual_codes = set(
        df["code"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )

    if actual_codes != {code}:
        raise ValueError(
            f"{code} 的行情数据中出现异常股票代码："
            f"{actual_codes}"
        )


# ==================================================
# 2. 加载一批股票的原始数据
# ==================================================

def _load_raw_data(
    codes,
    start_date,
    end_date,
    ensure_login,
    force_refresh=False,
    cache_dir=None,
):
    """
    加载一批股票的原始行情。

    优先读取缓存。
    缓存不存在或为空时才下载。

    Returns
    -------
    raw_panel : pd.DataFrame

    stats : dict
        cache_hits: 缓存命中数量
        downloaded: 实际下载数量
    """

    raw_frames = []

    cache_hits = 0
    downloaded = 0

    for code in codes:

        cached_df = None

        # ------------------------------------------
        # 1. 检查缓存
        # ------------------------------------------

        if not force_refresh:

            cached_df = cache.load_daily_cache(
                code,
                start_date,
                end_date,
                cache_dir=cache_dir,
            )

        if cached_df is not None and not cached_df.empty:

            _validate_single_stock_data(
                cached_df,
                code,
            )

            raw_df = cached_df
            cache_hits += 1

        else:

            # --------------------------------------
            # 2. 下载数据
            # --------------------------------------

            ensure_login()

            raw_df = baostock_client.query_daily(
                code,
                start_date,
                end_date,
            )

            # 不允许空数据进入有效缓存
            _validate_single_stock_data(
                raw_df,
                code,
            )

            # --------------------------------------
            # 3. 保存缓存
            # --------------------------------------

            cache.save_daily_cache(
                code,
                start_date,
                end_date,
                raw_df,
                cache_dir=cache_dir,
            )

            downloaded += 1

        raw_frames.append(raw_df)

    raw_panel = pd.concat(
        raw_frames,
        ignore_index=True,
    )

    stats = {
        "cache_hits": cache_hits,
        "downloaded": downloaded,
    }

    return raw_panel, stats


# ==================================================
# 3. 全市场批量 Pipeline
# ==================================================

def build_daily_panel_to_parquet(
    codes,
    start_date,
    end_date,
    output_path=None,
    batch_size=100,
    force_refresh=False,
    cache_dir=None,
):
    """
    分批获取全市场日线数据，并写入单个 Parquet 文件。

    不将全部股票数据同时保存在内存中。

    Parameters
    ----------
    codes : list
        全部股票代码。

    start_date : str
        行情开始日期。

    end_date : str
        行情结束日期。

    output_path : str or Path or None
        最终 Parquet 文件路径。

    batch_size : int
        每批股票数量，默认 100。

    force_refresh : bool
        是否强制重新下载。

    cache_dir : str or Path or None
        原始数据缓存目录。

    Returns
    -------
    report : dict
        数据构建报告。
    """

    # ------------------------------------------
    # 1. 参数检查
    # ------------------------------------------

    codes = _validate_codes(codes)

    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    # 固定排序，保证每次批次划分一致
    codes = sorted(codes)

    if output_path is None:

        output_path = (
            config.DATA_PROCESSED_DIR
            / "daily_panel.parquet"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 临时文件
    temp_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    # 清理上次遗留的临时文件
    if temp_path.exists():
        temp_path.unlink()

    total_batches = (
        len(codes) + batch_size - 1
    ) // batch_size

    # ------------------------------------------
    # 2. 初始化
    # ------------------------------------------

    ensure_login, logout_if_needed = (
        _create_login_manager()
    )

    writer = None
    arrow_schema = None

    total_rows = 0
    total_cache_hits = 0
    total_downloaded = 0

    actual_start_date = None
    actual_end_date = None

    # ------------------------------------------
    # 3. 分批处理
    # ------------------------------------------

    try:

        for batch_index, start in enumerate(
            range(0, len(codes), batch_size),
            start=1,
        ):

            batch_codes = codes[
                start:start + batch_size
            ]

            print(
                f"\n========== "
                f"Batch {batch_index}/{total_batches} "
                f"=========="
            )

            print(
                f"股票数量：{len(batch_codes)}"
            )

            # ----------------------------------
            # 3.1 加载原始数据
            # ----------------------------------

            raw_batch, load_stats = _load_raw_data(
                codes=batch_codes,
                start_date=start_date,
                end_date=end_date,
                ensure_login=ensure_login,
                force_refresh=force_refresh,
                cache_dir=cache_dir,
            )

            total_cache_hits += (
                load_stats["cache_hits"]
            )

            total_downloaded += (
                load_stats["downloaded"]
            )

            # ----------------------------------
            # 3.2 清洗
            # ----------------------------------

            daily_batch = panel.clean_daily_panel(
                raw_batch
            )

            # ----------------------------------
            # 3.3 数据质量检查
            # ----------------------------------

            batch_report = (
                panel.validate_daily_panel(
                    daily_batch
                )
            )

            # 确认本批股票没有遗漏或混入
            actual_codes = set(
                daily_batch["code"]
                .astype("string")
                .unique()
            )

            if actual_codes != set(batch_codes):

                raise ValueError(
                    "本批实际股票代码与预期不一致。"
                    f"缺失：{set(batch_codes) - actual_codes}；"
                    f"额外：{actual_codes - set(batch_codes)}"
                )

            # ----------------------------------
            # 3.4 统一排序
            # ----------------------------------

            # 按股票代码、日期排序。
            # 因为批次之间的股票代码不重复且有序，
            # 所以最终文件也保持确定的代码顺序。

            daily_batch = (
                daily_batch
                .sort_values(["code", "date"])
                .reset_index(drop=True)
            )

            # ----------------------------------
            # 3.5 转换为 Arrow Table
            # ----------------------------------

            table = pa.Table.from_pandas(
                daily_batch,
                preserve_index=False,
            )

            if writer is None:

                # 第一个批次确定 Parquet Schema
                arrow_schema = table.schema

                writer = pq.ParquetWriter(
                    str(temp_path),
                    arrow_schema,
                    compression="snappy",
                )

            else:

                # 后续批次必须具有一致的数据结构
                same_schema = (
                    table.schema
                    .remove_metadata()
                    .equals(
                        arrow_schema.remove_metadata()
                    )
                )

                if not same_schema:

                    raise TypeError(
                        "不同批次的 Arrow Schema 不一致："
                        f"{table.schema}"
                    )

                table = table.cast(
                    arrow_schema
                )

            # ----------------------------------
            # 3.6 写入 Parquet
            # ----------------------------------

            writer.write_table(
                table,
                row_group_size=100_000,
            )

            # ----------------------------------
            # 3.7 更新统计信息
            # ----------------------------------

            total_rows += len(daily_batch)

            batch_start = batch_report["start_date"]
            batch_end = batch_report["end_date"]

            if (
                actual_start_date is None
                or batch_start < actual_start_date
            ):
                actual_start_date = batch_start

            if (
                actual_end_date is None
                or batch_end > actual_end_date
            ):
                actual_end_date = batch_end

            print(
                f"缓存命中：{load_stats['cache_hits']}"
            )

            print(
                f"新下载：{load_stats['downloaded']}"
            )

            print(
                f"本批数据行数：{len(daily_batch)}"
            )

            print(
                f"累计数据行数：{total_rows}"
            )

            # 释放本批数据
            del raw_batch
            del daily_batch
            del table

        # --------------------------------------
        # 4. 完成写入
        # --------------------------------------

        if writer is None:
            raise RuntimeError(
                "没有生成任何有效行情数据"
            )

        writer.close()
        writer = None

        # 原子替换正式文件
        temp_path.replace(output_path)

    except Exception:

        # 关闭尚未完成的 Parquet Writer
        if writer is not None:

            try:
                writer.close()
            except Exception:
                pass

        # 删除未完成的临时文件
        if temp_path.exists():
            temp_path.unlink()

        raise

    finally:

        logout_if_needed()

    # ------------------------------------------
    # 5. 返回构建报告
    # ------------------------------------------

    report = {
        "output_path": str(output_path),
        "n_stocks": len(codes),
        "n_batches": total_batches,
        "n_rows": total_rows,
        "cache_hits": total_cache_hits,
        "downloaded": total_downloaded,
        "start_date": actual_start_date,
        "end_date": actual_end_date,
    }

    print("\n========== 数据构建完成 ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    return report
