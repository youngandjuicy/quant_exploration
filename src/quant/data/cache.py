
import hashlib
from pathlib import Path

import pandas as pd

from .. import config


def get_daily_cache_path(
    code,
    start_date,
    end_date,
    cache_dir=None,
):
    """
    根据股票代码、日期范围和行情参数生成缓存路径。
    """

    if cache_dir is None:
        cache_dir = (
            config.DATA_RAW_DIR
            / "baostock"
            / "daily"
        )
    else:
        cache_dir = Path(cache_dir)

    # 避免不同字段配置误用同一份缓存
    fields_hash = hashlib.sha256(
        config.DAILY_FIELDS.encode("utf-8")
    ).hexdigest()[:8]

    safe_code = code.replace(".", "_")

    filename = (
        f"{safe_code}_"
        f"{start_date.replace('-', '')}_"
        f"{end_date.replace('-', '')}_"
        f"{config.DAILY_FREQUENCY}_"
        f"adj{config.DAILY_ADJUST_FLAG}_"
        f"{fields_hash}.csv"
    )

    return cache_dir / filename


def load_daily_cache(
    code,
    start_date,
    end_date,
    cache_dir=None,
):
    """
    读取本地日线缓存。

    Returns
    -------
    pd.DataFrame or None

    缓存存在：返回 DataFrame。
    缓存不存在：返回 None。
    """

    path = get_daily_cache_path(
        code,
        start_date,
        end_date,
        cache_dir,
    )

    if not path.exists():
        return None

    try:
        df = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"缓存文件读取失败：{path}"
        ) from exc

    return df


def save_daily_cache(
    code,
    start_date,
    end_date,
    df,
    cache_dir=None,
):
    """
    将原始日线 DataFrame 保存到本地缓存。

    使用临时文件写入，完成后再替换正式文件，
    降低程序中断导致缓存文件不完整的风险。
    """

    path = get_daily_cache_path(
        code,
        start_date,
        end_date,
        cache_dir,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(".csv.tmp")

    df.to_csv(
        temp_path,
        index=False,
        encoding="utf-8-sig",
    )

    temp_path.replace(path)

    return path
