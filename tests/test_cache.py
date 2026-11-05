
import tempfile

import pandas as pd

from quant.data.cache import (
    save_daily_cache,
    load_daily_cache,
)


def test_cache_round_trip():

    # 构造一份模拟原始数据
    df = pd.DataFrame(
        [
            ["2024-01-02", "sh.600000", "10.5"],
            ["2024-01-03", "sh.600000", "10.8"],
        ],
        columns=["date", "code", "close"],
    )

    # 使用临时目录测试
    with tempfile.TemporaryDirectory() as tmp:

        # 第一次读取：缓存应该不存在
        cached = load_daily_cache(
            "sh.600000",
            "2024-01-01",
            "2024-01-31",
            cache_dir=tmp,
        )

        assert cached is None

        # 保存缓存
        path = save_daily_cache(
            "sh.600000",
            "2024-01-01",
            "2024-01-31",
            df,
            cache_dir=tmp,
        )

        assert path.exists()

        # 第二次读取：应该命中缓存
        cached = load_daily_cache(
            "sh.600000",
            "2024-01-01",
            "2024-01-31",
            cache_dir=tmp,
        )

        pd.testing.assert_frame_equal(
            df,
            cached,
        )

    print("Cache test passed!")


if __name__ == "__main__":
    test_cache_round_trip()
