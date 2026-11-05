
from pathlib import Path


# ==================================================
# 1. 项目路径
# ==================================================

# config.py 位于 src/quant/ 下
# parents[2] 指向项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ==================================================
# 2. Smoke Test 配置
# ==================================================

SMOKE_TEST_CODES = [
    "sh.600000",
    "sh.600519",
    "sh.601318",
    "sz.000001",
    "sz.000858",
]

SMOKE_TEST_START_DATE = "2024-01-01"
SMOKE_TEST_END_DATE = "2024-06-30"


# ==================================================
# 3. BaoStock 日线数据配置
# ==================================================

DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
)

DAILY_FREQUENCY = "d"

# 1：后复权
DAILY_ADJUST_FLAG = "1"
