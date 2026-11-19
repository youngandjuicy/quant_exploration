
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
# 2. BaoStock 日线数据配置
# ==================================================

DAILY_FIELDS = (
    "date,code,open,high,low,close,preclose,"
    "volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
)

DAILY_FREQUENCY = "d"

# 1：后复权
DAILY_ADJUST_FLAG = "1"


# ==================================================
# 3. 正式研究配置
# ==================================================

# 研究区间
RESEARCH_START_DATE = "2018-01-01"
RESEARCH_END_DATE = "2025-12-31"

# 数据集划分
DISCOVERY_START_DATE = "2018-01-01"
DISCOVERY_END_DATE = "2022-12-31"

VALIDATION_START_DATE = "2023-01-01"
VALIDATION_END_DATE = "2024-12-31"

TEST_START_DATE = "2025-01-01"
TEST_END_DATE = "2025-12-31"

# 因子参数
MOMENTUM_WINDOW = 20
FORWARD_HORIZON = 5

# 沪深 A 股
A_SHARE_CODE_PREFIXES = (
    "sh.6",
    "sz.0",
    "sz.30",
)

