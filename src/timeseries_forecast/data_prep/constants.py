from enum import Enum
from pathlib import Path

# Projektroot: 3 Ebenen Ã¼ber dieser Datei (src/timeseries_forecast/data_prep/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ALUMINIUM_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "Aluminium_2012_2022.csv"


class RawAluColumn(str, Enum):
    OPEN_TIME = "Date"
    PRICE = "Price"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    VOLUME = "Vol."
    PCT_CHANGE = "Change %"
