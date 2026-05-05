from enum import Enum
from pathlib import Path

# Projektroot: 3 Ebenen Ã¼ber dieser Datei (src/timeseries_forecast/data_prep/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_ALUMINIUM_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "Aluminium_2012_2022.csv"

METALS_ALUMINIUM_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Aluminium_Historical_Data.csv"
METALS_COPPER_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Copper_Historical_Data.csv"
METALS_GOLD_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Gold_Historical_Data.csv"
METALS_LEAD_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Lead_Historical_Data.csv"
METALS_NICKEL_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Nickel_Historical_Data.csv"
METALS_SILVER_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Silver_Historical_Data.csv"
METALS_ZINC_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "Zinc_Historical_Data.csv"


class RawMetalsColumn(str, Enum):
    OPEN_TIME = "Date"
    PRICE = "Price"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    VOLUME = "Vol."
    PCT_CHANGE = "Change %"
