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
METALS_SP500_DATA_PATH = PROJECT_ROOT / "data" / "raw_data" / "new_metals" / "SnP_500.csv"

# Konfiguration für das Merging von Datensätzen
# Ändere diese Liste, um Datensätze hinzuzufügen oder zu entfernen
# Format: [("PRÄFIX", "parameter_name_in_merge_function"), ...]
# MERGE_DATASET_CONFIG_OLD = [
#     ("BTC", "btc_df_cleaned"),
#     ("ETH", "eth_df_cleaned"),
#     ("GOLD", "gold_df_cleaned"),
#     ("SP500", "sp500_df_cleaned"),
# ]

MERGE_DATASET_CONFIG = [
    ("ALU", "aluminium_df_cleaned"),
    ("COPPER", "copper_df_cleaned"),
    ("GOLD", "gold_df_cleaned"),
    ("LEAD", "lead_df_cleaned"),
    ("NICKEL", "nickel_df_cleaned"),
    ("SILVER", "silver_df_cleaned"),
    ("ZINC", "zinc_df_cleaned"),
    ("SP500", "sp500_df_cleaned"),
]

class RawMetalsColumn(str, Enum):
    OPEN_TIME = "Date"
    PRICE = "Price"
    OPEN = "Open"
    HIGH = "High"
    LOW = "Low"
    VOLUME = "Vol."
    PCT_CHANGE = "Change %"
