import pandas as pd

from timeseries_forecast.data_prep.constants import RAW_ALUMINIUM_DATA_PATH, RawAluColumn, METALS_ALUMINIUM_DATA_PATH, METALS_COPPER_DATA_PATH, METALS_GOLD_DATA_PATH, METALS_LEAD_DATA_PATH, METALS_NICKEL_DATA_PATH, METALS_SILVER_DATA_PATH, METALS_ZINC_DATA_PATH   
# add more paths if more files


def _parse_volume_to_float(value: object) -> float:
    if pd.isna(value):
        return float("nan")

    text = str(value).strip()
    if text in {"", "-"}:
        return float("nan")

    multiplier = 1.0
    suffix = text[-1].upper()
    if suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]

    base = pd.to_numeric(text.replace(",", ""), errors="coerce")
    if pd.isna(base):
        return float("nan")
    return float(base) * multiplier


def read_raw_alu_data():
    df = pd.read_csv(METALS_ALUMINIUM_DATA_PATH)
    df[RawAluColumn.OPEN_TIME] = pd.to_datetime(df[RawAluColumn.OPEN_TIME])

    # Convert market-style numeric strings (e.g. 2,528.00 | 21.89K | -0.22%)
    # so downstream interpolation/checking works across different datasets.
    for col in [RawAluColumn.PRICE, RawAluColumn.OPEN, RawAluColumn.HIGH, RawAluColumn.LOW]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df[RawAluColumn.VOLUME] = df[RawAluColumn.VOLUME].map(_parse_volume_to_float)
    df[RawAluColumn.PCT_CHANGE] = pd.to_numeric(
        df[RawAluColumn.PCT_CHANGE].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )

    # The aluminium source is daily data, so keep the loader generic and only
    # remove duplicated dates if the raw file contains any.
    df = df.drop_duplicates(subset=[RawAluColumn.OPEN_TIME]).reset_index(drop=True)
    return df



