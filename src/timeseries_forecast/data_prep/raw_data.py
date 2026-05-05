import pandas as pd

from timeseries_forecast.data_prep.constants import (
    RAW_ALUMINIUM_DATA_PATH,
    METALS_ALUMINIUM_DATA_PATH,
    METALS_COPPER_DATA_PATH,
    METALS_GOLD_DATA_PATH,
    METALS_LEAD_DATA_PATH,
    METALS_NICKEL_DATA_PATH,
    METALS_SILVER_DATA_PATH,
    METALS_ZINC_DATA_PATH,
    RawMetalsColumn,
)


METAL_DATA_PATHS: dict[str, object] = {
    "aluminium": METALS_ALUMINIUM_DATA_PATH,
    "copper": METALS_COPPER_DATA_PATH,
    "gold": METALS_GOLD_DATA_PATH,
    "lead": METALS_LEAD_DATA_PATH,
    "nickel": METALS_NICKEL_DATA_PATH,
    "silver": METALS_SILVER_DATA_PATH,
    "zinc": METALS_ZINC_DATA_PATH,
}


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


def _load_raw_metals_data(data_path: object) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df[RawMetalsColumn.OPEN_TIME] = pd.to_datetime(
        df[RawMetalsColumn.OPEN_TIME],
        format="%d-%m-%Y",
        dayfirst=True,
        errors="coerce",
    )

    # Convert market-style numeric strings (e.g. 2,528.00 | 21.89K | -0.22%)
    # so downstream interpolation/checking works across different datasets.
    for col in [RawMetalsColumn.PRICE, RawMetalsColumn.OPEN, RawMetalsColumn.HIGH, RawMetalsColumn.LOW]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")

    df[RawMetalsColumn.VOLUME] = df[RawMetalsColumn.VOLUME].map(_parse_volume_to_float)
    df[RawMetalsColumn.PCT_CHANGE] = pd.to_numeric(
        df[RawMetalsColumn.PCT_CHANGE].astype(str).str.replace("%", "", regex=False),
        errors="coerce",
    )

    # The metals source is daily data, so keep the loader generic and only
    # remove duplicated dates if the raw file contains any.
    df = df.drop_duplicates(subset=[RawMetalsColumn.OPEN_TIME]).reset_index(drop=True)
    return df


def read_raw_metals_data(metal_name: str) -> pd.DataFrame:
    """Load one of the metals datasets by name.

    Parameters
    ----------
    metal_name : str
        Supported values are the keys of ``METAL_DATA_PATHS``.
    """

    normalized_name = metal_name.strip().lower()
    try:
        data_path = METAL_DATA_PATHS[normalized_name]
    except KeyError as exc:
        supported = ", ".join(sorted(METAL_DATA_PATHS))
        raise ValueError(f"Unsupported metal '{metal_name}'. Supported metals: {supported}") from exc

    return _load_raw_metals_data(data_path)



