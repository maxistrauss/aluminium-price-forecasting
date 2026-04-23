import pandas as pd


def split_data(
    df: pd.DataFrame,
    train_end: str = "2021-12-31 23:00:00",
    val_end: str = "2023-06-30 23:00:00",
    mode_column: str = "Mode",
    train_label: str = "Train",
    val_label: str = "Val",
    test_label: str = "Test",
    print_summary: bool = True,
) -> pd.DataFrame:
    """Splits the combined dataset into Train, Val and Test sets based on predefined date ranges.

    Args:
        df (pd.DataFrame): The combined dataset containing all features and the target variable.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the Train, Val and Test DataFrames.
    """
    train = df[df.index <= train_end].copy()
    val = df[(df.index > train_end) & (df.index <= val_end)].copy()
    test = df[df.index > val_end].copy()

    train[mode_column] = train_label
    val[mode_column] = val_label
    test[mode_column] = test_label

    if print_summary:
        print("Splits:")
        print(f"  {train_label}: {train.index.min()} -> {train.index.max()} | rows={len(train)}")
        print(f"  {val_label}:   {val.index.min()} -> {val.index.max()} | rows={len(val)}")
        print(f"  {test_label}:  {test.index.min()} -> {test.index.max()} | rows={len(test)}")

    return pd.concat([train, val, test])