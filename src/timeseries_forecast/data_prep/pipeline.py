from typing import Callable, Optional, Union
import pandas as pd
from timeseries_forecast.data_prep.data_cleaning import set_timestamp_as_index, interpolate_gaps
from timeseries_forecast.data_prep.checks import check_0_values, check_duplicates, check_missing_values, check_nan_values
from timeseries_forecast.data_prep.data_split import split_data


def run_all_checks(
    df: pd.DataFrame,
    expected_freq: Optional[str] = "h",
    run_zero_value_check: bool = True,
    zero_value_ignore_cols: Optional[list[str]] = None,
    fail_on_row_duplicates: bool = False,
) -> bool:

    # 1) Check for duplicates
    duplicate_check_result = check_duplicates(df)

    # 2) Check for missing values (gaps in time series)
    has_gaps = check_missing_values(df, expected_freq=expected_freq)

    # 3) Check for NaN values
    nan_check_result = check_nan_values(df)

    # 4) Check for Null values
    if run_zero_value_check:
        null_check_result = check_0_values(df, ignore_columns=zero_value_ignore_cols)
        has_null = null_check_result.has_null
    else:
        has_null = False

    all_checks_passed = (
        not duplicate_check_result.has_index_duplicates
        and (not duplicate_check_result.has_row_duplicates or not fail_on_row_duplicates)
        and not has_gaps
        and not nan_check_result.has_nan
        and not has_null
    )
    return all_checks_passed


def run_data_pipeline(
    data_frame_or_loader: Union[pd.DataFrame, Callable],
    timestamp_column: str = "timestamp",
    data_freq: Optional[str] = "h",
    interpolation_freq: Optional[str] = None,
    interpolate_exclude_cols: Optional[list[str]] = None,
    interpolate_datetime_cols: Optional[list[str]] = None,
    zero_as_nan_cols: Optional[list[str]] = None,
    run_zero_value_check: bool = True,
    zero_value_ignore_cols: Optional[list[str]] = None,
    fail_on_row_duplicates: bool = False,
    do_split: bool = True,
    split_train_end: str = "2021-12-31 23:00:00",
    split_val_end: str = "2023-06-30 23:00:00",
    split_mode_column: str = "Mode",
    split_train_label: str = "Train",
    split_val_label: str = "Val",
    split_test_label: str = "Test",
) -> pd.DataFrame:
    # Step 1: Load raw data (accept either DataFrame directly or a callable that returns one)
    if isinstance(data_frame_or_loader, pd.DataFrame):
        df_raw = data_frame_or_loader
    else:
        df_raw = data_frame_or_loader()
    df = set_timestamp_as_index(df_raw, timestamp_column=timestamp_column)

    all_checks_passed = run_all_checks(
        df,
        expected_freq=data_freq,
        run_zero_value_check=run_zero_value_check,
        zero_value_ignore_cols=zero_value_ignore_cols,
        fail_on_row_duplicates=fail_on_row_duplicates,
    )

    effective_interpolation_freq = interpolation_freq if interpolation_freq is not None else data_freq
    if effective_interpolation_freq is not None:
        df = interpolate_gaps(
            df,
            freq=effective_interpolation_freq,
            exclude_cols=interpolate_exclude_cols,
            datetime_cols=interpolate_datetime_cols,
            zero_as_nan_cols=zero_as_nan_cols,
        )

    all_checks_passed = run_all_checks(
        df,
        expected_freq=data_freq,
        run_zero_value_check=run_zero_value_check,
        zero_value_ignore_cols=zero_value_ignore_cols,
        fail_on_row_duplicates=fail_on_row_duplicates,
    )
    if not all_checks_passed:
        raise ValueError("Data checks failed after cleaning. Please investigate the issues before proceeding.")

    if do_split:
        df = split_data(
            df,
            train_end=split_train_end,
            val_end=split_val_end,
            mode_column=split_mode_column,
            train_label=split_train_label,
            val_label=split_val_label,
            test_label=split_test_label,
        )

    return df


if __name__ == "__main__":
    print("Use run_data_pipeline(...) by passing a dataset-specific loader and configuration.")

    
