from dataclasses import dataclass, field, MISSING
from typing import List
import copy
import pandas as pd


@dataclass
class FeatureConfig:
    date: List = field(
        default=MISSING,
        metadata={"help": "Column name of the date column"},
    )
    target: str = field(
        default=MISSING,
        metadata={"help": "Column name of the target column"},
    )

    original_target: str = field(
        default=None,
        metadata={
            "help": "Column name of the original target column in acse of transformed target. If None, it will be assigned same value as target"
        },
    )

    continuous_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields. Defaults to []"},
    )
    boolean_features: List[str] = field(
        default_factory=list,
        metadata={"help": "Column names of the boolean fields. Defaults to []"},
    )

    index_cols: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names which needs to be set as index in the X and Y dataframes."
        },
    )
    exogenous_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the exogenous features. Must be a subset of categorical and continuous features"
        },
    )
    metadata_features: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Column names of the metadata features. Those are not used for training, but kept in the dataframe"
        },
    )
    feature_list: List[str] = field(init=False)

    def __post_init__(self):
        assert (
                len(self.categorical_features) + len(self.continuous_features) > 0
        ), "There should be at-least one feature defined in categorical or continuous columns"
        self.feature_list = (
                self.categorical_features + self.continuous_features + self.boolean_features
        )
        assert (
                self.target not in self.feature_list
        ), f"`target`({self.target}) should not be present in either categorical, continuous or boolean feature list"
        assert (
                self.date not in self.feature_list
        ), f"`date`({self.target}) should not be present in either categorical, continuous or boolean feature list"
        extra_exog = set(self.exogenous_features) - set(self.feature_list)
        assert (
                len(extra_exog) == 0
        ), f"These exogenous features are not present in feature list: {extra_exog}"
        intersection = (
            set(self.continuous_features)
            .intersection(self.categorical_features + self.boolean_features)
            .union(
                set(self.categorical_features).intersection(
                    self.continuous_features + self.boolean_features
                )
            )
            .union(
                set(self.boolean_features).intersection(
                    self.continuous_features + self.categorical_features
                )
            )
        )
        assert (
                len(intersection) == 0
        ), f"There should not be any overlaps between the categorical contonuous and boolean features. {intersection} are present in more than one definition"
        if self.original_target is None:
            self.original_target = self.target

    def get_X_y(
            self, df: pd.DataFrame, categorical: bool = False, exogenous: bool = False
    ):
        feature_list = copy.deepcopy(self.continuous_features)
        feature_list += self.metadata_features
        if categorical:
            feature_list += self.categorical_features + self.boolean_features
        if not exogenous:
            feature_list = list(set(feature_list) - set(self.exogenous_features))
        feature_list = list(set(feature_list))
        delete_index_cols = list(set(self.index_cols) - set(self.feature_list))
        (X, y, y_orig) = (
            df.loc[:, list(set(feature_list + self.index_cols))]
            .set_index(self.index_cols, drop=False)
            .drop(columns=delete_index_cols),
            df.loc[:, [self.target] + self.index_cols].set_index(
                self.index_cols, drop=True
            )
            if self.target in df.columns
            else None,
            df.loc[:, [self.original_target] + self.index_cols].set_index(
                self.index_cols, drop=True
            )
            if self.original_target in df.columns
            else None,
        )
        return X, y, y_orig