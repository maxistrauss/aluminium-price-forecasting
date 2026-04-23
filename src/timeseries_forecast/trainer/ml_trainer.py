from typing import List, Literal
import plotly.express as px
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
from timeseries_forecast.eval.eval import EvaluationResults, evaluate
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
from lightgbm import LGBMRegressor


class MLTrainer():
    """
    The MLTrainer class is responsible for training machine learning models.

    Attributes:
        ml_models (list): A list to store the trained machine learning models.
    """

    def train_model_multi_step(
            self,
            df: pd.DataFrame,
            define_features: callable,
            model: BaseEstimator,
            cat_encoder_class: BaseEstimator.__class__ = None,
            param_bounds: callable = None,
            model_params: List[dict] = None,
            normalize: bool = False,
            encode_categorical: bool = False,
            n_steps: int = 9,
            fit_kwargs={},
            verbose: bool = True,
            feature_selection_enabled=False,
    ) -> EvaluationResults:
        """
        Trains a model for multi-step forecasting.
        In this method, we train a separate model for each step of the forecast horizon.
        Meaning for forecasting 9 Month, we train 9 models, each for forecasting its corresponding month.

        Args:
            df (DataFrame): The dataframe containing the data.
            define_features (callable): A function to define the features for the model.
            cat_encoder_class (class): The class of the categorical encoder to be used.
            model (BaseEstimator): The model to be trained.
            normalize (bool, optional): Whether to normalize the input or not. Defaults to False.
            encode_categorical (bool, optional): Whether to encode categorical values before fitting. Defaults to False.
            n_steps (int, optional): The number of months to forecast. Defaults to 9.
            param_bounds (dict, optional): The dictionary of hyperparameters to be tuned. Defaults to None. If None, no hyperparameter tuning is done.
            model_params (List[dict], optional): The list of model hyperparameters for each step. Defaults to None.
            fit_kwargs (dict, optional): Additional keyword arguments to be passed to the fit function of the model.
            verbose (bool, optional): Whether to print the feature importance or not. Defaults to True.
            feature_selection_enabled (bool, optional): Whether to enable feature selection or not. Defaults to False.

        Returns:
            eval_df (DataFrame): The dataframe containing the predictions for each step.
        """
        if model_params and len(model_params) != n_steps:
            raise ValueError(
                f"Length of model_params should be equal to n_steps. Expected {n_steps}, got {len(model_params)}"
            )

        eval_df = None

        test_scores = []
        for forecast_horizon in range(1, n_steps + 1):
            # get the features from the definition
            feature_config, features_df = define_features(df, forecast_horizon=forecast_horizon)

            if model_params:
                model_config.model.set_params(**model_params[forecast_horizon - 1])
                self.logger.debug(f"Set parameters: {model_config.model.get_params()}")

            # tune hyperparameters if param_bounds is provided
            if param_bounds is not None:
                # split the data into train and val
                train_df = features_df[(features_df["mode"] == "train")].dropna()
                val_df = features_df[features_df["mode"] == "val"].dropna()
                # get the features and target
                train_features, train_target, train_original_target = feature_config.get_X_y(
                    train_df, categorical=True, exogenous=True
                )
                val_features, val_target, val_original_target = feature_config.get_X_y(
                    val_df, categorical=True, exogenous=True
                )

                if feature_selection_enabled:
                    important_features, model_config = self.select_features(model_config, feature_config, train_features, train_target)
                    train_features = train_features[important_features]
                    val_features = val_features[important_features]
                    if "categorical_feature" in fit_kwargs:
                        fit_kwargs["categorical_feature"] = fit_kwargs["categorical_feature"].intersection(set(important_features))

                best_params, best_score = self.tune_hyperparameters(model_config, feature_config, train_features,
                                                                    train_target, val_features, val_target,
                                                                    param_bounds, fit_kwargs)
                model_config.model.set_params(**best_params)
                self.logger.debug(f"Best parameters: {model_config.model.get_params()}")

            # split the data into train and test
            train_df = features_df[features_df["BTC_Mode"] == "Train"].dropna()
            val_df = features_df[features_df["BTC_Mode"] == "Val"]
            test_df = features_df[features_df["BTC_Mode"] == "Test"]
         
            assert not test_df.isna().any().any() and not train_df.isna().any().any()

            # get the features and target
            train_features, train_target, train_original_target = feature_config.get_X_y(
                train_df, categorical=True, exogenous=True
            )

            test_features, test_target, test_original_target = feature_config.get_X_y(
                test_df, categorical=True, exogenous=True
            )

            if feature_selection_enabled and not param_bounds:
                important_features, model_config = self.select_features(model_config, feature_config, train_features, train_target)
                train_features = train_features[important_features]
                test_features = test_features[important_features]
                if "categorical_feature" in fit_kwargs:
                    fit_kwargs["categorical_feature"] = fit_kwargs["categorical_feature"].intersection(set(important_features))

            model.fit(train_features, train_target)
            # val_preds = model.predict(val_features)
            # val_eval_df = val_target.copy()
            # val_eval_df["pred"] = val_preds
            # val_score = evaluate(val_eval_df, target_col="BTC_Close_log_ret", y_pred_col="pred", do_calculate_directed_accuracy=True)
            test_preds = model.predict(test_features)
            test_eval_df = test_target.copy()
            test_eval_df["pred"] = test_preds
            test_score = evaluate(test_eval_df, target_col="BTC_Close_log_ret", y_pred_col="pred", do_calculate_directed_accuracy=True)
            test_scores.append(test_score)
        
        return self._aggregate_evaluation_scores(test_scores)

    def _aggregate_evaluation_scores(self, eval_scores: List[EvaluationResults]) -> EvaluationResults:
        if not eval_scores:
            raise ValueError("eval_scores must not be empty")

        aggregated_scores = {
            key: np.mean([getattr(eval_score, key) for eval_score in eval_scores])
            for key in vars(eval_scores[0]).keys()
        }

        return EvaluationResults(**aggregated_scores)

    def tune_hyperparameters(
            self,
            model_config,
            feature_config,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            val_target: pd.Series,
            param_bounds: callable,
            fit_kwargs={}
    ):
        """
        Tune the hyperparameters of a given model using Bayesian Optimization.

        Args:
            model_config (ModelConfig): The configuration for the model to be tuned.
            feature_config (FeatureConfig): The configuration for the features to be used in training.
            X_train (DataFrame): The feature values for training the model.
            y_train (Series or ndarray): The target values for training the model.
            X_val (DataFrame): The feature values for validating the model.
            val_target (Series or ndarray): The target values for validating the model.
            param_bounds (callable): The function to define the bounds of the hyperparameters.
            fit_kwargs (dict, optional): Additional keyword arguments to be passed to the fit function of the model.

        Returns:
            dict: The best hyperparameters.
            float: The best score.
        """
        self.logger.debug(f"Tuning hyperparameters for {model_config.model.__class__.__name__}")

        def objective(trial: optuna.Trial):
            params = param_bounds(trial)
            model_config.model.set_params(**params)
            ml_model = MLForecast(
                model_config=model_config,
                feature_config=feature_config,
            )
            y_pred, feat_df = self.train_model(
                ml_model,
                X_train,
                y_train,
                X_val,
                fit_kwargs=fit_kwargs,
            )
            # Return the MAE metric as the value
            y_pred[y_pred < 0] = 0
            return evaluation.calc_mae(val_target["demand"], y_pred)

        # Create a sampler and set seed for repeatability.
        # Set startup trials as 5 because out total trials is lower.
        sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=42)
        # Create a study
        study = optuna.create_study(direction="minimize", sampler=sampler)
        # Start the optimization run
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        # fig = optuna.visualization.plot_param_importances(study)
        # fig.show()

        bo_search_trials = study.trials_dataframe()
        best_params = study.best_params
        best_score = study.best_value
        self.logger.debug(bo_search_trials.sort_values("value").head())
        self.logger.debug(f"Best parameters: {best_params}")
        self.logger.debug(f"Best score: {best_score}")
        return best_params, best_score

    
    def select_features(self, model_config, feature_config, X_train, y_train, threshold='mean'):
        """
        Selects features based on importance using aLGBMRegressor .

        Args:
            X_train (DataFrame): Training features.
            y_train (Series): Training target.
            threshold (str or float): The threshold for feature selection.
                                      'mean' will use the mean importance as threshold.
                                      Can be a numerical value to set a specific threshold.

        Returns:
            DataFrame: Transformed DataFrame with selected features.
            model: The model used for feature importance.
        """
        model = LGBMRegressor()
        original_model = model_config.model
        model_config.model = model
        ml_model = MLForecast(
            model_config=model_config,
            feature_config=feature_config,
        )
        X_train = X_train.drop(columns=feature_config.metadata_features, errors='ignore')
        ml_model.fit(X_train, y_train, fit_kwargs={"categorical_feature": feature_config.categorical_features})
        selector = SelectFromModel(ml_model._model, threshold=threshold, prefit=True)
        selected_features = X_train.columns[selector.get_support()]
        feature_config.categorical_features = list(set(feature_config.categorical_features).intersection(set(selected_features)))
        feature_config.boolean_features = list(set(feature_config.boolean_features).intersection(set(selected_features)))
        feature_config.continuous_features = list(set(feature_config.continuous_features).intersection(set(selected_features)))
        feature_config.feature_list = feature_config.categorical_features + feature_config.continuous_features + feature_config.boolean_features
        self.logger.debug(f"Selected features: {len(selected_features)} / {len(X_train.columns)}")
        model_config.categorical_encoder.cols = feature_config.categorical_features
        model_config.model = original_model
        return list(selected_features) + feature_config.metadata_features, model_config
