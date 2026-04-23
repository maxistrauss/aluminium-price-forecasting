from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_squared_error 
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class EvaluationResults:
    mae: float
    rmse: float
    mse: float
    r2: float
    smape: float
    smape_total: float
    forecast_bias: float
    accuracy: float
    accuracy_total: float
    directed_accuracy: float

def calculate_forecast_bias(y_true: pd.Series, y_pred: pd.Series) -> float:
    bias = (y_pred - y_true).mean()
    return bias

def symmetric_mean_absolute_percentage_error(actual: np.array, forecast: np.array) -> np.float64:
    """
        This function measures the SMAPE https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        The lower, the better.

        :param actual: ground truth with dimension n*1
        :param predicted: predicted from model with dimension n*1
        :return: returns the error in float, if there is a shape missmatch the error will be 1
    """

    if actual.shape == forecast.shape and len(forecast.shape) == 1:
        return np.sum(np.abs((forecast - actual))) / (np.sum(actual + forecast) + 1e-10)

    print("Wrong format of the shapes, both should be n*1!")
    return np.float64(1)


def symmetric_mean_absolute_percentage_error_total(actual: np.array, forecast: np.array) -> np.float64:
    """
        This function measures the SMAPE total https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        The lower, the better.

        :param actual: ground truth with dimension n*1
        :param predicted: predicted from model with dimension n*1
        :return: returns the error in float, if there is a shape missmatch the error will be 1
    """

    if actual.shape == forecast.shape and len(forecast.shape) == 1:
        return np.abs(np.sum(forecast) - np.sum(actual)) /(np.sum(actual + forecast) + 1e-10)

    print("Wrong format of the shapes, both should be n*1!")
    return np.float64(1)

def accuracy(error: np.float64) -> float:
    """
        :param error: SMAPE error
        :return: returns accuracy in percent
    """
    return np.float64(100) - error*100

def directed_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    signs = np.sign(y_true) == np.sign(y_pred)
    return signs.sum() / len(signs)

def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series, do_calculate_directed_accuracy: bool) -> EvaluationResults:
    """
    Evaluates the predictions using MAE and RMSE metrics.
    
    Args:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values.
        """
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mse_val = mean_squared_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    forecasting_bias = calculate_forecast_bias(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true.values, y_pred.values)
    smape_total = symmetric_mean_absolute_percentage_error_total(y_true.values, y_pred.values)
    accuracy_ = accuracy(smape)
    accuracy_total = 100 - smape_total*100
    if do_calculate_directed_accuracy:
        directed_acc = directed_accuracy(y_true, y_pred)

    return EvaluationResults(mae=mae, rmse=rmse, 
                             mse=mse_val, r2=r2_val, 
                             forecast_bias=forecasting_bias, smape=smape, 
                             smape_total=smape_total, accuracy=accuracy_,
                               accuracy_total=accuracy_total, directed_accuracy=directed_acc if do_calculate_directed_accuracy else None)

def evaluate(df: pd.DataFrame, target_col: str, y_pred_col: str, do_calculate_directed_accuracy: bool = False) -> EvaluationResults:
    return evaluate_predictions(df[target_col], df[y_pred_col], do_calculate_directed_accuracy)