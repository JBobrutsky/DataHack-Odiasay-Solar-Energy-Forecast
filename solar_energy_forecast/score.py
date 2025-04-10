from typing import Callable

import numpy as np
import pandas as pd
from solar_energy_forecast.preprocess import get_data


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE)"""
    return np.mean(np.abs((y_true - y_pred) / y_true + 1e-10)) * 100

def dummy_predict(df: pd.DataFrame) -> np.ndarray:
    """Dummy prediction using the mean of the training set
    returns vector of the same length as the input dataframe
    """
    return np.full(df.shape[0], df["relative_power"].mean())

def zero_predict(df: pd.DataFrame) -> np.ndarray:
    """Zero prediction trivially
    returns vector of the same length as the input dataframe
    """
    return np.full(df.shape[0], 0)

def score(predict_fun: Callable) -> float:
    """Calculate and return the MAPE score for predicted values in 2018"""
    df = get_data()
    df["year"] = df.index.year
    df["pred"] = predict_fun(df)
    df_2018 = df[df["year"] == 2018]
    return mape(df_2018["relative_power"].to_numpy(), df_2018["pred"].to_numpy())

