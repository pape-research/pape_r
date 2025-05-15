import abc
import datetime
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    f1_score,
    roc_auc_score, mean_squared_log_error, roc_curve, precision_score, recall_score,
)


def mse(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    return mean_squared_error(y_true, y_pred)


def rmsle(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """

    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if len(y_true) == 0:
        return np.nan
    else:
        try:
            return mean_squared_log_error(y_true, y_pred, squared=False)
        except:
            return np.nan


def mape(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if len(y_true) == 0:
        return np.nan
    else:
        return mean_absolute_percentage_error(y_true, y_pred)


def r2(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    return r2_score(y_true, y_pred)


def precision(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
        return np.nan
    else:
        return precision_score(y_true, y_pred)


def recall(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
        return np.nan
    else:
        return recall_score(y_true, y_pred)


def f1(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if (y_true.nunique() <= 1) or (y_pred.nunique() <= 1):
        return np.nan
    else:
        return f1_score(y_true, y_pred)


def roc_auc(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if y_true.nunique() <= 1:
        return np.nan
    else:
        return roc_auc_score(y_true, y_pred)


def roc_curve_custom(y_true, y_pred):
    """
    Redefine to handle NaNs and edge cases.
    """
    y_true, y_pred = (
        pd.Series(y_true).reset_index(drop=True),
        pd.Series(y_pred).reset_index(drop=True),
    )
    y_true = y_true[~y_pred.isna()]
    y_pred.dropna(inplace=True)

    if y_true.nunique() <= 1:
        return None, None, None
    else:
        return roc_curve(y_true, y_pred)


class MetricsLogger(abc.ABC):

    def start_logging(self, data: Dict) -> Dict:
        raise NotImplementedError

    def end_logging(self, data: Dict) -> Dict:
        raise NotImplementedError


