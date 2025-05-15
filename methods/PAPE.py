import sys
sys.path.append("..")
from sklearn.metrics import auc
import numpy as np
import pandas as pd
from .model_wrappers import LGBMClassifierWrapper
from lightgbm import LGBMRegressor



RANDOM_STATE=42

def get_expected_TP(true_y_pred_proba: pd.Series, model_y_pred: pd.Series) -> float:
    TP = np.where(model_y_pred == 1, true_y_pred_proba, 0)
    return np.sum(TP)

def get_expected_FP(true_y_pred_proba: pd.Series, model_y_pred: pd.Series) -> float:
    FP = np.where(model_y_pred == 1, 1-true_y_pred_proba, 0)
    return np.sum(FP)

def get_expected_TN(true_y_pred_proba: pd.Series, model_y_pred: pd.Series) -> float:
    TN = np.where(model_y_pred == 0, 1 - true_y_pred_proba, 0)
    return np.sum(TN)


def get_expected_FN(true_y_pred_proba: pd.Series, model_y_pred: pd.Series) -> float:
    FN = np.where(model_y_pred == 0, true_y_pred_proba, 0)
    return np.sum(FN)

def estimate_accuracy(true_y_pred_proba: pd.Series, model_y_pred: pd.Series) -> float:

    true_y_pred_proba = np.asarray(true_y_pred_proba)
    model_y_pred = np.asarray(model_y_pred)

    TP = get_expected_TP(true_y_pred_proba, model_y_pred)
    TN = get_expected_TN(true_y_pred_proba, model_y_pred)

    val = TP + TN

    metric = val/len(model_y_pred)

    return metric
    
    
def estimate_f1(true_y_pred_proba: pd.Series, model_y_pred: pd.Series) -> float:

    true_y_pred_proba = np.asarray(true_y_pred_proba)
    model_y_pred = np.asarray(model_y_pred)

    TP = get_expected_TP(true_y_pred_proba, model_y_pred)
    FN = get_expected_FN(true_y_pred_proba, model_y_pred)
    FP = get_expected_FP(true_y_pred_proba, model_y_pred)

    metric = TP / (TP + ((1 / 2) * (FP + FN)))

    return metric

def estimate_auroc(true_y_pred_proba: pd.Series, model_y_pred_proba: pd.Series) -> float:
    
    sorted_index = np.argsort(model_y_pred_proba)[::-1]
    model_y_pred_proba = model_y_pred_proba[sorted_index]
    true_y_pred_proba = true_y_pred_proba[sorted_index]

    tps = np.cumsum(true_y_pred_proba)
    fps = 1 + np.arange(len(true_y_pred_proba)) - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    tps = np.round(tps, 5)
    fps = np.round(fps, 5)

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    
    metric = auc(fpr, tpr)
    
    return metric







class PAPE:
    def __init__(
            self,
            y_pred_proba,
            y_pred,
            y_true,
            cont_features,
            cat_features,
            metrics,
            DRE_estimator=LGBMClassifierWrapper,
            calibrator_algo=LGBMRegressor,
            minimum_denominator=0.05,
            maximum_weight = 0.001
            ):

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.features = cat_features + cont_features
        self.metrics = metrics
        self.DRE_estimator = DRE_estimator
        self.calibrator_algo = calibrator_algo
        self.minimum_denominator = minimum_denominator
        self.maximum_weight = maximum_weight

    def fit(self, df_reference):
        reference_ypp = df_reference[self.y_pred_proba]
        reference_yp = df_reference[self.y_pred]
        reference_y_true = df_reference[self.y_true]
        reference_X = df_reference[self.features]

        self.reference_ypp = reference_ypp
        self.reference_yp = reference_yp
        self.reference_y_true = reference_y_true
        self.reference_X = reference_X


    def fit_DRE_model(self, chunk_X, reference_X):
        chunk_y = np.ones(len(chunk_X))
        reference_y = np.zeros(len(reference_X))

        X = pd.concat([reference_X, chunk_X]).reset_index(drop=True)
        y = np.concatenate([reference_y, chunk_y])

        DRE_model = self.DRE_estimator(categorical_features=self.cat_features, continuous_features=self.cont_features)
        DRE_model.fit(X, y)

        return DRE_model

    def get_weights(self, weight_y_pred_probas, size_chunk, size_reference):

        correcting_factor = size_reference/size_chunk
        numerator = weight_y_pred_probas
        denominator = np.maximum(self.minimum_denominator, 1-weight_y_pred_probas) # avoid infinity weights
        density_ratio = correcting_factor*numerator/denominator
        density_ratio = np.maximum(density_ratio, self.maximum_weight)# avoid 0 weights
        return density_ratio

    def get_chunk_weights_on_ref_data(self, DRE_model, reference_X, chunk_X):
        reference_DRE_probas = DRE_model.predict_proba(reference_X)[:, 1]
        chunk_weights_on_ref_data = self.get_weights(reference_DRE_probas, len(chunk_X), len(reference_X))
        return chunk_weights_on_ref_data


    def get_estimated_performance(self, true_y_pred_proba, client_y_pred_proba, client_y_pred, metric):
        if metric == 'auroc':
            metric_value = estimate_auroc(true_y_pred_proba, client_y_pred_proba)
        elif metric == 'accuracy':
            metric_value = estimate_accuracy(true_y_pred_proba, client_y_pred)
        elif metric == 'f1':
            metric_value = estimate_f1(true_y_pred_proba, client_y_pred)

        return metric_value

    def estimate(self, chunk):

        chunk_X = chunk[self.features]
        reference_X = self.reference_X
        DRE_model = self.fit_DRE_model(chunk_X, reference_X)
        chunk_weights_on_ref_data = self.get_chunk_weights_on_ref_data(DRE_model, reference_X, chunk_X)

        reference_y_true = np.asarray(self.reference_y_true)
        reference_y_pred_proba = np.asarray(self.reference_ypp)

        calibrator = self.calibrator_algo(random_state=RANDOM_STATE)
        calibrator.fit(np.asarray(reference_y_pred_proba).reshape(-1, 1), np.asarray(reference_y_true),
                       sample_weight=chunk_weights_on_ref_data)

        chunk_y_pred_proba = np.asarray(chunk[self.y_pred_proba])
        chunk_y_pred = np.asarray(chunk[self.y_pred])

        calibrated_ypp_chunk = calibrator.predict(chunk_y_pred_proba.reshape(-1, 1))
        calibrated_ypp_chunk = np.minimum(calibrated_ypp_chunk, 1)
        calibrated_ypp_chunk = np.maximum(calibrated_ypp_chunk, 0)
        
        
        values = []
        for metric in self.metrics:
            value = self.get_estimated_performance(calibrated_ypp_chunk, chunk_y_pred_proba, chunk_y_pred, metric)
            values.append(value)

        return values
        
        return value