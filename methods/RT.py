import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from .model_wrappers import LGBMClassifierWrapper

class ReverseTraining:
    def __init__(
            self,
            y_pred_proba,
            y_pred,
            y_true,
            cont_features,
            cat_features,
            metrics,
            RT_estimator=LGBMClassifierWrapper,
            ):

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.features = cat_features + cont_features
        self.metrics = metrics
        self.RT_estimator = RT_estimator


    def fit(self, df_reference):
        reference_ypp = df_reference[self.y_pred_proba]
        reference_yp = df_reference[self.y_pred]
        reference_y_true = df_reference[self.y_true]
        reference_X = df_reference[self.features]

        self.reference_ypp = reference_ypp
        self.reference_yp = reference_yp
        self.reference_y_true = reference_y_true
        self.reference_X = reference_X


    def fit_RT_estimator(self, chunk_X, chunk_client_model_y_pred):
        RT_model = self.RT_estimator(categorical_features=self.cat_features, continuous_features=self.cont_features)
        RT_model.fit(chunk_X, chunk_client_model_y_pred)

        return RT_model


    def get_performance(self, y_true, y_pred, y_pred_proba, metric):

        if metric == 'auroc':
            metric_value = roc_auc_score(y_true, y_pred_proba)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            metric_value = f1_score(y_true, y_pred)

        return metric_value

    def estimate(self, chunk):

        chunk_X = chunk[self.features]
        chunk_client_model_y_pred = chunk[self.y_pred]
        RT_model = self.fit_RT_estimator(chunk_X, chunk_client_model_y_pred)

        reference_X = self.reference_X
        reversed_reference_y_pred = RT_model.predict(reference_X)
        reversed_reference_y_pred_proba = RT_model.predict_proba(reference_X)[:, 1]

        reference_y_true = self.reference_y_true


        values = []
        for metric in self.metrics:
            value = self.get_performance(reference_y_true,
                                     reversed_reference_y_pred,
                                     reversed_reference_y_pred_proba,
                                     metric)
            values.append(value)

        return values