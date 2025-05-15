import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from .model_wrappers import LGBMClassifierWrapper

class ImportanceWeighting:
    def __init__(
            self,
            y_pred_proba,
            y_pred,
            y_true,
            cont_features,
            cat_features,
            metrics,
            DRE_estimator=LGBMClassifierWrapper,
            ):

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.cont_features = cont_features
        self.cat_features = cat_features
        self.features = cat_features + cont_features
        self.metrics= metrics
        self.DRE_estimator = DRE_estimator

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

    def get_weights(self, weight_y_pred_probas, size_chunk, size_reference, minimum_denominator=0.05):

        correcting_factor = size_reference/size_chunk
        numerator = weight_y_pred_probas
        denominator = np.maximum(minimum_denominator, 1-weight_y_pred_probas)
        likelihood_ratio = correcting_factor*numerator/denominator
        likelihood_ratio = np.maximum(likelihood_ratio, 0.001)# avoid 0 weights
        return likelihood_ratio

    def get_chunk_weights_on_ref_data(self, DRE_model, reference_X, chunk_X):
        reference_DRE_probas = DRE_model.predict_proba(reference_X)[:, 1]
        chunk_weights_on_ref_data = self.get_weights(reference_DRE_probas, len(chunk_X), len(reference_X))
        return chunk_weights_on_ref_data


    def get_weighted_performance(self, chunk_weights_on_ref_data, y_true, y_pred, y_pred_proba, metric):

        if metric == 'auroc':
            metric_value = roc_auc_score(y_true, y_pred_proba, sample_weight=chunk_weights_on_ref_data)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_true, y_pred, sample_weight=chunk_weights_on_ref_data)
        elif metric == 'f1':
            metric_value = f1_score(y_true, y_pred, sample_weight=chunk_weights_on_ref_data)

        return metric_value

    def estimate(self, chunk):

        chunk_X = chunk[self.features]
        reference_X = self.reference_X
        DRE_model = self.fit_DRE_model(chunk_X, reference_X)
        chunk_weights_on_ref_data = self.get_chunk_weights_on_ref_data(DRE_model, reference_X, chunk_X)

        reference_y_true = self.reference_y_true
        reference_y_pred = self.reference_yp
        reference_y_pred_proba = self.reference_ypp


        values = []
        for metric in self.metrics:
            value = self.get_weighted_performance(chunk_weights_on_ref_data,
                                                  reference_y_true,
                                                  reference_y_pred,
                                                  reference_y_pred_proba,
                                                  metric)
            values.append(value)

        return values