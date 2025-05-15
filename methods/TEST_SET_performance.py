import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


class TEST_SET_performance:
    def __init__(
        self,
        y_pred_proba,
        y_pred,
        y_true,
        metric):

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.metric = metric


    def get_performance_for_reference(self, y_pred_proba, y_pred, y_true, metric):

        if metric == 'auroc':
            metric_value = roc_auc_score(y_true, y_pred_proba)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            metric_value = f1_score(y_true, y_pred)

        return metric_value


    def fit(self, df_reference):
        reference_ypp = df_reference[self.y_pred_proba]
        reference_yp = df_reference[self.y_pred]
        reference_y_true = df_reference[self.y_true]
        
        self.reference_performance = self.get_performance_for_reference(reference_ypp, reference_yp, reference_y_true, self.metric)

    def estimate(self, chunk):

        return self.reference_performance