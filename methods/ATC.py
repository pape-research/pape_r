import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


class ATC:
    def __init__(
        self,
        y_pred_proba,
        y_pred,
        y_true,
        metric,
        loss = 'max_confidence'):

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.metric = metric
        self.loss = loss

    def neg_entropy(self, y_pred_proba):
        neg_y_pred_proba = 1 - y_pred_proba
        epsilon = 1e-10
        neg_entropy = y_pred_proba*np.log(y_pred_proba+epsilon) + neg_y_pred_proba*np.log(neg_y_pred_proba+epsilon)
        return neg_entropy
    
    def max_conf(self, y_pred_proba):
        return np.maximum(y_pred_proba, 1-y_pred_proba)

    def get_threshold(self, scores, metric_value):
        sorted_scores = np.sort(scores)
        thr = sorted_scores[int((1-metric_value)*len(sorted_scores))]
        return thr  
        
    def get_threshold_for_metric(self, y_pred_proba, y_pred, y_true, metric, loss):
        if metric == 'auroc':
            metric_value = roc_auc_score(y_true, y_pred_proba)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            metric_value = f1_score(y_true, y_pred)

        if loss == 'entropy':
            self.scoring_function = self.neg_entropy
        elif loss == 'max_confidence':
            self.scoring_function = self.max_conf
  
        
        scores = self.scoring_function(y_pred_proba)
                  
        if metric_value == 0:
            thr = np.max(scores)
        elif metric_value == 1:
            thr = np.min(scores)
        else:
            thr = self.get_threshold(scores, metric_value)

        return thr
    
    def fit(self, df_reference):
        reference_ypp = df_reference[self.y_pred_proba]
        reference_yp = df_reference[self.y_pred]
        reference_y_true = df_reference[self.y_true]
        
        self.thr = self.get_threshold_for_metric(reference_ypp, reference_yp, reference_y_true, self.metric, self.loss)

    def estimate(self, chunk):
        
        y_pred_proba = chunk[self.y_pred_proba]
        scores = self.scoring_function(y_pred_proba)
        value = np.mean(scores>=self.thr)
    
        return value