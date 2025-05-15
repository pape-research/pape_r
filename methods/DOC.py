from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np

from sklearn.linear_model import LinearRegression


class DOC():

    def __init__(
            self,
            y_pred_proba,
            y_pred,
            y_true,
            features,
            metric,
            sample_size,
            sampling='random',
            n_experiments=200

    ):

        self.y_pred_proba = y_pred_proba
        self.y_pred = y_pred
        self.y_true = y_true
        self.features = features
        self.metric = metric
        self.sample_size = sample_size
        self.sampling = sampling
        self.n_experiments = n_experiments

    def get_confidences(self, y_pred_proba):
        confidences = np.where(y_pred_proba > 0.5, y_pred_proba, 1 - y_pred_proba)
        return confidences

    def get_performance_metric(self, y_pred_proba, y_pred, y_true, metric):

        if metric == 'auroc':
            metric_value = roc_auc_score(y_true, y_pred_proba)
        elif metric == 'accuracy':
            metric_value = accuracy_score(y_true, y_pred)
        elif metric == 'f1':
            metric_value = f1_score(y_true, y_pred)

        return metric_value

    def get_doc(self, reference_y_pred_proba, chunk_y_pred_proba):
        reference_confidence = self.get_confidences(reference_y_pred_proba)
        chunk_confidence = self.get_confidences(chunk_y_pred_proba)
        doc = np.mean(chunk_confidence) - np.mean(reference_confidence)
        return doc

    def get_data_for_regression_model_random(self, df_reference, reference_metric):

        sample_DOC = []
        sample_metric_diff = []

        for i in range(self.n_experiments):
            sample = df_reference.sample(n=self.sample_size)
            DOC = self.get_doc(df_reference[self.y_pred_proba], sample[self.y_pred_proba])
            sample_metric = self.get_performance_metric(
                sample[self.y_pred_proba],
                sample[self.y_pred],
                sample[self.y_true],
                self.metric
            )
            metric_diff = sample_metric - reference_metric

            sample_DOC.append(DOC)
            sample_metric_diff.append(metric_diff)

        return np.asarray(sample_DOC).reshape(-1, 1), np.asarray(sample_metric_diff)

    def get_data_for_regression_model_non_random(self, df_reference, reference_metric):

        sample_DOC = []
        sample_metric_diff = []

        for feature in self.features:
            for const in [-1, 1]:
                weights = np.max(df_reference[feature]) + const * abs(df_reference[feature].fillna(0)) + 1
                sample = df_reference.sample(n=self.sample_size, weights=weights)

                DOC = self.get_doc(df_reference[self.y_pred_proba], sample[self.y_pred_proba])
                sample_metric = self.get_performance_metric(
                    sample[self.y_pred_proba],
                    sample[self.y_pred],
                    sample[self.y_true],
                    self.metric
                )
                metric_diff = sample_metric - reference_metric

                sample_DOC.append(DOC)
                sample_metric_diff.append(metric_diff)

        return np.asarray(sample_DOC).reshape(-1, 1), np.asarray(sample_metric_diff)

    def get_data_for_regression_model(self, df_reference, reference_metric, sampling):

        if sampling == 'random':
            X, y = self.get_data_for_regression_model_random(df_reference, reference_metric)
        elif sampling == 'non-random':
            X, y = self.get_data_for_regression_model_non_random(df_reference, reference_metric)

        return X, y

    def fit(self, df_reference):
        reference_ypp = df_reference[self.y_pred_proba]
        reference_yp = df_reference[self.y_pred]
        reference_y_true = df_reference[self.y_true]

        reference_metric = self.get_performance_metric(reference_ypp, reference_yp, reference_y_true, self.metric)

        DOCs, metric_change = self.get_data_for_regression_model(df_reference, reference_metric, self.sampling)

        DOC_model = LinearRegression()
        DOC_model.fit(DOCs, metric_change)

        self.DOC_model = DOC_model
        self.reference_ypp = reference_ypp
        self.reference_metric = reference_metric

    def estimate(self, chunk):

        chunk_DOC = self.get_doc(self.reference_ypp, chunk[self.y_pred_proba])
        est_metric_diff = self.DOC_model.predict(np.asarray(chunk_DOC).reshape(-1, 1))[0]
        est_metric = self.reference_metric + est_metric_diff

        return est_metric
