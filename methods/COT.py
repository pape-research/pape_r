from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.isotonic import IsotonicRegression



class COT():
    """
    Based on https://github.com/luyuzhe111/COT which is the official implementation of NeurIPS 2023 paper Charaterizing Out-of-distribution Error via Optimal Transport.
    """

    def __init__(
            self,
            y_pred_proba,
            y_true,
    ):

        self.y_pred_proba = y_pred_proba
        self.y_true = y_true
        
   
    def fit(self, df_reference):
        reference_ypp = df_reference[self.y_pred_proba].values
        reference_y_true = df_reference[self.y_true].values

        calibrator = IsotonicRegression(increasing=True, out_of_bounds='clip')
        calibrator.fit(reference_ypp.reshape(-1,1), reference_y_true)
        self.calibrator = calibrator
        self.ref_y_true = reference_y_true

    def estimate(self, chunk):

        chunk_ypp = chunk[self.y_pred_proba]
        ypp = self.calibrator.predict(chunk_ypp)
        error = wasserstein_distance(ypp, self.ref_y_true)
        accuracy = 1-error
        
        return accuracy
