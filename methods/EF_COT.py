from .COT import COT
import numpy as np


def calculate_COT_on_chunk(data, Y_PRED_PROBA_start):
    chunks = data['list_with_chunks']
    df_reference = data['df_reference']

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'

    rocs = []
    f1s = []
    acs = []

    estimator_accuracy = COT(CLIENT_MODEL_Y_PRED_PROBA, 'y_true')

    estimator_accuracy.fit(df_reference)

    for i, chunk in enumerate(chunks):
        print(i / len(chunks), end='\r')
           
        acc_est = estimator_accuracy.estimate(chunk)
        roc_auc = acc_est
        accuracy = acc_est
        f1 = acc_est

        rocs.append(roc_auc)
        acs.append(accuracy)
        f1s.append(f1)


    data['est_accuracy'] = acs
    data['est_f1'] = f1s
    data['est_roc'] = rocs
    data['zeros'] = np.zeros(len(chunks))

    return data