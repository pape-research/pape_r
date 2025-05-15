from .ATC import ATC
import numpy as np


def calculate_ATC_on_chunk(data, Y_PRED_PROBA_start, loss='entropy'):
    chunks = data['list_with_chunks']
    df_reference = data['df_reference']

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'

    rocs = []
    f1s = []
    acs = []

    atc_roc_auc = ATC(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', 'auroc')
    atc_accuracy = ATC(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', 'accuracy')
    atc_f1 = ATC(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', 'f1')

    atc_roc_auc.fit(df_reference)
    atc_accuracy.fit(df_reference)
    atc_f1.fit(df_reference)

    for i, chunk in enumerate(chunks):
        print(i / len(chunks), end='\r')

        roc_auc = atc_roc_auc.estimate(chunk)
        accuracy = atc_accuracy.estimate(chunk)
        f1 = atc_f1.estimate(chunk)

        rocs.append(roc_auc)
        acs.append(accuracy)
        f1s.append(f1)


    data['est_accuracy'] = acs
    data['est_f1'] = f1s
    data['est_roc'] = rocs
    data['zeros'] = np.zeros(len(chunks))

    return data