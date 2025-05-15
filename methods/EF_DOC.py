from .DOC import DOC
import numpy as np


def calculate_DOC_on_chunk(data, Y_PRED_PROBA_start, sample_size, sampling):
    chunks = data['list_with_chunks']
    df_reference = data['df_reference']
    features = data['features_selected']

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'

    rocs = []
    f1s = []
    acs = []

    estimator_roc_auc = DOC(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED , 'y_true', features, 'auroc', sample_size, sampling)
    estimator_accuracy = DOC(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED , 'y_true', features, 'accuracy', sample_size, sampling)
    estimator_f1 = DOC(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED , 'y_true', features, 'f1', sample_size, sampling)

    estimator_roc_auc.fit(df_reference)
    estimator_accuracy.fit(df_reference)
    estimator_f1.fit(df_reference)

    for i, chunk in enumerate(chunks):
        print(i / len(chunks), end='\r')

        roc_auc = estimator_roc_auc.estimate(chunk)
        accuracy = estimator_accuracy.estimate(chunk)
        f1 = estimator_f1.estimate(chunk)

        rocs.append(roc_auc)
        acs.append(accuracy)
        f1s.append(f1)


    data['est_accuracy'] = acs
    data['est_f1'] = f1s
    data['est_roc'] = rocs
    data['zeros'] = np.zeros(len(chunks))

    return data