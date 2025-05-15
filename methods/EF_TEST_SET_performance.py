from .TEST_SET_performance import TEST_SET_performance
import numpy as np


def calculate_TEST_SET_performance_on_chunk(data, Y_PRED_PROBA_start):
    chunks = data['list_with_chunks']
    df_reference = data['df_reference']

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'
    print("EVALUTING FOR:")
    print(Y_PRED_PROBA_start)

    rocs = []
    f1s = []
    acs = []

    estimator_auroc = TEST_SET_performance(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', 'auroc')
    estimator_accuracy = TEST_SET_performance(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', 'accuracy')
    estimator_f1 = TEST_SET_performance(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', 'f1')

    estimator_auroc.fit(df_reference)
    estimator_accuracy.fit(df_reference)
    estimator_f1.fit(df_reference)

    for i, chunk in enumerate(chunks):
        print(i / len(chunks), end='\r')

        auroc = estimator_auroc.estimate(chunk)
        accuracy = estimator_accuracy.estimate(chunk)
        f1 = estimator_f1.estimate(chunk)

        rocs.append(auroc)
        acs.append(accuracy)
        f1s.append(f1)


    data['est_accuracy'] = acs
    data['est_f1'] = f1s
    data['est_roc'] = rocs
    data['zeros'] = np.zeros(len(chunks))

    return data