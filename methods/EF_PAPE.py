from .PAPE import PAPE
import numpy as np



def calculate_PAPE_on_chunk(data, Y_PRED_PROBA_start):
    chunks = data['list_with_chunks']
    df_reference = data['df_reference']
    features = data['features_selected']
    cat_features = data['features_categorical']
    cont_features = [x for x in features if x not in cat_features]

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'

    rocs = []
    f1s = []
    acs = []

    estimator = PAPE(CLIENT_MODEL_Y_PRED_PROBA, CLIENT_MODEL_PRED, 'y_true', cont_features, cat_features,
                      ['auroc', 'accuracy', 'f1'])

    estimator.fit(df_reference)

    for i, chunk in enumerate(chunks):
        print(i / len(chunks), end='\r')

        res = estimator.estimate(chunk)
        roc_auc, accuracy, f1 = res[0], res[1], res[2]
        rocs.append(roc_auc)
        acs.append(accuracy)
        f1s.append(f1)


    data['est_accuracy'] = acs
    data['est_f1'] = f1s
    data['est_roc'] = rocs
    data['zeros'] = np.zeros(len(chunks))

    return data