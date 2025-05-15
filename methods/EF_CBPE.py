import nannyml as nml
import numpy as np


def calculate_CBPE_on_chunk(data, Y_PRED_PROBA_start, ):
    chunks = data['list_with_chunks']
    features = data['features_selected']
    cat_features = data['features_categorical']
    cont_features = [x for x in features if x not in cat_features]
    df_reference = data['df_reference']

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'

    estimator = nml.CBPE(
        y_pred_proba=CLIENT_MODEL_Y_PRED_PROBA,
        y_pred=CLIENT_MODEL_PRED,
        y_true='y_true',
        metrics=['roc_auc', 'accuracy', 'f1'],
        chunk_number=1,
        problem_type='classification_binary',
    )

    estimator.fit(df_reference)

    rocs = []
    f1s = []
    acs = []

    for i, chunk in enumerate(chunks):
        print(i / len(chunks), end='\r')

        res = estimator.estimate(chunk)
        chunk_res = res.filter(period='analysis').to_df()
        accuracy = chunk_res['accuracy']['value'][0]
        f1 = chunk_res['f1']['value'][0]
        roc_auc = chunk_res['roc_auc']['value'][0]

        acs.append(accuracy)
        f1s.append(f1)
        rocs.append(roc_auc)

    data['est_accuracy'] = acs
    data['est_f1'] = f1s
    data['est_roc'] = rocs
    data['zeros'] = np.zeros(len(chunks))
    
    return data