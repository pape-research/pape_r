import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_std_on_dataset(data, Y_PRED_PROBA_start, chunk_size=2000, n_experiments=100):
    chunks = data['list_with_chunks']
    df_reference = data['df_reference']

    CLIENT_MODEL_PRED = Y_PRED_PROBA_start + 'y_pred'
    CLIENT_MODEL_Y_PRED_PROBA = Y_PRED_PROBA_start + 'y_pred_proba'

   
    rocs = []
    f1s = []
    acs = []
      
      
    for _ in range(n_experiments):
        sample = df_reference.sample(chunk_size)
        rocs.append(roc_auc_score(sample['y_true'], sample[CLIENT_MODEL_Y_PRED_PROBA]))
        f1s.append(f1_score(sample['y_true'], sample[CLIENT_MODEL_PRED]))
        acs.append(accuracy_score(sample['y_true'], sample[CLIENT_MODEL_PRED]))
        


    data['std_accuracy'] = np.std(acs)
    data['std_f1'] = np.std(f1s)
    data['std_roc'] = np.std(rocs)
    
    return data