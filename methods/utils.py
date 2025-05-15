import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


ORDER = ['TEST_SET',
 'RT-mod',
 'ATC',
 'DOC',
 'CBPE',
 'IW',
 'PAPE',
 ] # order for tables

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00'] # colors from color-blind cycle

colors = {
    'PAPE':CB_color_cycle[7],
    'CBPE':CB_color_cycle[5],
    'IW':CB_color_cycle[0],
    'ATC':CB_color_cycle[1],
    'DOC':CB_color_cycle[2],
    'TEST_SET':CB_color_cycle[4],
    "RT-mod":CB_color_cycle[3]
}



METRIC_MAPPER = {
    'accuracy': 'Accuracy',
    'roc_auc': 'AUROC',
    'f1': 'F1'
}


def get_chunks(df, chunksize=2000, step=2000):
    copy_df = df.reset_index(drop=True).copy()
    step = 2000
    chunksize = 2000
    n_chunks = len(copy_df)//step
    chunks = [copy_df.loc[i*chunksize: (i+1)*(chunksize)-1] for i in range(n_chunks)]
    
    return chunks
    
    
def calculate_targets(data, Y_PRED_PROBA_start):
    chunks = data['list_with_chunks']
    
    rocs = []
    accs = []
    f1s = []
    
    for chunk in chunks:
        rocs.append(roc_auc_score(chunk['y_true'], chunk[Y_PRED_PROBA_start + 'y_pred_proba']))
        accs.append(accuracy_score(chunk['y_true'], chunk[Y_PRED_PROBA_start + 'y_pred']))
        f1s.append(f1_score(chunk['y_true'], chunk[Y_PRED_PROBA_start + 'y_pred']))
        
    data['roc'] = rocs
    data['accuracy'] = accs
    data['f1'] = f1s
    
    return data
    
    
def save_results_to_df(dat, method, monitored_model):
    res_data = dat[0]
    df = pd.DataFrame(index=range(len(res_data['accuracy'])))

    df['method'] = method
    df['dataset'] = res_data['dataset_name']
    df['monitored_model'] = monitored_model
    
    df['accuracy'] = res_data['accuracy']
    df['est_accuracy'] = res_data['est_accuracy']
    df['roc_auc'] = res_data['roc']
    df['est_roc_auc'] = res_data['est_roc']
    df['f1'] = res_data['f1']
    df['est_f1'] = res_data['est_f1']

    n_ref_chunks = res_data['n_reference_chunks'] 
    n_transition_chunks = res_data['n_transition_chunks']

    df.loc[:n_ref_chunks-1, 'period'] = 'reference'
    df.loc[n_ref_chunks-1 + n_transition_chunks, 'period'] = 'transition'
    df['period'] = df['period'].fillna('production')
    df['chunksize'] = res_data['observations_in_chunk']
    df['step'] = res_data['step_size']
    
    
    return df
    