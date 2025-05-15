from pathlib import Path
import pandas as pd
from folktables import (
    ACSDataSource, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSIncome, ACSTravelTime
)
from utils import MODELS_LIST, US_STATE_LIST
from json import dump as jdump
import warnings
warnings.filterwarnings("ignore")

def get_employment_state_data_for_year(state_list, year: int):
    data_source = ACSDataSource(
        survey_year=year, horizon='1-Year', survey='person'
    )
    ydata = data_source.get_data(states=state_list, download=True)
    features, labels, _ = ACSEmployment.df_to_numpy(ydata)
    df = pd.DataFrame(features)
    df.columns = ACSEmployment.features
    df[ACSEmployment.target] = labels.astype('int')
    df = df.rename(columns={ACSEmployment.target: 'y_true'})
    return df


def create_employment_dataset(state_list, year: int, model_list=MODELS_LIST):

    train = get_employment_state_data_for_year(state_list, year)
    reference = get_employment_state_data_for_year(state_list, year+1)
    analysis1 = get_employment_state_data_for_year(state_list, year+2)
    analysis2 = get_employment_state_data_for_year(state_list, year+3)
    analysis = pd.concat([analysis1, analysis2]).reset_index(drop=True)
    del analysis1, analysis2

    # List of columns used by Folktables, for column descriptions look at:
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict15.pdf
    categorical_features = [
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'SEX',
        'RAC1P'
    ]
    numeric_features = ['AGEP']
    features = numeric_features + categorical_features
    target_col = 'y_true'

    for model in model_list:
        print(model[1])
        client_model = model[0](categorical_features,numeric_features)
        client_model.fit(train[features], train[target_col])

        train[model[1]+'_y_pred'] = client_model.predict(train[features])
        train[model[1]+'_y_pred_proba'] = client_model.predict_proba(train[features])[:,1]

        reference[model[1]+'_y_pred'] = client_model.predict(reference[features])
        reference[model[1]+'_y_pred_proba'] = client_model.predict_proba(reference[features])[:,1]

        analysis[model[1]+'_y_pred'] = client_model.predict(analysis[features])
        analysis[model[1]+'_y_pred_proba'] = client_model.predict_proba(analysis[features])[:,1]

    train['partition'] = 'train'
    reference['partition'] = 'reference'
    analysis['partition'] = 'production'

    data = pd.concat([train, reference, analysis], axis=0, ignore_index=True)

    years_list = [str(year), str(year+1), str(year+2), str(year+3)]
    years_names = '_'.join(years_list)
    states_names = '_'.join(state_list)
    name = f"employment-{years_names}-{states_names}.pq"
    print(f"Storing: {name}")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    data.to_parquet(fileloc, index=False)
    metadata_dict = {
        "data_type": "real",
        "features_selected": features,
        "features_categorical": categorical_features,
        "ml_problem": "binary_classification",
        "n_rows": data.shape[0],
        "n_cols": len(features),
        "partitions": list(data["partition"].unique()),
        "missing_values": False,
        "target_desc": "Is the person employed or not",
        "model_type": "LGBMClassifier+LogisticRegression+RandomForestClassifier+XGB+FT_Transformer",
        "notes": f"FolkTables Employment Problem from {years_names} and {states_names}.",
    }
    name = name.replace(".pq", "_metadata.json")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    with open(fileloc, "w") as fh:
        jdump(metadata_dict, fh)


def get_medicalcov_state_data_for_year(state_list, year: int):
    data_source = ACSDataSource(
        survey_year=year, horizon='1-Year', survey='person'
    )
    ydata = data_source.get_data(states=state_list, download=True)
    features, labels, _ = ACSPublicCoverage.df_to_numpy(ydata)
    df = pd.DataFrame(features)
    df.columns = ACSPublicCoverage.features
    df[ACSPublicCoverage.target] = labels.astype('int')
    df = df.rename(columns={ACSPublicCoverage.target: 'y_true'})
    return df


def create_medicalcov_dataset(state_list, year: int, model_list=MODELS_LIST):

    train = get_medicalcov_state_data_for_year(state_list, year)
    reference = get_medicalcov_state_data_for_year(state_list, year+1)
    analysis1 = get_medicalcov_state_data_for_year(state_list, year+2)
    analysis2 = get_medicalcov_state_data_for_year(state_list, year+3)
    analysis = pd.concat([analysis1, analysis2]).reset_index(drop=True)
    del analysis1, analysis2

    # List of columns used by Folktables, for column descriptions look at:
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict15.pdf
    categorical_features = [
        'SCHL',
        'MAR',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'SEX',
        'RAC1P',
        'ESR', # Employment Status Recode
        'FER', # Gave birth to child within the past 12 months
        ]
    numeric_features = [
        'AGEP',
        'PINCP',# : numeric (personal income) //with string for NA//
    ]
    features = numeric_features + categorical_features
    target_col = 'y_true'

    for model in model_list:
        print(model[1])
        client_model = model[0](categorical_features,numeric_features)
        client_model.fit(train[features], train[target_col])

        train[model[1]+'_y_pred'] = client_model.predict(train[features])
        train[model[1]+'_y_pred_proba'] = client_model.predict_proba(train[features])[:,1]

        reference[model[1]+'_y_pred'] = client_model.predict(reference[features])
        reference[model[1]+'_y_pred_proba'] = client_model.predict_proba(reference[features])[:,1]

        analysis[model[1]+'_y_pred'] = client_model.predict(analysis[features])
        analysis[model[1]+'_y_pred_proba'] = client_model.predict_proba(analysis[features])[:,1]

    train['partition'] = 'train'
    reference['partition'] = 'reference'
    analysis['partition'] = 'production'

    data = pd.concat([train, reference, analysis], axis=0, ignore_index=True)

    years_list = [str(year), str(year+1), str(year+2), str(year+3)]
    years_names = '_'.join(years_list)
    states_names = '_'.join(state_list)
    name = f"medicalcov-{years_names}-{states_names}.pq"
    print(f"Storing: {name}")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    data.to_parquet(fileloc, index=False)
    metadata_dict = {
        "data_type": "real",
        "features_selected": features,
        "features_categorical": categorical_features,
        "ml_problem": "binary_classification",
        "n_rows": data.shape[0],
        "n_cols": len(features),
        "partitions": list(data["partition"].unique()),
        "missing_values": False,
        "target_desc": "Does the person have medical coverage or not",
        "model_type": "LGBMClassifier+LogisticRegression+RandomForestClassifier+XGB+FT_Transformer",
        "notes": f"FolkTables Medical Coverage Problem from {years_names} and {states_names}.",
    }
    name = name.replace(".pq", "_metadata.json")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    with open(fileloc, "w") as fh:
        jdump(metadata_dict, fh)


def get_mobility_state_data_for_year(state_list, year: int):
    data_source = ACSDataSource(
        survey_year=year, horizon='1-Year', survey='person'
    )
    ydata = data_source.get_data(states=state_list, download=True)
    features, labels, _ = ACSMobility.df_to_numpy(ydata)
    df = pd.DataFrame(features)
    df.columns = ACSMobility.features
    df[ACSMobility.target] = labels.astype('int')
    df = df.rename(columns={ACSMobility.target: 'y_true'})
    # df['year'] = year
    return df


def create_mobility_dataset(state_list, year: int, model_list=MODELS_LIST):

    train = get_mobility_state_data_for_year(state_list, year)
    reference = get_mobility_state_data_for_year(state_list, year+1)
    analysis1 = get_mobility_state_data_for_year(state_list, year+2)
    analysis2 = get_mobility_state_data_for_year(state_list, year+3)
    analysis = pd.concat([analysis1, analysis2]).reset_index(drop=True)
    del analysis1, analysis2

    # List of columns used by Folktables, for column descriptions look at:
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict15.pdf
    categorical_features = [
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIL',
        'ANC',
        'NATIVITY',
        'DEAR',
        'DEYE',
        'DREM',
        'SEX',
        'RAC1P',
        'ESR', # Employment Status Recode
        'GCL', # Grandparents living with grandchildren
        'COW', # Class of worker
        'WKHP', # Usual hours worked per week past 12 months
        'JWMNP', # Travel time to work
        ]
    numeric_features = [
        'AGEP',
        'PINCP',# : numeric (personal income) //with string for NA//
    ]
    features = numeric_features + categorical_features
    target_col = 'y_true'

    for model in model_list:
        print(model[1])
        client_model = model[0](categorical_features,numeric_features)
        client_model.fit(train[features], train[target_col])

        train[model[1]+'_y_pred'] = client_model.predict(train[features])
        train[model[1]+'_y_pred_proba'] = client_model.predict_proba(train[features])[:,1]

        reference[model[1]+'_y_pred'] = client_model.predict(reference[features])
        reference[model[1]+'_y_pred_proba'] = client_model.predict_proba(reference[features])[:,1]

        analysis[model[1]+'_y_pred'] = client_model.predict(analysis[features])
        analysis[model[1]+'_y_pred_proba'] = client_model.predict_proba(analysis[features])[:,1]

    train['partition'] = 'train'
    reference['partition'] = 'reference'
    analysis['partition'] = 'production'

    data = pd.concat([train, reference, analysis], axis=0, ignore_index=True)

    years_list = [str(year), str(year+1), str(year+2), str(year+3)]
    years_names = '_'.join(years_list)
    states_names = '_'.join(state_list)
    name = f"mobility-{years_names}-{states_names}.pq"
    print(f"Storing: {name}")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    data.to_parquet(fileloc, index=False)
    metadata_dict = {
        "data_type": "real",
        "features_selected": features,
        "features_categorical": categorical_features,
        "ml_problem": "binary_classification",
        "n_rows": data.shape[0],
        "n_cols": len(features),
        "partitions": list(data["partition"].unique()),
        "missing_values": False,
        "target_desc": "Did the person move not move for more than a year?",
        "model_type": "LGBMClassifier+LogisticRegression+RandomForestClassifier+XGB+FT_Transformer",
        "notes": f"FolkTables Mobility Problem from {years_names} and {states_names}.",
    }
    name = name.replace(".pq", "_metadata.json")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    with open(fileloc, "w") as fh:
        jdump(metadata_dict, fh)


def get_income_state_data_for_year(state_list, year: int):
    data_source = ACSDataSource(
        survey_year=year, horizon='1-Year', survey='person'
    )
    ydata = data_source.get_data(states=state_list, download=True)
    features, labels, _ = ACSIncome.df_to_numpy(ydata)
    df = pd.DataFrame(features)
    df.columns = ACSIncome.features
    df[ACSIncome.target] = labels.astype('int')
    df = df.rename(columns={ACSIncome.target: 'y_true'})
    # df['year'] = year
    return df


def create_income_dataset(state_list, year: int, model_list=MODELS_LIST):

    train = get_income_state_data_for_year(state_list, year)
    reference = get_income_state_data_for_year(state_list, year+1)
    analysis1 = get_income_state_data_for_year(state_list, year+2)
    analysis2 = get_income_state_data_for_year(state_list, year+3)
    analysis = pd.concat([analysis1, analysis2]).reset_index(drop=True)
    del analysis1, analysis2

    # List of columns used by Folktables, for column descriptions look at:
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict15.pdf
    categorical_features = [
        'OCCP', # occuparional records
        'SCHL',
        'MAR',
        'RELP',
        'SEX',
        'RAC1P',
        'POBP', # Place of Birth
        'COW', # Class of worker
        'WKHP', # Usual hours worked per week past 12 months
        ]
    numeric_features = [
        'AGEP',
    ]
    features = numeric_features + categorical_features
    target_col = 'y_true'

    for model in model_list:
        print(model[1])
        client_model = model[0](categorical_features,numeric_features)
        client_model.fit(train[features], train[target_col])

        train[model[1]+'_y_pred'] = client_model.predict(train[features])
        train[model[1]+'_y_pred_proba'] = client_model.predict_proba(train[features])[:,1]

        reference[model[1]+'_y_pred'] = client_model.predict(reference[features])
        reference[model[1]+'_y_pred_proba'] = client_model.predict_proba(reference[features])[:,1]

        analysis[model[1]+'_y_pred'] = client_model.predict(analysis[features])
        analysis[model[1]+'_y_pred_proba'] = client_model.predict_proba(analysis[features])[:,1]

    train['partition'] = 'train'
    reference['partition'] = 'reference'
    analysis['partition'] = 'production'

    data = pd.concat([train, reference, analysis], axis=0, ignore_index=True)

    years_list = [str(year), str(year+1), str(year+2), str(year+3)]
    years_names = '_'.join(years_list)
    states_names = '_'.join(state_list)
    name = f"income-{years_names}-{states_names}.pq"
    print(f"Storing: {name}")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    data.to_parquet(fileloc, index=False)
    metadata_dict = {
        "data_type": "real",
        "features_selected": features,
        "features_categorical": categorical_features,
        "ml_problem": "binary_classification",
        "n_rows": data.shape[0],
        "n_cols": len(features),
        "partitions": list(data["partition"].unique()),
        "missing_values": False,
        "target_desc": "Is a person's income above $50,000?",
        "model_type": "LGBMClassifier+LogisticRegression+RandomForestClassifier+XGB+FT_Transformer",
        "notes": f"FolkTables Income Problem from {years_names} and {states_names}.",
    }
    name = name.replace(".pq", "_metadata.json")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    with open(fileloc, "w") as fh:
        jdump(metadata_dict, fh)


def get_traveltime_state_data_for_year(state_list, year: int):
    data_source = ACSDataSource(
        survey_year=year, horizon='1-Year', survey='person'
    )
    ydata = data_source.get_data(states=state_list, download=True)
    features, labels, _ = ACSTravelTime.df_to_numpy(ydata)
    df = pd.DataFrame(features)
    df.columns = ACSTravelTime.features
    df[ACSTravelTime.target] = labels.astype('int')
    df = df.rename(columns={ACSTravelTime.target: 'y_true'})
    return df


def create_traveltime_dataset(state_list, year: int, model_list=MODELS_LIST):

    train = get_traveltime_state_data_for_year(state_list, year)
    reference = get_traveltime_state_data_for_year(state_list, year+1)
    analysis1 = get_traveltime_state_data_for_year(state_list, year+2)
    analysis2 = get_traveltime_state_data_for_year(state_list, year+3)
    analysis = pd.concat([analysis1, analysis2]).reset_index(drop=True)
    del analysis1, analysis2

    # List of columns used by Folktables, for column descriptions look at:
    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMSDataDict15.pdf
    categorical_features = [
        'OCCP', # occuparional records
        'SCHL',
        'MAR',
        'RELP',
        'DIS',
        'ESP',
        'CIT',
        'MIG',
        'SEX',
        'RAC1P',
        'ST', # State Numeric Code
        'PUMA', # Public use microdata area code (PUMA)
        'POWPUMA', # Place of work PUMA based
        'JWTR', # Means of transportation to work
        'POVPIP', # Income-to-poverty ratio recode
    ]
    numeric_features = [
        'AGEP',
    ]
    features = numeric_features + categorical_features
    target_col = 'y_true'

    for model in model_list:
        print(model[1])
        client_model = model[0](categorical_features,numeric_features)
        client_model.fit(train[features], train[target_col])

        train[model[1]+'_y_pred'] = client_model.predict(train[features])
        train[model[1]+'_y_pred_proba'] = client_model.predict_proba(train[features])[:,1]

        reference[model[1]+'_y_pred'] = client_model.predict(reference[features])
        reference[model[1]+'_y_pred_proba'] = client_model.predict_proba(reference[features])[:,1]

        analysis[model[1]+'_y_pred'] = client_model.predict(analysis[features])
        analysis[model[1]+'_y_pred_proba'] = client_model.predict_proba(analysis[features])[:,1]

    train['partition'] = 'train'
    reference['partition'] = 'reference'
    analysis['partition'] = 'production'

    data = pd.concat([train, reference, analysis], axis=0, ignore_index=True)

    years_list = [str(year), str(year+1), str(year+2), str(year+3)]
    years_names = '_'.join(years_list)
    states_names = '_'.join(state_list)
    name = f"traveltime-{years_names}-{states_names}.pq"
    print(f"Storing: {name}")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    data.to_parquet(fileloc, index=False)
    metadata_dict = {
        "data_type": "real",
        "features_selected": features,
        "features_categorical": categorical_features,
        "ml_problem": "binary_classification",
        "n_rows": data.shape[0],
        "n_cols": len(features),
        "partitions": list(data["partition"].unique()),
        "missing_values": False,
        "target_desc": "Is the person's commute more than 20m?",
        "model_type": "LGBMClassifier+LogisticRegression+RandomForestClassifier+XGB+FT_Transformer",
        "notes": f"FolkTables Traveltime Problem from {years_names} and {states_names}.",
    }
    name = name.replace(".pq", "_metadata.json")
    fileloc = Path(__file__).resolve().parent.joinpath("datasets/"+name)
    with open(fileloc, "w") as fh:
        jdump(metadata_dict, fh)

### Use functions above to create the datasets

# create_employment_dataset(["AL"], 2015)
# create_medicalcov_dataset(["AL"], 2015)
# create_mobility_dataset(["AL"], 2015)
# create_income_dataset(["AL"], 2015)
# create_traveltime_dataset(["AL"], 2015)

for state in US_STATE_LIST:
    create_employment_dataset([state], 2015)
    create_medicalcov_dataset([state], 2015)
    create_mobility_dataset([state], 2015)
    create_income_dataset([state], 2015)
    create_traveltime_dataset([state], 2015)
    # break
print("datasets created")
