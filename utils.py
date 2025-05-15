# import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.models import FTTransformerConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models.common.heads import LinearHeadConfig


RANDOM_STATE = 3

class LogisticRegressionWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.classifier = LogisticRegression(random_state=RANDOM_STATE, max_iter=200)
        pipe = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ])

    def fit(self, X, y):
        X_transformed = self.column_transformer.fit_transform(X)
        self.classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.classifier.predict_proba(X_transformed)


class RandomForestClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.classifier = RandomForestClassifier(random_state=RANDOM_STATE)

        pipe = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ])

    def fit(self, X, y):
        X = self.column_transformer.fit_transform(X)
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        X = self.column_transformer.transform(X)
        return self.classifier.predict(X)

    def predict_proba(self, X):
        X = self.column_transformer.transform(X)
        return self.classifier.predict_proba(X)

class LGBMClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.lgbm_classifier = LGBMClassifier(random_state=RANDOM_STATE)
        self.column_transformer = None

    def fit(self, X, y, sample_weight=None):
        pipe = Pipeline(
            [('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]
        )
        self.column_transformer = ColumnTransformer(
            [('categorical', pipe, self.categorical_features),],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        X_transformed = self.column_transformer.fit_transform(X)
        features_out = list(self.column_transformer.get_feature_names_out())
        df_X_transformed = pd.DataFrame(X_transformed, columns = features_out)

        if sample_weight is None:
            self.lgbm_classifier.fit(
                df_X_transformed,
                y,
                feature_name=features_out,
                categorical_feature=self.categorical_features
            )
        else:
            self.lgbm_classifier.fit(
                df_X_transformed,
                y,
                sample_weight = sample_weight,
                feature_name=features_out,
                categorical_feature=self.categorical_features
            )

        self.features_out = features_out
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        df_X_transformed = pd.DataFrame(X_transformed, columns = self.features_out)
        return self.lgbm_classifier.predict(df_X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        df_X_transformed = pd.DataFrame(X_transformed, columns = self.features_out)
        return self.lgbm_classifier.predict_proba(df_X_transformed)


class XGBoostClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.classifier = XGBClassifier(tree_method="hist", enable_categorical=True)
        self.features_selected = self.categorical_features + self.continuous_features
        # self.column_transformer = None

    def fit(self, X: pd.DataFrame, y:pd.Series):
        self.classifier.fit(
            X[self.features_selected],
            y,
        )    
        return self

    def predict(self, X):
        return self.classifier.predict(X[self.features_selected])

    def predict_proba(self, X):
        return self.classifier.predict_proba(X[self.features_selected])


class FTTransformerClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        # self.features_selected = self.categorical_features + self.continuous_features

        trainer_config = TrainerConfig(
            auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
            batch_size=512,
            max_epochs=120,
            early_stopping="valid_loss",  # Monitor valid_loss for early stopping
            early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
            early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
            checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
            load_best=True,  # After training, load the best checkpoint
        )
        optimizer_config = OptimizerConfig()
        head_config = LinearHeadConfig(
            layers="",  # No additional layer in head, just a mapping layer to output_dim
            dropout=0.1,
            initialization="kaiming",
        ).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)
        model_config = FTTransformerConfig(
            task="classification",
            num_attn_blocks=3,
            num_heads=4,
            learning_rate=1e-3,
            head="LinearHead",  # Linear Head
            head_config=head_config,  # Linear Head Config
        )
        data_config = DataConfig(
            target=[
                "y_true"
            ],  # target should always be a list.
            continuous_cols=self.continuous_features,
            categorical_cols=self.categorical_features,
            validation_split=0.25
        )
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            verbose=False,
            suppress_lightning_logger=True,
        )
        self.classifier = tabular_model

    def fit(self, X: pd.DataFrame, y:pd.Series):
        data = pd.concat([X,y], axis=1)
        # ideally the astype(object) happens before data are split for consistency!
        data[self.categorical_features] = data[self.categorical_features].astype('object')
        self.classifier.fit(train=data)
        return self

    def predict(self, data: pd.DataFrame):
        data[self.categorical_features] = data[self.categorical_features].astype('object')
        preds = self.classifier.predict(data)
        return preds['prediction']

    def predict_proba(self, data: pd.DataFrame):
        data[self.categorical_features] = data[self.categorical_features].astype('object')
        preds = self.classifier.predict(data)
        return preds.iloc[:,:2].to_numpy()


US_STATE_LIST = [
    'AL',  # Alabama
    'AK',  # Alaska
    'AZ',  # Arizona
    'AR',  # Arkansas
    'CA',  # California
    'CO',  # Colorado
    'CT',  # Connecticut
    'DE',  # Delaware
    'FL',  # Florida
    'GA',  # Georgia
    'HI',  # Hawaii
    'ID',  # Idaho
    'IL',  # Illinois
    'IN',  # Indiana
    'IA',  # Iowa
    'KS',  # Kansas
    'KY',  # Kentucky
    'LA',  # Louisiana
    'ME',  # Maine
    'MD',  # Maryland
    'MA',  # Massachusetts
    'MI',  # Michigan
    'MN',  # Minnesota
    'MS',  # Mississippi
    'MO',  # Missouri
    'MT',  # Montana
    'NE',  # Nebraska
    'NV',  # Nevada
    'NH',  # New Hampshire
    'NJ',  # New Jersey
    'NM',  # New Mexico
    'NY',  # New York
    'NC',  # North Carolina
    'ND',  # North Dakota
    'OH',  # Ohio
    'OK',  # Oklahoma
    'OR',  # Oregon
    'PA',  # Pennsylvania
    'RI',  # Rhode Island
    'SC',  # South Carolina
    'SD',  # South Dakota
    'TN',  # Tennessee
    'TX',  # Texas
    'UT',  # Utah
    'VT',  # Vermont
    'VA',  # Virginia
    'WA',  # Washington
    'WV',  # West Virginia
    'WI',  # Wisconsin
    'WY',  # Wyoming
]


MODELS_LIST = [
    [LogisticRegressionWrapper, "LogisticRegression"],
    [LGBMClassifierWrapper, "LGBMClassifier"],
    [RandomForestClassifierWrapper, "RandomForestClassifier"],
    [XGBoostClassifierWrapper, "XGB"],
    [FTTransformerClassifierWrapper, "FT_Transformer"]
]
