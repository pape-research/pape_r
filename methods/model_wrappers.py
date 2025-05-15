import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RANDOM_STATE=42

class SGDWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.logistic_regression = None
        self.column_transformer = None

    def fit(self, X, y):
        
        pipe = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
    
    
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features), 


        ], remainder='passthrough')

        X_transformed = self.column_transformer.fit_transform(X)
        self.logistic_regression = SGDClassifier(loss='log_loss')
        self.logistic_regression.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.logistic_regression.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.logistic_regression.predict_proba(X_transformed)



class LogisticRegressionWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.logistic_regression = None
        self.column_transformer = None

    def fit(self, X, y):
        
        pipe = Pipeline([
                ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
    
    
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
            

        ], remainder='passthrough')

        X_transformed = self.column_transformer.fit_transform(X)
        self.logistic_regression = LogisticRegression()
        self.logistic_regression.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.logistic_regression.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.logistic_regression.predict_proba(X_transformed)


import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV


class LGBMClassifierWrapperCalibrated(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.lgbm_classifier = None
        self.column_transformer = None

    def fit(self, X, y, sample_weight=None):
        pipe = Pipeline([
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.column_transformer = ColumnTransformer(
            [('categorical', pipe, self.categorical_features),],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        X_transformed = self.column_transformer.fit_transform(X)
        base_clf = lgb.LGBMClassifier()
        self.lgbm_classifier = CalibratedClassifierCV(base_clf, cv=5, method='isotonic', n_jobs=-1)
        
        if sample_weight is None:
            self.lgbm_classifier.fit(X_transformed, y)
        else:
            self.lgbm_classifier.fit(X_transformed, y, sample_weight=sample_weight)


        return self


    def predict(self, X):
 
        return self.lgbm_classifier.predict(X)

    def predict_proba(self, X):
        return self.lgbm_classifier.predict_proba(X)


class LGBMClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.lgbm_classifier = None
        self.column_transformer = None

    def fit(self, X, y, sample_weight=None):
    
    
        pipe = Pipeline([
                ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
    
        self.column_transformer = ColumnTransformer([
            ('categorical', pipe, self.categorical_features),
        ], remainder='passthrough', verbose_feature_names_out=False)

        X_transformed = self.column_transformer.fit_transform(X)
        
        features_out = list(self.column_transformer.get_feature_names_out())
                
        df_X_transformed = pd.DataFrame(X_transformed, columns = features_out)
        
        self.lgbm_classifier = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        
        if sample_weight is None:
            self.lgbm_classifier.fit(X_transformed, y, 
                                     feature_name=features_out,  categorical_feature=self.categorical_features)
        else:
            self.lgbm_classifier.fit(X_transformed, y, sample_weight = sample_weight,
                                    feature_name=features_out,  categorical_feature=self.categorical_features)
        
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

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class RandomForestClassifierWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_features, continuous_features):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.random_forest_classifier = None
        self.column_transformer = None

    def fit(self, X, y):
    
        pipe = Pipeline([
            ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
            ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
    
    
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ], remainder='passthrough')

        X_transformed = self.column_transformer.fit_transform(X)
        self.random_forest_classifier = RandomForestClassifier()
        self.random_forest_classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.random_forest_classifier.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.random_forest_classifier.predict_proba(X_transformed)


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class SVCWrapperRBF(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, continuous_features, kernel='rbf'):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.svc_classifier = None
        self.column_transformer = None
        self.kernel = kernel

    def fit(self, X, y):
    
    
        pipe = Pipeline([
            ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
    
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ], remainder='passthrough')
        
        
        X_transformed = self.column_transformer.fit_transform(X)
        self.svc_classifier = SVC(probability=True, kernel=self.kernel)
        self.svc_classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.svc_classifier.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.svc_classifier.predict_proba(X_transformed)
        
        
class SVCWrapperPoly(BaseEstimator, TransformerMixin):

    def __init__(self, categorical_features, continuous_features, kernel='poly'):
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        self.svc_classifier = None
        self.column_transformer = None
        self.kernel = kernel

    def fit(self, X, y):
    
    
        pipe = Pipeline([
            ('imputer_categorical', SimpleImputer(strategy='most_frequent')),
            ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
    
        self.column_transformer = ColumnTransformer([
            ('imputer_continuous', SimpleImputer(strategy='mean'), self.continuous_features),
            ('categorical', pipe, self.categorical_features),
        ], remainder='passthrough')
        
        
        X_transformed = self.column_transformer.fit_transform(X)
        self.svc_classifier = SVC(probability=True, kernel=self.kernel)
        self.svc_classifier.fit(X_transformed, y)
        return self

    def predict(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.svc_classifier.predict(X_transformed)

    def predict_proba(self, X):
        X_transformed = self.column_transformer.transform(X)
        return self.svc_classifier.predict_proba(X_transformed)
