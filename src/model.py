import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

import lightgbm as lgb

def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average rides based on the last 4 weeks
    """
    X['average_rides_last_4_weeks'] = (X[f'rides_previous_{7*24}_hour'] + X[f'rides_previous_{7*2*24}_hour'] + X[f'rides_previous_{7*3*24}_hour'] 
                                    + X[f'rides_previous_{7*4*24}_hour'])/4
    return X

from sklearn.base import BaseEstimator, TransformerMixin

class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y = None):
        X_ = X.copy()

        #Generate numeric columns from the pickup_hour
        X_['hour'] = X_['pickup_hour'].dt.hour
        X_['day_of_week'] = X_['pickup_hour'].dt.dayofweek

        return X_.drop(['pickup_hour'], axis = 1)
    
def get_pipeline(**hyperparams) -> Pipeline:

    #sklearn transform
    add_feature_average_rides_last_4_weeks = FunctionTransformer(average_rides_last_4_weeks, validate = False)
    add_temporal_features = TemporalFeatureEngineer()

    #sklearn pipeline
    return make_pipeline(add_feature_average_rides_last_4_weeks, add_temporal_features, lgb.LGBMRegressor(**hyperparams))

