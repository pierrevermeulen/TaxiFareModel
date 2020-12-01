import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import pygeohash as gh

dist_args = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")

# Implement DistanceTransformer and TimeFeaturesEncoder
class DistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, X, y=None):
        X['distance'] = haversine_vectorized(df, start_lat="pickup_latitude", \
            start_lon="pickup_longitude", end_lat="dropoff_latitude", end_lon="dropoff_longitude")
        return X[["distance"]]

    def fit(self, X, y=None):
        return self


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.weekday
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[["dow", "hour", "month", "year"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class AddGeohash(BaseEstimator, TransformerMixin):

    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash_pickup'] = X.apply(
            lambda x: gh.encode(x.pickup_latitude, x.pickup_longitude, precision=self.precision), axis=1)
        X['geohash_dropoff'] = X.apply(
            lambda x: gh.encode(x.dropoff_latitude, x.dropoff_longitude, precision=self.precision), axis=1)
        return X[['geohash_pickup', 'geohash_dropoff']]

