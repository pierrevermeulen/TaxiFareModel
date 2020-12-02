import time
import datetime
import os
import warnings
import pandas as pd
import numpy as np
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib
import json
from google.cloud import storage

from TaxiFareModel.data import get_data, clean_df, DIST_ARGS
from TaxiFareModel.utils import compute_rmse, simple_time_tracker, \
    haversine_vectorized, minkowski_distance
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash, \
    Direction, DistanceToCenter, DistanceTojfk
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor


### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-bootcamp-1-297409'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
# /!\ here you need to decide if you are going to train using the provided and uploaded data/train_1k.csv sample file
# or if you want to use the full dataset (you need need to upload it first of course)
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v2'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -


warnings.filterwarnings("ignore", category=FutureWarning)
MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "Pierre"
EXPERIMENT_NAME = "taxifare_PV_GCP"

class Trainer(object):

    ESTIMATOR = "Linear"

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containing all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X: pandas DataFrame
        :param y: pandas DataFrame
        :param kwargs:
        """
        self.pipeline = None
        self.model = None
        self.kwargs = kwargs
        self.split = self.kwargs.get("split", True)  # cf doc above
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(self.X, self.y,test_size=0.1)
        self.estimator = self.kwargs.get("estimator", 'Linear')

        self.distance_type = kwargs.get("distance_type", "euclidian")

        self.mlflow = kwargs.get("mlflow", False)
        self.mlflow_uri = kwargs.get("mlflow_uri", None)
        self.experiment_name = kwargs.get("experiment_name", EXPERIMENT_NAME)

        if self.mlflow :
            self.mlflow_log_param("nrows", nrows)
            self.mlflow_log_param("distance_type", distance_type)
            self.mlflow_log_param("estimator", estimator)

    def get_estimator(self):
        if self.estimator == "Lasso" :
            model = Lasso()
        elif self.estimator == "Ridge" :
            model = Ridge()
        elif self.estimator == "GBM":
            model = GradientBoostingRegressor()
        elif self.estimator == "Linear" :
            model = LinearRegression()
        elif self.estimator == "RandomForest" :
            model  = RandomForestRegressor()
        elif self.estimator == "xgboost" :
            model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,gamma=3)
        else :
            model = Lasso()

        return model

    def set_pipeline(self):
        pipe_time_feature = make_pipeline(TimeFeaturesEncoder(time_column="pickup_datetime"), OneHotEncoder())
        pipe_distance = make_pipeline(DistanceTransformer(distance_type=self.distance_type, **DIST_ARGS ))
        pipe_distancejfk = make_pipeline(DistanceTojfk())

        features_encoder = ColumnTransformer(
                [
                    ("distance_feature", pipe_distance, list(DIST_ARGS.values())),
                    ("time_feature", pipe_time_feature, ["pickup_datetime"]),
                    ("distance_jfk", pipe_distancejfk, list(DIST_ARGS.values()))
                    ]
                )

        self.pipeline = Pipeline(
                steps = [
                    ("features_encoder", features_encoder),
                    ("model", self.get_estimator())
                    ]
                )

    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def compute_rmse(self, X, y):
        y_pred = self.pipeline.predict(X)
        rmse = compute_rmse(y_pred, y)
        return round(rmse, 3)

    def evaluate(self):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(self.X_test)
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        rmse_test = self.compute_rmse(self.X_test, self.y_test)
        print("rmse_train = {}".format(rmse_train))
        print("rmse_test = {}".format(rmse_test))

        if self.mlflow :
            self.mlflow_log_metric("rmse_train", rmse_train)
            self.mlflow_log_metric("rmse_test", rmse_test)

        return round(rmse_test, 3)

    def save_model(self):
        """Save the model into a .joblib format"""

        storage_name= f"{MODEL_NAME}_{MODEL_VERSION}_{self.estimator}.joblib"
        print(f"saved {storage_name} locally")

        joblib.dump(self.pipeline, storage_name)

        # Implement here
        storage_location = BUCKET_NAME
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(storage_location)
        blob = bucket.blob(f"models/{storage_name}")
        blob.upload_from_filename(storage_name)
        print(f"saved {storage_name} on GS")

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)

# if __name__ == "__main__":
#     # Get and clean data
#     N = 1000
#     df = get_data(nrows=N)
#     df = clean_df(df)
#     #df = feat_eng(df)

#     y = df["fare_amount"]
#     X = df.drop("fare_amount", axis=1)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#     # Train and save model, locally and
#     #t = Trainer(X=X_train, y=y_train, estimator="RandomForest")

#     t = Trainer(X=X_train, y=y_train, estimator="Linear")
#     t.train()
#     t.mlflow_log_param("model", t.estimator)
#     t.mlflow_log_param("student_name", myname)
#     rmse = t.evaluate(X_test, y_test)
#     t.mlflow_log_metric("rmse", rmse)
#     t.save_model()


if __name__=="__main__":


    '''
    Load data
    '''
    print("Load data")
    df = get_data(nrows=1000)
    df = clean_df(df)

    X = df.drop("fare_amount", axis =1)
    y = df["fare_amount"]

    param_set = [
            dict(
                nrows=1000,
                estimator="Linear",
                mlflow=False,
                distance_type="manhattan",
                mlflow_uri="http://localhost:5000",
                experiment_name="TaxiFareModel"
            ),
            dict(
                nrows=1000,
                estimator="Linear",
                mlflow=False,
                distance_type="haversine",
                mlflow_uri="http://localhost:5000",
                experiment_name="TaxiFareModel"
            ),
            dict(
                nrows=1000,
                estimator="RandomForest",
                mlflow=False,
                distance_type="manhattan",
                mlflow_uri="http://localhost:5000",
                experiment_name="TaxiFareModel"
            ),
            dict(
                nrows=1000,
                estimator="RandomForest",
                mlflow=False,
                distance_type="haversine",
                mlflow_uri="http://localhost:5000",
                experiment_name="TaxiFareModel"
            ),
            dict(
                nrows=1000,
                estimator="xgboost",
                mlflow=False,
                distance_type="haversine",
                mlflow_uri="http://localhost:5000",
                experiment_name="TaxiFareModel"
            ),
            ]

    for params in param_set :

        '''INPUT PARAMS :'''
        print(json.dumps(params, indent = 2))

        '''Train'''
        print("Train")
        trainer = Trainer(X, y, **params)
        trainer.train()

        '''Evaluate'''
        print("Evaluate")
        trainer.evaluate()

        '''Save'''
        print("Save")
        trainer.save_model()
