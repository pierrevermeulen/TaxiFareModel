import time
import warnings
import pandas as pd
from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient
import joblib

from TaxiFareModel.data import get_data, clean_df
from TaxiFareModel.utils import compute_rmse, simple_time_tracker
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "Pierre"
EXPERIMENT_NAME = "taxifare_PV"

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
        self.X_train = X
        self.y_train = y
        self.estimator = self.kwargs.get("estimator", 'Linear')

        self.experiment_name = EXPERIMENT_NAME

    def get_estimator(self):
        if self.estimator == 'Linear':
            model = LinearRegression()
        return model

    def set_pipeline(self):

        CT = ColumnTransformer([('Time_processing', TimeFeaturesEncoder("pickup_datetime"),
            ["pickup_datetime"])])

        self.pipeline = Pipeline([('prep', CT), ('scaler', StandardScaler()), \
            ('model', self.get_estimator())])

    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self, X_test, y_test):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)

        return round(rmse, 3)
    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, \
            f"Models/{self.estimator}_Model.joblib")

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


if __name__ == "__main__":
    # Get and clean data
    N = 1000
    df = get_data(nrows=N)
    df = clean_df(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)

    df_test = pd.read_csv('~/code/pierrevermeulen/TaxiFareModel/raw_data/train_10k.csv', nrows=N)
    df_test = clean_df(df_test)
    y_test = df_test["fare_amount"]
    X_test = df_test.drop("fare_amount", axis=1)

    # Train and save model, locally and
    #t = Trainer(X=X_train, y=y_train, estimator="RandomForest")
    t = Trainer(X=X_train, y=y_train, estimator="Linear")
    t.train()
    t.mlflow_log_param("model", t.estimator)
    t.mlflow_log_param("student_name", myname)
    rmse = t.evaluate(X_test, y_test)
    t.mlflow_log_metric("rmse", rmse)
    t.save_model()
