import os
from math import sqrt

import joblib
import pandas as pd
#from TaxiFareModel.params import MODEL_NAME
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error

BUCKET_NAME = "wagon-bootcamp-1-297409"
BUCKET_TRAIN_DATA_PATH = "data/train_1k.csv"
MODEL_NAME = "taxifare_v2_xgboost.joblib"

def get_test_data():
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    client = storage.Client()
    df = pd.read_csv("gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH), nrows=1000)

    return df


def download_model(bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/{}'.format(MODEL_NAME)
    print(storage_location)
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print(f"=> pipeline downloaded from storage")
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(folder="Pipeline", kaggle_upload=False):
    df_test = get_test_data()
    X_test = df_test.drop("fare_amount", axis =1)
    y_test = None
    if "fare_amount" in df_test.columns:
        y_test = df_test["fare_amount"]
    pipeline = download_model()
    # Check if model saved was the ouptut of RandomSearch or Gridsearch
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(X_test)
    else:
        y_pred = pipeline.predict(X_test)
    if isinstance(y_test, pd.Series):
        res = evaluate_model(y_test, y_pred)
        for k, v in res.items():
            print(f"Metric {k}: value {v:.2f}")

    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]

    if kaggle_upload:
        name = f"predictions_{folder}.csv"
        df_sample.to_csv(name, index=False)
        print("prediction saved under kaggle format")
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':
    folder = "Pipeline"
    # model = download_model(folder)
    generate_submission_csv(folder, kaggle_upload=False)
