import pandas as pd
import numpy as np
from TaxiFareModel.utils import simple_time_tracker, haversine_vectorized, minkowski_distance
from google.cloud import storage
##AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"

DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")

BUCKET_NAME = 'wagon-bootcamp-1-297409'
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'

@simple_time_tracker
def get_data(nrows=10_000):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    #df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    # if nrows<=10000:
    #     df = pd.read_csv('~/code/pierrevermeulen/TaxiFareModel/raw_data/train_10k.csv', nrows=nrows)
    # else:
    #     df = pd.read_csv('~/code/pierrevermeulen/TaxiFareModel/raw_data/train.csv', nrows=nrows)
    # return df

    client = storage.Client()
    df = pd.read_csv("gs://{}/{}".format(BUCKET_NAME, BUCKET_TRAIN_DATA_PATH), nrows=nrows)
    return df

def clean_df(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df

    return df
if __name__ == '__main__':
    df = get_data()
