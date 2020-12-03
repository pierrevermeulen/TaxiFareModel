from fastapi import FastAPI
import json
import pandas as pd
from TaxiFareModel.predict import download_model

app = FastAPI()

# define a root `/` endpoint
@app.get("/")
def index():
    return {"Salut": "Nounou"}

@app.get("/predict_fare/{pickup_datetime}/{pickup_longitude}/{pickup_latitude}/{dropoff_longitude}/{dropoff_latitude}/{passenger_count}")
def predict_fare(pickup_datetime, pickup_longitude,pickup_latitude, dropoff_longitude,\
    dropoff_latitude, passenger_count):

    data_dict = {'key':'To_predict',
    'pickup_datetime': pickup_datetime,
    'pickup_longitude': float(pickup_longitude),
    'pickup_latitude': float(pickup_latitude),
    'dropoff_longitude': float(dropoff_longitude),
    'dropoff_latitude': float(dropoff_latitude),
    'passenger_count': float(passenger_count)}

    df = pd.DataFrame.from_dict(data_dict, orient='index').T

    pipeline = download_model()
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df)
    else:
        y_pred = pipeline.predict(df)

    return {"predict_fare": float(y_pred[0])}

