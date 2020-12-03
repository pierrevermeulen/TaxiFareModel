FROM python:3.8.6-buster

RUN mkdir api
RUN mkdir TaxiFareModel

COPY api/fast.py api/fast.py
COPY TaxiFareModel/predict.py TaxiFareModel/predict.py
COPY TaxiFareModel/encoders.py TaxiFareModel/encoders.py
COPY TaxiFareModel/utils.py TaxiFareModel/utils.py
COPY requirements_docker.txt requirements_docker.txt
COPY taxifare_v2_Linear.joblib taxifare_v2_Linear.joblib

RUN pip install -r requirements_docker.txt
EXPOSE 8080
CMD uvicorn api.fast:app --host 0.0.0.0 --port 8080
