FROM python:3.7-slim-buster

RUN pip3 install boto3 numpy==1.18.1 pandas==1.0.1 scikit-learn==0.23.2 imbalanced-learn==0.7.0 statsmodels xgboost
RUN rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]