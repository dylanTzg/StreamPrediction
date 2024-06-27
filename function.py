import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import UploadFile, File, HTTPException
from pydantic import BaseModel, Field

model_ia = joblib.load('model.pkl')
API_URL = "https://api-inference.huggingface.co/models/gpt2"
API_TOKEN = "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"


class InputData(BaseModel):
    danceability: float = Field(alias='danceability_%')
    valence: float = Field(alias='valence_%')
    energy: float = Field(alias='energy_%')
    acousticness: float = Field(alias='acousticness_%')
    instrumentalness: float = Field(alias='instrumentalness_%')
    liveness: float = Field(alias='liveness_%')
    speechiness: float = Field(alias='speechiness_%')


def predict(input_data: InputData):
    check_valid_prediction_data(input_data)

    input_array = np.array([[input_data.danceability, input_data.valence, input_data.energy,
                             input_data.acousticness, input_data.instrumentalness, input_data.liveness,
                             input_data.speechiness]])

    prediction = model_ia.predict(input_array)
    return {"Streams prediction": prediction[0]}


def training(file: UploadFile = File(...)):
    check_valid_file_extension(file.filename)
    df = pd.read_csv(file.file, encoding='latin1')
    check_valid_file_data(df)
    data_clean(df)

    train_model(df)
    joblib.dump(model_ia, 'model.pkl')

    return {"detail": "Training completed"}


def model(message: str):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=message)
    return response.json()


# Utils
def check_valid_prediction_data(input_data: InputData):
    if any(value is None for value in input_data.dict().values()):
        missing_values = ', '.join([key for key, value in input_data.dict().items() if value is None])
        raise HTTPException(
            status_code=400,
            detail=f"Missing values for: {missing_values}")
    if not all(0 <= value <= 100 for value in input_data.dict().values()):
        raise HTTPException(
            status_code=400,
            detail="The values must be between 0 and 100")


def check_valid_file_data(df):
    if not all(col in df.columns for col in ['danceability_%', 'valence_%', 'energy_%',
                                             'acousticness_%', 'instrumentalness_%', 'liveness_%',
                                             'speechiness_%', 'streams']):
        raise HTTPException(
            status_code=400,
            detail="The dataset must have columns: danceability_%, valence_%, energy_%, acousticness_%, "
                   "instrumentalness_%, liveness_%, speechiness_% and streams")


def check_valid_file_extension(filename):
    if not filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="The file must be a csv file")


def data_clean(df):
    df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
    df['streams'] = df['streams'].fillna(0)
    df['streams'] = df['streams'].astype(int)
    df['streams'].unique()
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def train_model(df):
    X = df[['danceability_%', 'valence_%', 'energy_%',
            'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']]
    y = df['streams']

    model_ia.fit(X, y)
