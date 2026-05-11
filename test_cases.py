import pytest
import pandas as pd
import numpy as np
import joblib
from utils.data_preprocessing import preprocess_data

def test_preprocessing_consistency():
    df = pd.DataFrame({
        'crop_type': ['Rice'], 'rainfall_mm': [100], 'avg_temp_c': [25],
        'fertilizer_kg_ha': [50], 'pesticide_kg_ha': [2], 'soil_pH': [6.5],
        'humidity_percent': [60], 'days_to_harvest': [120], 'previous_yield': [4.0],
        'plant_density_index': [80], 'yield_tons_ha': [5.2]
    })
    X, y, scaler = preprocess_data(df, is_training=True)
    assert X.shape[1] == 10
    assert y[0] == 5.2
    assert scaler is not None

def test_model_loading():
    xgb = joblib.load('models/xgboost_model.pkl')
    assert xgb is not None

def test_prediction_range():
    xgb = joblib.load('models/xgboost_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    sample = pd.DataFrame([[0, 120, 28, 100, 2, 6.5, 65, 120, 4.5, 100]],
                          columns=['crop_encoded', 'rainfall_mm', 'avg_temp_c',
                                   'fertilizer_kg_ha', 'pesticide_kg_ha', 'soil_pH',
                                   'humidity_percent', 'days_to_harvest',
                                   'previous_yield', 'plant_density_index'])
    scaled = scaler.transform(sample)
    pred = xgb.predict(scaled)[0]
    assert 1.5 <= pred <= 12.0