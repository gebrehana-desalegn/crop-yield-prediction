import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(df, is_training=True, scaler=None):
    df_processed = df.copy()
    
    if is_training:
        le = LabelEncoder()
        df_processed['crop_encoded'] = le.fit_transform(df_processed['crop_type'])
    else:
        le = LabelEncoder()
        le.fit(['Rice', 'Wheat', 'Maize'])
        df_processed['crop_encoded'] = le.transform(df_processed['crop_type'])
    
    feature_cols = ['crop_encoded', 'rainfall_mm', 'avg_temp_c', 'fertilizer_kg_ha',
                    'pesticide_kg_ha', 'soil_pH', 'humidity_percent',
                    'days_to_harvest', 'previous_yield', 'plant_density_index']
    X = df_processed[feature_cols]
    
    if 'yield_tons_ha' in df_processed.columns:
        y = df_processed['yield_tons_ha']
    else:
        y = None
    
    if is_training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler
    else:
        if scaler is None:
            raise ValueError("Scaler required for prediction")
        X_scaled = scaler.transform(X)
        return X_scaled, scaler

def train_test_split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)