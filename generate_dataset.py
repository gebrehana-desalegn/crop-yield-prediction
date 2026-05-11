import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
n_samples = 2000
crops = ['Rice', 'Wheat', 'Maize']

data = {
    'crop_type': np.random.choice(crops, n_samples),
    'rainfall_mm': np.random.uniform(50, 300, n_samples),
    'avg_temp_c': np.random.uniform(15, 40, n_samples),
    'fertilizer_kg_ha': np.random.uniform(20, 180, n_samples),
    'pesticide_kg_ha': np.random.uniform(0.5, 8, n_samples),
    'soil_pH': np.random.uniform(5.0, 8.5, n_samples),
    'humidity_percent': np.random.uniform(40, 90, n_samples),
    'days_to_harvest': np.random.randint(80, 200, n_samples),
    'previous_yield': np.random.uniform(2.0, 8.0, n_samples),
    'plant_density_index': np.random.uniform(50, 200, n_samples)
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['crop_encoded'] = le.fit_transform(df['crop_type'])

base_yield = (0.02 * df['rainfall_mm'] + 0.15 * df['fertilizer_kg_ha'] +
              0.03 * df['plant_density_index'] - 0.1 * df['avg_temp_c'] +
              1.5 * (df['crop_encoded'] == 0) +
              1.2 * (df['crop_encoded'] == 1) +
              1.8 * (df['crop_encoded'] == 2))

noise = np.random.normal(0, 0.5, n_samples)
df['yield_tons_ha'] = (base_yield + noise).clip(1.5, 12.0)
df_final = df.drop('crop_encoded', axis=1)
df_final.to_csv('data/crop_yield_dataset.csv', index=False)
print(f"✅ Dataset created: data/crop_yield_dataset.csv ({n_samples} samples)")