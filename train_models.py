import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from utils.data_preprocessing import preprocess_data, train_test_split_data
from utils.visualization import plot_feature_importance

df = pd.read_csv('data/crop_yield_dataset.csv')
X, y, scaler = preprocess_data(df, is_training=True)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

joblib.dump(scaler, 'models/scaler.pkl')

xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.07, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'models/xgboost_model.pkl')

rf_model = RandomForestRegressor(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/random_forest.pkl')

print("\n===== Model Performance =====")
for name, model in [('XGBoost', xgb_model), ('Random Forest', rf_model)]:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"{name:15} -> R²: {r2:.4f}, MAE: {mae:.4f}")

plt.figure(figsize=(10,6))
plt.scatter(y_test, xgb_model.predict(X_test), alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Yield (tons/ha)')
plt.ylabel('Predicted Yield')
plt.title('XGBoost: Predicted vs Actual')
plt.savefig('static/model_performance.png')

feature_names = ['crop_encoded', 'rainfall_mm', 'avg_temp_c', 'fertilizer_kg_ha',
                 'pesticide_kg_ha', 'soil_pH', 'humidity_percent',
                 'days_to_harvest', 'previous_yield', 'plant_density_index']
plot_feature_importance(xgb_model, feature_names, save_path='static/feature_importance.png')
print("✅ Models and plots ready.")