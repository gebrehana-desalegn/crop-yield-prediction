from flask import Flask, request, render_template
import joblib
import pandas as pd
import os
from werkzeug.utils import secure_filename
from utils.yolo_integration import YOLOPlantDetector
from utils.data_preprocessing import preprocess_data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
xgb_model = joblib.load('models/xgboost_model.pkl')
scaler = joblib.load('models/scaler.pkl')
detector = YOLOPlantDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Handle image upload
        plant_density = None
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            plant_density = detector.extract_feature_from_image(filepath)
        else:
            plant_density = float(request.form.get('plant_density_index', 100))

        # Build input dictionary
        input_data = {
            'crop_type': request.form['crop_type'],
            'rainfall_mm': float(request.form['rainfall_mm']),
            'avg_temp_c': float(request.form['avg_temp_c']),
            'fertilizer_kg_ha': float(request.form['fertilizer_kg_ha']),
            'pesticide_kg_ha': float(request.form['pesticide_kg_ha']),
            'soil_pH': float(request.form['soil_pH']),
            'humidity_percent': float(request.form['humidity_percent']),
            'days_to_harvest': int(request.form['days_to_harvest']),
            'previous_yield': float(request.form['previous_yield']),
            'plant_density_index': plant_density
        }
        input_df = pd.DataFrame([input_data])
        X_input, _ = preprocess_data(input_df, is_training=False, scaler=scaler)
        prediction = xgb_model.predict(X_input)[0]

        return render_template('index.html', prediction=round(prediction, 2))
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)