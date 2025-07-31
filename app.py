from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('models/crop_model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('models/fertilizer_clf.pkl', 'rb') as f:
    clf = pickle.load(f)

with open('models/le_crop.pkl', 'rb') as f:
    le_crop = pickle.load(f)
with open('models/le_soil.pkl', 'rb') as f:
    le_soil = pickle.load(f)
with open('models/le_fertilizer.pkl', 'rb') as f:
    le_fertilizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')  # Your form
@app.route('/crop_form')
def crop_form():
    return render_template('crop_form.html')  # Your form
@app.route('/fertilizer_form')
def fertilizer_form():
    return render_template('fertilizer_form.html')
@app.route('/disease_form')
def disease_form():
    return render_template('disease_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = []
        for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            value = request.form.get(key)
            if value is None or value.strip() == '':
                return f"Error: Missing value for {key}"
            data.append(float(value))
        
        prediction = model.predict([data])[0]
        return render_template('result.html',
                               result_title="🌾 Crop Prediction Result",
                               result_value=f"The suitable crop is: **{prediction}**.")
    
    except ValueError as e:
        return f"Invalid input: {e}", 400


@app.route('/fertilizer_predict', methods=['POST'])
def fertilizer_predict():
    try:
        data = request.form
        print("Form data received:", data)

        # Extract and convert inputs
        temperature = float(data['Temperature'])
        moisture = float(data['Moisture'])
        rainfall = float(data['Rainfall'])
        ph = float(data['PH'])
        nitrogen = float(data['Nitrogen'])
        phosphorous = float(data['Phosphorous'])
        potassium = float(data['Potassium'])
        carbon = float(data['Carbon'])

        # Encode soil and crop types
        soil = le_soil.transform([data['Soil']])[0]
        crop = le_crop.transform([data['Crop']])[0]

        # Combine features
        features = np.array([[temperature, moisture, rainfall, ph,
                              nitrogen, phosphorous, potassium, carbon,
                              soil, crop]])
        
        # Predict
        prediction = clf.predict(features)
        fertilizer_name = le_fertilizer.inverse_transform(prediction)[0]

        return render_template('fertilizer_result.html', fertilizer=fertilizer_name)
    
    except Exception as e:
        print("Error:", e)
        return f"Error occurred: {str(e)}", 400



if __name__ == '__main__':
    app.run(debug=True)

