from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import traceback  # To capture and log detailed error stack traces

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('model/wine_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Feature order for correct input mapping
FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    form_data = request.form.to_dict()
    
    # Now form_data won't contain 'type', so adjust the data preparation accordingly.
    # Convert form data to the correct feature format and predict the result.
    
    # For example, extract features from the form data
    features = [
        float(form_data['fixed_acidity']),
        float(form_data['volatile_acidity']),
        float(form_data['citric_acid']),
        float(form_data['residual_sugar']),
        float(form_data['chlorides']),
        float(form_data['free_sulfur_dioxide']),
        float(form_data['density']),
        float(form_data['pH']),
        float(form_data['sulphates']),
        float(form_data['alcohol'])
    ]
    
    # Make prediction using your trained model
    prediction = model.predict([features])
    prediction_value = int(prediction[0]) if isinstance(prediction[0], np.int64) else prediction[0]
    prediction_label = "HIGH" if prediction_value == 1 else "LOW"
    return jsonify({'prediction': prediction_label})


if __name__ == "__main__":
    # Enable debugging mode and log errors to the console
    app.run(debug=True)
