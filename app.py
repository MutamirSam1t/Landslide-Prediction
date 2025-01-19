from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your model
with open("model_rf.pkl", "rb") as f:
    model = pickle.load(f)

# List of 8 features
feature_names = [
    "ELEVATION", "SLOPE", "ASPECT", "TWI", 
    "SPI", "NDVI", "RAINFALL", "LANDUSE"
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from the form
        input_values = [
            float(request.form[feature]) for feature in feature_names
        ]
        
        # Convert inputs to a DataFrame
        input_data = pd.DataFrame([input_values], columns=feature_names)
        
        # Predict using the loaded model
        prediction = model.predict(input_data)[0]
        
        # Map prediction to High Risk or Low Risk
        risk = "High Risk" if prediction == 1 else "Low Risk"
        return render_template('index.html', prediction=risk)
    
    return render_template('index.html', prediction=None)

if __name__ == "__main__":
    app.run(debug=True)