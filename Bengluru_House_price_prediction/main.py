from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load CSV and model
data_path = os.path.join('Bengluru_House_price_prediction', 'Cleaned_data.csv')
model_path = os.path.join('Bengluru_House_price_prediction', 'RidgeModel.pkl')

data = pd.read_csv(data_path)
pipe = pickle.load(open(model_path, 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    location = request.form.get('location')

    try:
        bhk = int(request.form.get('bhk', 0))
        bath = int(request.form.get('bath', 0))
        sqft = float(request.form.get('total_sqft', 0))
    except ValueError:
        return "Invalid input!"

    print(location, bhk, bath, sqft)

    # Prepare input DataFrame
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location','total_sqft','bath','bhk'])

    # Predict price
    prediction = pipe.predict(input_df)[0] * 1e5

    # Return rounded prediction
    return str(np.round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
