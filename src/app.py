from flask import Flask, request, jsonify
import torch
import numpy as np
from model import LSTMModel
from preprocess import custom_inverse_scale, min_receipts, max_receipts
import pandas as pd

app = Flask(__name__)

# Load the trained model and the necessary data used for training
def load_model_and_predict():
    # Load data and model used during training
    data = pd.read_csv('/app/data/data_daily.csv', parse_dates=['# Date'], index_col='# Date')
    monthly_data = data['Receipt_Count'].resample('M').sum()

    # Scale the data
    min_val = monthly_data.min()
    max_val = monthly_data.max()

    def custom_scale(data, min_val, max_val):
        return (data - min_val) / (max_val - min_val)

    def custom_inverse_scale(scaled_data, min_val, max_val):
        return scaled_data * (max_val - min_val) + min_val

    scaled_monthly_data = custom_scale(monthly_data.values, min_val, max_val)

    # Prepare input for 2022 predictions (months 13-24)
    months_input = np.array([[i] for i in range(len(scaled_monthly_data) + 1, len(scaled_monthly_data) + 13)])  # Months 13 to 24

    # Load model
    model_path = 'model.pth'
    model = LSTMModel(input_size=1, hidden_size=1000, output_size=1, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Predict for 2022
    months_tensor = torch.tensor(months_input, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        predicted_scaled = model(months_tensor).numpy()

    # Inverse scale the predictions to get the actual receipt counts
    predicted_receipts = custom_inverse_scale(predicted_scaled, min_val, max_val)

    # Adjust the predicted receipts
    last_actual_2021 = monthly_data.values[-1]
    first_predicted_2022 = predicted_receipts[0]
    shift = last_actual_2021 - first_predicted_2022
    adjusted_predicted_receipts = predicted_receipts + shift

    return adjusted_predicted_receipts[:12]  # Return the first 12 months of 2022 predictions

# Load the predictions once when the app starts
predicted_receipts_2022 = load_model_and_predict()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    month = data.get('month')

    if month is None or month < 1 or month > 12:
        return jsonify({'error': 'Valid month (1-12) is required'}), 400

    # Return the predicted receipt for the requested month
    predicted_receipt = predicted_receipts_2022[month - 1]

    return jsonify({
        'month': month,
        'predicted_receipts': float(predicted_receipt)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
