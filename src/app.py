from flask import Flask, render_template, request, jsonify, send_file
import torch
import numpy as np
import pandas as pd
from model import LSTMModel
from preprocess import custom_inverse_scale
import io
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static', template_folder='templates') # give our static and template folders for simple frontend

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

    months_input = np.array([[i] for i in range(len(scaled_monthly_data) + 1, len(scaled_monthly_data) + 13)])  # Months 13 to 24

    # Load the model
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

    return adjusted_predicted_receipts[:12], monthly_data  # Return the first 12 months of 2022 predictions and actual data for 2021

predicted_receipts_2022, actual_receipts_2021 = load_model_and_predict()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    month = data.get('month')
    
    if month is None or month < 1 or month > 12:
        return jsonify({'error': 'Valid month (1-12) is required'}), 400
    
    predicted_receipt = predicted_receipts_2022[month - 1]
    predicted_receipt_rounded = round(float(predicted_receipt), 2)
    return jsonify({
        'month': month,
        'predicted_receipts': predicted_receipt_rounded
    })

@app.route('/plot', methods=['GET'])
def plot_receipts():
    highlight_month = request.args.get('month', default=None, type=int)

    months_labels = [
        f"2021/{i}" for i in range(1, 13)
    ] + [f"2022/{i}" for i in range(1, 13)]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(actual_receipts_2021) + 1), actual_receipts_2021.values, label='Actual Receipts (2021)', marker='o', color='blue')
    plt.plot(range(len(actual_receipts_2021) + 1, len(actual_receipts_2021) + 13), predicted_receipts_2022, label='Predicted Receipts (2022)', marker='x', color='red')

    if highlight_month is not None and 1 <= highlight_month <= 12:
        plt.scatter([len(actual_receipts_2021) + highlight_month], [predicted_receipts_2022[highlight_month - 1]],
                    color='green', s=100, label=f'Highlighted Month {highlight_month}')
    
    plt.title('Actual 2021 vs Predicted 2022 Monthly Receipts')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')

    plt.xticks(ticks=range(1, len(months_labels) + 1), labels=months_labels, rotation=45)

    plt.grid(True)
    plt.legend()

    # Save plot to a BytesIO object for sending via Flask
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
