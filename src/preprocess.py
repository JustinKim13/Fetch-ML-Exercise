import pandas as pd
import os

# Load data for scaling
data_file_path = os.path.join(os.path.dirname(__file__), '../data/data_daily.csv')
data = pd.read_csv(data_file_path, parse_dates=['# Date'], index_col='# Date')

min_receipts = data['Receipt_Count'].min()
max_receipts = data['Receipt_Count'].max()

def custom_scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def custom_inverse_scale(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val) + min_val
