import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import product

torch.manual_seed(42)
np.random.seed(42)

# Load the data used during training
data = pd.read_csv('./data/data_daily.csv', parse_dates=['# Date'], index_col='# Date')

monthly_data = data['Receipt_Count'].resample('M').sum()

# Scale the data
min_val = monthly_data.min()
max_val = monthly_data.max()

def custom_scale(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def custom_inverse_scale(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val) + min_val

scaled_monthly_data = custom_scale(monthly_data.values, min_val, max_val)

# Prepare the dataset
X = np.array([[i] for i in range(1, len(scaled_monthly_data) + 1)])  # Month index
y = scaled_monthly_data

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reshape X_train_tensor for LSTM [batch_size, time_step, input_size]
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # Add one dimension for the sequence
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)  # Add one dimension for the target

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)  # Add one dimension for the sequence
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)  # Add one dimension for the target

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
    
    def forward(self, x):
        # Initialize hidden state and cell state with the correct size
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last output
        return out

# Hyperparameter grid
hidden_sizes = [100, 500, 1000]
num_layers_list = [1, 2]
learning_rates = [0.001, 0.0005]
epochs = 50
patience = 10  # Early stopping patience

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()

# For tracking the best model and best loss
best_val_loss = float('inf')
best_hyperparameters = None
best_model_path = 'src/model.pth'

# Grid search over hyperparameters
for hidden_size, num_layers, learning_rate in product(hidden_sizes, num_layers_list, learning_rates):
    print(f'Training model with hidden_size={hidden_size}, num_layers={num_layers}, learning_rate={learning_rate}')
    
    # Initialize model and optimizer for each combination of hyperparameters
    model = LSTMModel(input_size=1, hidden_size=hidden_size, output_size=1, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop with validation and early stopping
    train_losses = []
    val_losses = []
    early_stop_count = 0

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
        
        # Early stopping check
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            early_stop_count = 0
            best_hyperparameters = (hidden_size, num_layers, learning_rate)
            # Save the entire model
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.6f}")
                break

# After grid search, load the best model
best_model = LSTMModel(input_size=1, hidden_size=best_hyperparameters[0], output_size=1, num_layers=best_hyperparameters[1]).to(device)
best_model.load_state_dict(torch.load(best_model_path))

print(f"Best hyperparameters: hidden_size={best_hyperparameters[0]}, num_layers={best_hyperparameters[1]}, learning_rate={best_hyperparameters[2]}")

best_model.eval()

# Predict for training and validation data
with torch.no_grad():
    all_X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)  # Use the entire data
    predicted_scaled = best_model(all_X).cpu().numpy()  # Predict and move to CPU

# Inverse scale the predictions
predicted_receipts = custom_inverse_scale(predicted_scaled, min_val, max_val)

# Get the last actual value from 2021 and the first predicted value from 2022
last_actual_2021 = monthly_data.values[-1]
first_predicted_2022 = predicted_receipts[0]

# Calculate the shift needed to match the predicted values with the actual ones
shift = last_actual_2021 - first_predicted_2022

# Adjust all the predicted values for 2022 by adding the shift
adjusted_predicted_receipts = predicted_receipts + shift

# Actual 2021 vs Adjusted Predicted 2022 Monthly Receipts Graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(monthly_data) + 1), monthly_data.values, label='Actual Receipts (2021)', marker='o', color='blue')
plt.plot(range(len(monthly_data) + 1, len(monthly_data) + 13), adjusted_predicted_receipts, label='Predicted Receipts (2022)', marker='x', color='red')
plt.title('Actual 2021 vs Adjusted Predicted 2022 Monthly Receipts')
plt.xlabel('Month')
plt.ylabel('Receipt Count')
plt.legend()

# Save the figure instead of displaying it
plt.savefig('adjusted_predicted_vs_actual_receipts_2021_2022.png')

# Predict for the entire year of 2022 (months 13 to 24)
months_2022 = list(range(1, 13))  
predicted_receipts_2022 = adjusted_predicted_receipts[:12]  # Get the first 12 predicted months of 2022

# Print the predicted monthly receipts for each month of 2022 (for reference in graph created)
for month, predicted_receipt in zip(months_2022, predicted_receipts_2022):
    predicted_receipt_value = predicted_receipt.item()  # Convert array to scalar
    print(f"Month {month} of 2022: Predicted Receipts = {predicted_receipt_value:.2f}")