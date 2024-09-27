# Receipt Prediction Application

## Overview
This project is aimed at predicting the number of monthly receipts scanned in 2022 using an LSTM model. The prediction is based on historical daily receipt data from 2021, which is resampled into monthly data for training.

## Prerequisites
- Python 3.9
- PyTorch
- Pandas
- Matplotlib
- Flask
- Scikit-learn 

## Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-repo/fetch-ml-exercise.git](https://github.com/JustinKim13/Fetch-ML-Exercise.git)
   ``````
2. Navigate to the project directory:
   ```bash
   cd fetch-ml-exercise
   ``````
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application
To run the web application:
1. Build the Docker image:
   ```bash
   docker build -t fetch-ml-exercise .  
   ```
2. Run the Docker container:
   ```bash
   docker run -p 5002:5000 fetch-ml-exercise
   ```
3. Open your browser and go to `http://localhost:5002`.

You can enter a month (1-12) to see the predicted receipts for that month.

## Key Files
- `app.py`: The Flask application that serves the frontend and handles prediction requests.
- `model.py`: Defines the LSTM model architecture.
- `train.py`: Contains the training loop for the LSTM model, including hyperparameter tuning and early stopping.
- `preprocess.py`: Handles scaling and inverse scaling of the receipt data.
- `static/`: Contains the frontend assets (CSS, JS).
- `templates/`: Contains the HTML files for the web interface.
- `data/data_daily.csv`: Historical data of daily receipts for 2021.

## Model Overview
I used an LSTM model to predict monthly receipt totals for 2022, drawing from 2021's data. Here are the key steps:
1. **Data Resampling**: I resampled daily data into monthly totals to better capture long-term trends.
2. **Data Scaling**: To ensure the model trained smoothly, I scaled the data between 0 and 1.
3. **LSTM Model**:  I designed a 2-layer LSTM model, tuned hyperparameters like hidden layers and learning rates, and employed early stopping to avoid overfitting.
4. **Prediction**: The model generates receipt forecasts for the next 12 months, which are then inverse-scaled back to the original range.
5. **Adjustment**: To ensure continuity, I adjusted the model’s predictions to match the last known data point from 2021.
