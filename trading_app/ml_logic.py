import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
        
    def prepare_data(self, data, lookback=60):
        """Prepare data for LSTM model"""
        # Fit the scaler on all the data first
        self.scaler = self.scaler.fit(data['Close'].values.reshape(-1,1))
        scaled_data = self.scaler.transform(data['Close'].values.reshape(-1,1))
        
        x_train = []
        y_train = []
        
        for i in range(lookback, len(scaled_data)):
            x_train.append(scaled_data[i-lookback:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        return x_train, y_train
        
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, ticker, years=5):
        """Train model on historical data"""
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years*365)
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Prepare data
        x_train, y_train = self.prepare_data(df)
        
        # Build and train model
        self.model = self.build_model((x_train.shape[1], 1))
        self.model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=0)
        
        return df
        
    def predict_next_day(self, data):
        """Predict next day's closing price"""
        if self.model is None:
            # Train the model if it hasn't been trained yet
            self.train(data.name)
            
        # Prepare last 60 days of data
        last_60_days = data['Close'].values[-60:]
        last_60_days_scaled = self.scaler.transform(last_60_days.reshape(-1, 1))
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Predict
        pred_scaled = self.model.predict(X_test, verbose=0)
        prediction = float(self.scaler.inverse_transform(pred_scaled)[0][0])  # Convert to Python float
        
        # Calculate prediction metrics
        current_price = float(data['Close'].iloc[-1])  # Convert to Python float
        predicted_change = float(((prediction - current_price) / current_price) * 100)  # Convert to Python float
        
        return {
            'current_price': current_price,
            'predicted_price': prediction,
            'predicted_change': predicted_change
        } 