# src/stop_take/dynamic_sltp.py
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class DynamicSLTPStrategy:
    def __init__(self, config, dynamic_model=None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        if dynamic_model is not None:
            self.model = dynamic_model.model
            self.logger.info("Using pre-trained dynamic model for SL/TP.")
        else:
            self.model = None

    def train_model(self, data):
        """Train a simple model to predict SL/TP levels based on historical data."""
        try:
            # Create a copy of the data to avoid modifying the original
            df = data.copy()

            # Prepare features (e.g., price differences, volatility)
            df.loc[:, 'price_diff'] = df['Close'].diff()
            df.loc[:, 'volatility'] = df['Close'].rolling(window=20).std()
            df.loc[:, 'returns'] = df['Close'].pct_change()

            # Drop rows with NaN values after feature creation
            df = df.dropna()

            # Target: Future price movement (simplified)
            df.loc[:, 'target'] = df['Close'].shift(-1) - df['Close']

            # Drop rows with NaN values after target creation
            df = df.dropna()

            # Features and target
            X = df[['price_diff', 'volatility', 'returns']]
            y = df['target']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Evaluate model (optional)
            score = self.model.score(X_test, y_test)
            self.logger.info(f"Dynamic SL/TP model trained successfully with R^2 score: {score}")
        except Exception as e:
            self.logger.error(f"Error training dynamic SL/TP model: {str(e)}")
            self.model = None

    def custom_sl(self, price, data, direction):
        """Calculate stop-loss dynamically using the trained model."""
        if self.model is None:
            self.logger.warning("No trained model available, using default SL.")
            return price - 0.01 if direction == 'buy' else price + 0.01

        try:
            # Prepare features for prediction
            latest_data = pd.DataFrame({
                'price_diff': [data['Close'][-1] - data['Close'][-2]],
                'volatility': [data['Close'][-20:].std()],
                'returns': [(data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]]
            })

            # Predict price movement
            predicted_movement = self.model.predict(latest_data)[0]

            # Adjust SL based on predicted movement
            if direction == 'buy':
                sl = price - abs(predicted_movement) * 2  # Conservative SL
            else:
                sl = price + abs(predicted_movement) * 2
            return sl
        except Exception as e:
            self.logger.error(f"Error calculating dynamic SL: {str(e)}")
            return price - 0.01 if direction == 'buy' else price + 0.01

    def custom_tp(self, price, data, direction, entry_price):
        """Calculate take-profit dynamically using the trained model."""
        if self.model is None:
            self.logger.warning("No trained model available, using default TP.")
            return price + 0.02 if direction == 'buy' else price - 0.02

        try:
            # Prepare features for prediction
            latest_data = pd.DataFrame({
                'price_diff': [data['Close'][-1] - data['Close'][-2]],
                'volatility': [data['Close'][-20:].std()],
                'returns': [(data['Close'][-1] - data['Close'][-2]) / data['Close'][-2]]
            })

            # Predict price movement
            predicted_movement = self.model.predict(latest_data)[0]

            # Adjust TP based on predicted movement
            if direction == 'buy':
                tp = price + abs(predicted_movement) * 3  # Aggressive TP
            else:
                tp = price - abs(predicted_movement) * 3
            return tp
        except Exception as e:
            self.logger.error(f"Error calculating dynamic TP: {str(e)}")
            return price + 0.02 if direction == 'buy' else price - 0.02