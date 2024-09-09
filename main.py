import logging
from stock_preprocessor import StockPreprocessor
from config import Config

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='stock_preprocessing.log'
    )

def main():
    """Main function to run the stock market data preprocessing pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting stock market data preprocessing pipeline")

    try:
        config = Config()
        preprocessor = StockPreprocessor(config)
        X, y, preprocessed_df = preprocessor.preprocess()

        logger.info(f"Preprocessed data shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Preprocessed dataframe head:\n{preprocessed_df.head()}")

        # Save preprocessed data
        preprocessed_df.to_csv(config.output_file)
        logger.info(f"Preprocessed data saved to {config.output_file}")

    except Exception as e:
        logger.exception(f"An error occurred during preprocessing: {str(e)}")

if __name__ == "__main__":
    main()

# File: config.py

class Config:
    """Configuration class for stock market data preprocessing."""

    def __init__(self):
        self.input_file = "data/stock_data.csv"
        self.output_file = "data/preprocessed_stock_data.csv"
        self.sequence_length = 10
        self.sma_window = 20
        self.rsi_window = 14

# File: stock_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

class StockPreprocessor:
    """A class for preprocessing stock market data."""

    def __init__(self, config):
        """
        Initialize the StockPreprocessor.

        Args:
            config (Config): Configuration object containing preprocessing parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

    def preprocess(self):
        """
        Main method to preprocess stock market data.

        Returns:
            tuple: X (np.array), y (np.array), preprocessed_df (pd.DataFrame)
        """
        df = self._load_data()
        df = self._handle_missing_values(df)
        df = self._calculate_returns(df)
        df = self._add_technical_indicators(df)
        df = df.dropna()
        df = self._normalize_features(df)
        X, y = self._create_sequences(df)
        return X, y, df

    def _load_data(self):
        """Load the stock market data from a CSV file."""
        self.logger.info(f"Loading data from {self.config.input_file}")
        return pd.read_csv(self.config.input_file, parse_dates=['Date'], index_col='Date')

    def _handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        self.logger.info("Handling missing values")
        df = df.ffill().bfill()
        return df

    def _calculate_returns(self, df):
        """Calculate daily returns."""
        self.logger.info("Calculating daily returns")
        df['Returns'] = df['Close'].pct_change()
        return df

    def _add_technical_indicators(self, df):
        """Add technical indicators to the dataset."""
        self.logger.info("Adding technical indicators")
        
        # Simple Moving Average
        df['SMA'] = df['Close'].rolling(window=self.config.sma_window).mean()
        
        # Relative Strength Index
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def _normalize_features(self, df):
        """Normalize numerical features using Min-Max scaling."""
        self.logger.info("Normalizing features")
        scaler = MinMaxScaler()
        columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA', 'RSI']
        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
        return df

    def _create_sequences(self, df):
        """Create sequences for time series forecasting."""
        self.logger.info("Creating sequences for time series forecasting")
        sequences = []
        targets = []
        for i in range(len(df) - self.config.sequence_length):
            seq = df.iloc[i:i+self.config.sequence_length]
            target = df.iloc[i+self.config.sequence_length]['Close']
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)
