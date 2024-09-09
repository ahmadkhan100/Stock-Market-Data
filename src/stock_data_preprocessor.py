# src/stock_data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer
import logging
from typing import Tuple, List

class StockPreprocessor:
    """
    A comprehensive class for preprocessing stock market data.
    
    This class provides methods for loading, cleaning, transforming, and preparing
    stock market data for further analysis or machine learning tasks.
    """

    def __init__(self, config):
        """
        Initialize the StockPreprocessor.

        Args:
            config (Config): Configuration object containing preprocessing parameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Main method to preprocess stock market data.

        Returns:
            tuple: X (np.ndarray), y (np.ndarray), preprocessed_df (pd.DataFrame)
        """
        df = self._load_data()
        df = self._handle_missing_values(df)
        df = self._handle_outliers(df)
        df = self._calculate_returns(df)
        df = self._add_technical_indicators(df)
        df = self._add_fundamental_indicators(df)
        df = self._feature_engineering(df)
        df = self._normalize_features(df)
        X, y = self._create_sequences(df)
        return X, y, df

    def _load_data(self) -> pd.DataFrame:
        """Load the stock market data from a CSV file."""
        self.logger.info(f"Loading data from {self.config.input_file}")
        df = pd.read_csv(self.config.input_file, parse_dates=['Date'], index_col='Date')
        self.logger.info(f"Loaded data shape: {df.shape}")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset using forward fill, backward fill,
        and KNN imputation for any remaining missing values.
        """
        self.logger.info("Handling missing values")
        
        # First, use forward fill and backward fill
        df = df.ffill().bfill()
        
        # If there are still missing values, use KNN imputation
        if df.isnull().sum().sum() > 0:
            self.logger.info("Using KNN imputation for remaining missing values")
            df = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns, index=df.index)
        
        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using Interquartile Range (IQR) method.
        """
        self.logger.info("Handling outliers")
        
        for column in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower_bound, upper_bound)
        
        return df

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily and cumulative returns."""
        self.logger.info("Calculating returns")
        df['Daily_Return'] = df['Close'].pct_change()
        df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        self.logger.info("Adding technical indicators")
        
        # Simple Moving Average
        df['SMA'] = df['Close'].rolling(window=self.config.sma_window).mean()
        
        # Exponential Moving Average
        df['EMA'] = df['Close'].ewm(span=self.config.ema_span, adjust=False).mean()
        
        # Relative Strength Index
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.rsi_window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        return df

    def _add_fundamental_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fundamental indicators to the dataset.
        Note: This is a placeholder method. In a real-world scenario, you would
        need to obtain fundamental data from financial statements or APIs.
        """
        self.logger.info("Adding fundamental indicators")
        
        # Placeholder for Price-to-Earnings ratio
        df['P/E_Ratio'] = np.random.uniform(10, 30, size=len(df))
        
        # Placeholder for Price-to-Book ratio
        df['P/B_Ratio'] = np.random.uniform(1, 5, size=len(df))
        
        return df

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering to create new features."""
        self.logger.info("Performing feature engineering")
        
        # Volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=self.config.volatility_window).std()
        
        # Price momentum
        df['Price_Momentum'] = df['Close'] / df['Close'].shift(self.config.momentum_window) - 1
        
        # Volume momentum
        df['Volume_Momentum'] = df['Volume'] / df['Volume'].shift(self.config.momentum_window) - 1
        
        # Day of week
        df['Day_of_Week'] = df.index.dayofweek
        
        # Is month end
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using Min-Max scaling."""
        self.logger.info("Normalizing features")
        
        columns_to_normalize = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Cumulative_Return',
            'SMA', 'EMA', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'P/E_Ratio', 'P/B_Ratio', 'Volatility', 'Price_Momentum', 'Volume_Momentum'
        ]
        
        df[columns_to_normalize] = self.scaler.fit_transform(df[columns_to_normalize])
        return df

    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_feature_names(self) -> List[str]:
        """Return a list of feature names used in the preprocessed data."""
        return self.scaler.feature_names_in_.tolist()

    def inverse_transform_close(self, normalized_close: np.ndarray) -> np.ndarray:
        """
        Inverse transform the normalized closing prices back to their original scale.

        Args:
            normalized_close (np.ndarray): Array of normalized closing prices.

        Returns:
            np.ndarray: Array of closing prices in their original scale.
        """
        close_index = self.get_feature_names().index('Close')
        return self.scaler.inverse_transform(np.zeros((len(normalized_close), len(self.get_feature_names()))))[:, close_index]

    def save_preprocessed_data(self, df: pd.DataFrame):
        """Save the preprocessed data to a CSV file."""
        self.logger.info(f"Saving preprocessed data to {self.config.output_file}")
        df.to_csv(self.config.output_file)
        self.logger.info("Preprocessed data saved successfully")

if __name__ == "__main__":
    # This block allows for basic testing of the StockPreprocessor class
    from config import Config
    
    logging.basicConfig(level=logging.INFO)
    config = Config()
    preprocessor = StockPreprocessor(config)
    X, y, df = preprocessor.preprocess()
    
    print(f"Preprocessed data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Preprocessed dataframe columns: {df.columns.tolist()}")
    print(f"Preprocessed dataframe head:\n{df.head()}")
    
    preprocessor.save_preprocessed_data(df)
