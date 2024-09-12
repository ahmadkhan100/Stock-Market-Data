# src/stock_data_preprocessor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
import logging
from typing import Tuple, List, Optional


class Config:
    """
    Configuration class containing preprocessing parameters.
    Adjust these parameters according to your dataset and requirements.
    """
    # File paths
    input_file: str = 'data/stock_data.csv'
    output_file: str = 'data/preprocessed_stock_data.csv'
    
    # Technical indicator parameters
    sma_window: int = 20
    ema_span: int = 20
    rsi_window: int = 14
    volatility_window: int = 20
    momentum_window: int = 5
    sequence_length: int = 60
    
    # KNN Imputer parameters
    knn_neighbors: int = 5


class StockPreprocessor:
    """
    A comprehensive class for preprocessing stock market data.

    This class provides methods for loading, cleaning, transforming, and preparing
    stock market data for further analysis or machine learning tasks.
    """

    def __init__(self, config: Config):
        """
        Initialize the StockPreprocessor.

        Args:
            config (Config): Configuration object containing preprocessing parameters.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scalers = {}
        self.imputer = KNNImputer(n_neighbors=self.config.knn_neighbors)

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
        df = df.dropna()  # Drop rows with NaN values after processing
        X, y = self._create_sequences(df)
        return X, y, df

    def _load_data(self) -> pd.DataFrame:
        """Load the stock market data from a CSV file."""
        try:
            self.logger.info(f"Loading data from {self.config.input_file}")
            df = pd.read_csv(self.config.input_file, parse_dates=['Date'], index_col='Date')
            self.logger.info(f"Loaded data shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset using forward fill, backward fill,
        and KNN imputation for any remaining missing values.
        """
        self.logger.info("Handling missing values")
        try:
            # First, use forward fill and backward fill
            df = df.ffill().bfill()

            # If there are still missing values, use KNN imputation
            if df.isnull().sum().sum() > 0:
                self.logger.info("Using KNN imputation for remaining missing values")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_numeric = df[numeric_cols]
                df_numeric = pd.DataFrame(
                    self.imputer.fit_transform(df_numeric),
                    columns=df_numeric.columns,
                    index=df_numeric.index
                )
                df.update(df_numeric)
            return df
        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}")
            raise

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using Interquartile Range (IQR) method.
        """
        self.logger.info("Handling outliers")
        try:
            for column in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower_bound, upper_bound)
            return df
        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}")
            raise

    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily and cumulative returns."""
        self.logger.info("Calculating returns")
        try:
            df['Daily_Return'] = df['Close'].pct_change()
            df['Cumulative_Return'] = (1 + df['Daily_Return']).cumprod() - 1
            return df
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataset."""
        self.logger.info("Adding technical indicators")
        try:
            # Simple Moving Average
            df['SMA'] = df['Close'].rolling(window=self.config.sma_window).mean()

            # Exponential Moving Average
            df['EMA'] = df['Close'].ewm(span=self.config.ema_span, adjust=False).mean()

            # Relative Strength Index
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=self.config.rsi_window, min_periods=1).mean()
            avg_loss = loss.rolling(window=self.config.rsi_window, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Moving Average Convergence Divergence (MACD)
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Std'] = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']

            return df
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            raise

    def _add_fundamental_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fundamental indicators to the dataset.
        Note: This is a placeholder method. In a real-world scenario, you would
        need to obtain fundamental data from financial statements or APIs.
        """
        self.logger.info("Adding fundamental indicators")
        try:
            # Placeholder for Price-to-Earnings ratio
            df['P/E_Ratio'] = np.random.uniform(10, 30, size=len(df))

            # Placeholder for Price-to-Book ratio
            df['P/B_Ratio'] = np.random.uniform(1, 5, size=len(df))

            return df
        except Exception as e:
            self.logger.error(f"Error adding fundamental indicators: {e}")
            raise

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering to create new features."""
        self.logger.info("Performing feature engineering")
        try:
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
        except Exception as e:
            self.logger.error(f"Error during feature engineering: {e}")
            raise

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using Min-Max scaling."""
        self.logger.info("Normalizing features")
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                scaler = MinMaxScaler()
                df[[col]] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            return df
        except Exception as e:
            self.logger.error(f"Error normalizing features: {e}")
            raise

    def _create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series forecasting."""
        self.logger.info("Creating sequences for time series forecasting")
        try:
            sequences = []
            targets = []
            feature_cols = df.columns.tolist()
            target_col = 'Close'

            for i in range(len(df) - self.config.sequence_length):
                seq = df.iloc[i:i + self.config.sequence_length][feature_cols].values
                target = df.iloc[i + self.config.sequence_length][target_col]
                sequences.append(seq)
                targets.append(target)

            return np.array(sequences), np.array(targets)
        except Exception as e:
            self.logger.error(f"Error creating sequences: {e}")
            raise

    def get_feature_names(self) -> List[str]:
        """Return a list of feature names used in the preprocessed data."""
        return list(self.scalers.keys())

    def inverse_transform_close(self, normalized_close: np.ndarray) -> np.ndarray:
        """
        Inverse transform the normalized closing prices back to their original scale.

        Args:
            normalized_close (np.ndarray): Array of normalized closing prices.

        Returns:
            np.ndarray: Array of closing prices in their original scale.
        """
        try:
            scaler = self.scalers.get('Close')
            if scaler is not None:
                return scaler.inverse_transform(normalized_close.reshape(-1, 1)).flatten()
            else:
                raise ValueError("Scaler for 'Close' not found. Ensure 'Close' was normalized.")
        except Exception as e:
            self.logger.error(f"Error in inverse transforming 'Close': {e}")
            raise

    def save_preprocessed_data(self, df: pd.DataFrame):
        """Save the preprocessed data to a CSV file."""
        try:
            self.logger.info(f"Saving preprocessed data to {self.config.output_file}")
            df.to_csv(self.config.output_file)
            self.logger.info("Preprocessed data saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving preprocessed data: {e}")
            raise


if __name__ == "__main__":
    # This block allows for basic testing of the StockPreprocessor class

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = Config()
    preprocessor = StockPreprocessor(config)
    X, y, df = preprocessor.preprocess()

    print(f"Preprocessed data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Preprocessed dataframe columns: {df.columns.tolist()}")
    print(f"Preprocessed dataframe head:\n{df.head()}")

    preprocessor.save_preprocessed_data(df)
