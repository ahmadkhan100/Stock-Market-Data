import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

class DataPreprocessor:
    """
    Class to handle data preprocessing for stock market data.

    Attributes:
        data_path (str): Path to the stock market dataset file.
        features (list): List of feature column names to process.
    """
    def __init__(self, data_path, features):
        self.data_path = data_path
        self.features = features

    def load_data(self):
        """
        Load data from a CSV file.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame.
        """
        try:
            logging.info(f"Loading data from {self.data_path}")
            return pd.read_csv(self.data_path)
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def preprocess_data(self, df):
        """
        Preprocesses the stock market data by filling missing values and scaling.

        Args:
            df (pd.DataFrame): The DataFrame containing stock market data.

        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        logging.info("Handling missing values with forward fill")
        df.fillna(method='ffill', inplace=True)

        logging.info("Scaling numerical features")
        scaler = StandardScaler()
        df[self.features] = scaler.fit_transform(df[self.features])

        return df
