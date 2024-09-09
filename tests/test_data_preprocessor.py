# test_stock_preprocessor.py

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.stock_preprocessor import StockPreprocessor
from src.config import Config

class TestStockPreprocessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        self.config.input_file = 'test_data.csv'
        self.config.output_file = 'test_output.csv'
        self.config.sequence_length = 5
        self.config.sma_window = 3
        self.config.ema_span = 3
        self.config.rsi_window = 3
        self.config.volatility_window = 3
        self.config.momentum_window = 3

        self.preprocessor = StockPreprocessor(self.config)

        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=10)
        self.sample_data = pd.DataFrame({
            'Date': dates,
            'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'Low': [98, 99, 100, 101, 102, 103, 104, 105, 106, 107],
            'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        self.sample_data.set_index('Date', inplace=True)
        self.sample_data.to_csv(self.config.input_file)

    def tearDown(self):
        """Clean up test fixtures"""
        import os
        if os.path.exists(self.config.input_file):
            os.remove(self.config.input_file)
        if os.path.exists(self.config.output_file):
            os.remove(self.config.output_file)

    def test_load_data(self):
        """Test if data is loaded correctly"""
        df = self.preprocessor._load_data()
        self.assertEqual(len(df), 10)
        self.assertEqual(list(df.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_handle_missing_values(self):
        """Test handling of missing values"""
        df = self.sample_data.copy()
        df.loc['2023-01-05', 'Close'] = np.nan
        df = self.preprocessor._handle_missing_values(df)
        self.assertFalse(df.isnull().any().any())

    def test_handle_outliers(self):
        """Test handling of outliers"""
        df = self.sample_data.copy()
        df.loc['2023-01-05', 'Close'] = 1000  # Introduce an outlier
        df = self.preprocessor._handle_outliers(df)
        self.assertLess(df.loc['2023-01-05', 'Close'], 1000)

    def test_calculate_returns(self):
        """Test calculation of returns"""
        df = self.preprocessor._calculate_returns(self.sample_data)
        self.assertIn('Daily_Return', df.columns)
        self.assertIn('Cumulative_Return', df.columns)

    def test_add_technical_indicators(self):
        """Test addition of technical indicators"""
        df = self.preprocessor._add_technical_indicators(self.sample_data)
        self.assertIn('SMA', df.columns)
        self.assertIn('EMA', df.columns)
        self.assertIn('RSI', df.columns)
        self.assertIn('MACD', df.columns)
        self.assertIn('Signal_Line', df.columns)
        self.assertIn('BB_Upper', df.columns)
        self.assertIn('BB_Middle', df.columns)
        self.assertIn('BB_Lower', df.columns)

    def test_add_fundamental_indicators(self):
        """Test addition of fundamental indicators"""
        df = self.preprocessor._add_fundamental_indicators(self.sample_data)
        self.assertIn('P/E_Ratio', df.columns)
        self.assertIn('P/B_Ratio', df.columns)

    def test_feature_engineering(self):
        """Test feature engineering"""
        df = self.preprocessor._feature_engineering(self.sample_data)
        self.assertIn('Volatility', df.columns)
        self.assertIn('Price_Momentum', df.columns)
        self.assertIn('Volume_Momentum', df.columns)
        self.assertIn('Day_of_Week', df.columns)
        self.assertIn('Is_Month_End', df.columns)

    def test_normalize_features(self):
        """Test normalization of features"""
        df = self.preprocessor._normalize_features(self.sample_data)
        self.assertTrue((df['Close'] >= 0).all() and (df['Close'] <= 1).all())

    def test_create_sequences(self):
        """Test creation of sequences"""
        X, y = self.preprocessor._create_sequences(self.sample_data)
        self.assertEqual(X.shape, (5, 5, 5))  # (n_samples, sequence_length, n_features)
        self.assertEqual(y.shape, (5,))

    def test_full_preprocessing(self):
        """Test the full preprocessing pipeline"""
        X, y, df = self.preprocessor.preprocess()
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(X.shape[1], self.config.sequence_length)

    def test_get_feature_names(self):
        """Test getting feature names"""
        self.preprocessor.preprocess()  # Run preprocessing to populate feature names
        feature_names = self.preprocessor.get_feature_names()
        self.assertIsInstance(feature_names, list)
        self.assertIn('Close', feature_names)

    def test_inverse_transform_close(self):
        """Test inverse transform of closing prices"""
        X, y, df = self.preprocessor.preprocess()
        original_close = self.sample_data['Close'].values[-len(y):]
        inverse_close = self.preprocessor.inverse_transform_close(y)
        np.testing.assert_array_almost_equal(original_close, inverse_close, decimal=2)

    def test_save_preprocessed_data(self):
        """Test saving preprocessed data"""
        _, _, df = self.preprocessor.preprocess()
        self.preprocessor.save_preprocessed_data(df)
        self.assertTrue(os.path.exists(self.config.output_file))

if __name__ == '__main__':
    unittest.main()
