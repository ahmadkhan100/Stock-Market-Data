import unittest
from src.data_preprocessor import DataPreprocessor
import pandas as pd

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # This will run before each test
        self.data = pd.DataFrame({
            'Open': [100, 101, 102, None, 105],
            'High': [110, 111, None, 113, 115],
            'Low': [90, 91, 92, 93, None],
            'Close': [105, None, 108, 109, 112],
            'Volume': [1000, None, 1200, 1300, 1400]
        })
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.processor = DataPreprocessor("", self.features)

    def test_fill_na(self):
        # Test forward fill
        processed_data = self.processor.preprocess_data(self.data.copy())
        self.assertFalse(processed_data.isnull().any().any(), "Should not have any NaNs after preprocessing")

    def test_feature_scaling(self):
        # Test that features are scaled
        processed_data = self.processor.preprocess_data(self.data.fillna(method='ffill'))
        for feature in self.features:
            column = processed_data[feature]
            # Check if column mean is close to 0 (scaled data property)
            self.assertTrue(abs(column.mean()) < 1e-6, f"Mean of {feature} should be close to 0 after scaling")
            # Check if column std is close to 1 (scaled data property)
            self.assertTrue(abs(column.std() - 1) < 1e-6, f"Std of {feature} should be close to 1 after scaling")

if __name__ == '__main__':
    unittest.main()
