import logging
from src.data_preprocessor import DataPreprocessor

def setup_logging():
    logging.basicConfig(filename='logs/preprocessing.log',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

def main():
    setup_logging()
    DATA_PATH = 'data/example_stock_data.csv'
    FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    logging.info("Initializing the data preprocessor.")
    processor = DataPreprocessor(DATA_PATH, FEATURES)
    
    logging.info("Loading data.")
    data = processor.load_data()
    
    logging.info("Starting data preprocessing.")
    processed_data = processor.preprocess_data(data)
    
    logging.info("Preprocessing complete. Displaying first few rows of the processed data.")
    print(processed_data.head())

if __name__ == "__main__":
    main()
